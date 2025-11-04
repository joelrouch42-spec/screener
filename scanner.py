#!/usr/bin/env python3
"""
Stock Scanner - Surveillance continue sans GUI
Détecte les catalyseurs et affiche les alertes en temps réel
Version CLI du système de détection de catalyseurs
"""

import logging
import sys
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from threading import Thread, Event
import signal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

# Local imports
from data_providers import MultiSourceDataProvider
from catalyst_analyzer import CatalystAnalyzer, load_settings
from tabs import get_latest_data_with_cached_levels

# Timezone
EST = ZoneInfo("America/New_York")

# ----------------------
# Logging
# ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------------
# Support/Resistance Detection for Scanner
# ----------------------
def detect_scanner_breakouts(df: pd.DataFrame, support_levels: List[float], resistance_levels: List[float], settings: Dict = None) -> Optional[Dict]:
    """Detect support/resistance breakouts on latest candle - VERSION SCANNER"""
    if len(df) < 5:
        return None
        
    # Current and previous candle
    current = df.iloc[-1]
    previous = df.iloc[-2]
    
    current_high = current['High']
    current_low = current['Low']
    current_close = current['Close']
    prev_close = previous['Close']
    
    # Calcul du % de mouvement
    change_pct = ((current_close - prev_close) / prev_close) * 100
    
    # Paramètres configurables
    if settings:
        min_move = settings.get("breakout", {}).get("min_move_percent", 1.0)
        vol_threshold = settings.get("breakout", {}).get("volume_spike_threshold", 1.5)
    else:
        min_move = 1.0
        vol_threshold = 1.5
    
    # Volume spike check (si disponible)
    volume_spike = False
    if 'Volume' in df.columns and len(df) >= 5:
        current_vol = current['Volume']
        avg_vol = df['Volume'].iloc[-5:].mean()
        volume_spike = current_vol / avg_vol >= vol_threshold if avg_vol > 0 else False
    
    # Check resistance breakouts
    for resistance in resistance_levels:
        if (prev_close < resistance and current_high > resistance and 
            current_close > resistance and  # Close au-dessus aussi
            change_pct >= min_move and  # Seuil configurable
            volume_spike):  # Volume élevé
            return {
                'type': 'resistance_breakout',
                'level': float(resistance),
                'direction': 'UP',
                'description': f'Breakout résistance à ${resistance:.2f}'
            }
    
    # Check support breakdowns  
    for support in support_levels:
        if (prev_close > support and current_low < support and
            current_close < support and  # Close en dessous aussi
            change_pct <= -min_move and  # Seuil configurable
            volume_spike):  # Volume élevé
            return {
                'type': 'support_breakdown',
                'level': float(support),
                'direction': 'DOWN', 
                'description': f'Breakdown support à ${support:.2f}'
            }
            
    return None

class StockScanner:
    def __init__(self, config_file: str = 'config.txt'):
        self.config_file = config_file
        self.symbols_config = {}
        self.sector_map = {}
        self.data_provider = None
        self.catalyst_analyzer = None
        self.settings = {}
        self.is_running = False
        self.stop_event = Event()
        self.alerts_count = 0
        
        # Historique des alertes pour éviter les doublons
        self.recent_alerts = {}
        
    def is_market_open(self, dt: Optional[datetime] = None) -> bool:
        """Vérifie si le marché US est ouvert"""
        dt = dt or datetime.now(EST)
        
        # Weekend
        if dt.weekday() > 4:  # Lundi=0, Vendredi=4
            return False
        
        # Heures d'ouverture : 9h30-16h EST
        market_open = dt.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = dt.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= dt <= market_close
        
    def load_configuration(self):
        """Charge la configuration des symboles et secteurs"""
        # Charger les settings
        self.settings = load_settings('settings.json')
        
        # Contrôle des printouts
        self.alerts_only = self.settings.get("logging", {}).get("alerts_only", False)
        
        if not self.alerts_only:
            logger.info(f"📋 Settings: refresh={self.settings['interface']['refresh_seconds']}s")
        
        # Charger les symboles
        self.symbols_config = self._parse_config(self.config_file)
        if not self.symbols_config:
            logger.error('❌ Aucun symbole dans la configuration!')
            return False
            
        # Charger le mapping secteurs
        self.sector_map = self._load_sector_mapping('sector_mapping.txt')
        
        if not self.alerts_only:
            logger.info(f"📊 {len(self.symbols_config)} symboles à surveiller: {list(self.symbols_config.keys())}")
        return True
        
    def _parse_config(self, config_file: str) -> Dict[str, dict]:
        """Parse le fichier de configuration"""
        symbols_config = {}
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 1:
                        symbol = parts[0].upper()
                        provider = parts[1].lower() if len(parts) >= 2 else 'auto'
                        symbols_config[symbol] = {'provider': provider}
        except Exception as e:
            logger.exception(f'❌ Erreur lecture config: {e}')
        return symbols_config
        
    def _load_sector_mapping(self, file: str) -> Dict[str, str]:
        """Charge le mapping symbole -> secteur"""
        sector_map = {}
        try:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        sector_map[parts[0].upper()] = parts[1].lower()
        except Exception:
            logger.warning('⚠️  Erreur lecture sector mapping')
        return sector_map
        
    def initialize_services(self):
        """Initialise les services de données et d'analyse"""
        try:
            # Passer debug=False sauf si mode debug activé
            debug_mode = not self.alerts_only
            self.data_provider = MultiSourceDataProvider(debug=debug_mode)
            if not self.alerts_only:
                logger.info("✅ Data provider initialisé")
        except Exception as e:
            logger.exception('❌ ERREUR INITIALISATION DATA PROVIDER')
            return False
            
        try:
            self.catalyst_analyzer = CatalystAnalyzer(settings=self.settings)
            if not self.alerts_only:
                logger.info("✅ Catalyst analyzer initialisé")
            
            # Désactiver les logs verbeux si alerts_only
            if self.alerts_only:
                logging.getLogger('tabs').setLevel(logging.WARNING)
                logging.getLogger('catalyst_analyzer').setLevel(logging.WARNING)
        except Exception as e:
            logger.exception('⚠️  Erreur initialisation Catalyst Analyzer')
            self.catalyst_analyzer = None
            
        return True
        
    def scan_symbol(self, symbol: str) -> Optional[Dict]:
        """Scan un symbole et retourne une alerte si détectée (VERSION OPTIMISÉE)"""
        try:
            # Vérifier si le marché est ouvert
            if not self.is_market_open():
                return None  # Pas d'analyse si marché fermé
                
            sector = self.sector_map.get(symbol, 'unknown')
            
            # NOUVELLE API OPTIMISÉE : récupérer dernière bougie + niveaux S/R en cache
            data_result = get_latest_data_with_cached_levels(symbol, self.data_provider, self.settings)
            
            if data_result is None:
                logger.warning(f'⚠️  Impossible de récupérer les données pour {symbol}')
                return None
                
            latest_candle = data_result['latest_candle']
            support_levels = data_result['support_levels'] 
            resistance_levels = data_result['resistance_levels']
            provider_used = data_result['provider_used']
            cache_hit = data_result['cache_hit']
            
            if cache_hit:
                logger.debug(f"📦 {symbol}: Cache HIT S/R")
            else:
                logger.debug(f"🔄 {symbol}: Recalcul S/R")
            
            # Créer un mini-DataFrame pour les analyses (2 bougies : précédente + actuelle)
            # Pour l'instant on simule la bougie précédente
            current_price = float(latest_candle['Close'])
            
            # 1. Analyser les breakouts techniques avec niveaux en cache
            breakout_info = None
            if len(support_levels) > 0 or len(resistance_levels) > 0:
                # Pour détecter un breakout, on a besoin de comparer avec la bougie précédente
                # Récupérons 2 bougies pour avoir prev + current
                try:
                    df_mini, _ = self.data_provider.fetch_data(symbol, days=2, period=2)
                    if df_mini is not None and len(df_mini) >= 2:
                        breakout_info = detect_scanner_breakouts(df_mini, support_levels, resistance_levels, self.settings)
                except Exception as e:
                    logger.debug(f"Erreur mini-fetch pour breakout {symbol}: {e}")
            
            # 2. Analyser avec catalyst analyzer (nécessite un DataFrame complet)
            catalyst_info = None
            if self.catalyst_analyzer:
                try:
                    # Le catalyst analyzer a besoin d'un historique pour calculate_average_move
                    df_full, _ = self.data_provider.fetch_data(
                        symbol,
                        days=self.settings["data"]["days_fetch"], 
                        period=self.settings["data"]["period_candles"]
                    )
                    if df_full is not None and len(df_full) >= 2:
                        catalyst_info = self.catalyst_analyzer.analyze_symbol(symbol, df_full, sector)
                except Exception as e:
                    logger.debug(f"Erreur catalyst pour {symbol}: {e}")
            
            # Priorité: Catalyst (IA) > Breakout technique
            alert_info = catalyst_info or breakout_info
            
            if alert_info:
                # Déterminer le type d'alerte
                alert_type = alert_info.get('type', 'unknown')
                is_technical = alert_type in ['resistance_breakout', 'support_breakdown']
                
                # Éviter les alertes en double
                alert_key = f"{symbol}_{alert_type}"
                now = datetime.now()
                
                # Vérifier si alerte récente (dernières 2 heures pour catalyst, 30 min pour technique)
                cooldown = timedelta(minutes=30) if is_technical else timedelta(hours=2)
                if alert_key in self.recent_alerts:
                    time_diff = now - self.recent_alerts[alert_key]
                    if time_diff < cooldown:
                        return None  # Alerte trop récente
                
                # Enregistrer cette alerte
                self.recent_alerts[alert_key] = now
                
                # Pour previous_price, utiliser les 2 dernières bougies si disponibles
                try:
                    df_price, _ = self.data_provider.fetch_data(symbol, days=2, period=2)
                    if df_price is not None and len(df_price) >= 2:
                        previous_price = float(df_price['Close'].iloc[-2])
                    else:
                        previous_price = current_price * 0.99  # Estimation
                except:
                    previous_price = current_price * 0.99  # Fallback
                
                # Créer l'alerte
                alert = {
                    'symbol': symbol,
                    'sector': sector,
                    'timestamp': now.isoformat(),
                    'catalyst': alert_info,
                    'provider': provider_used,
                    'current_price': current_price,
                    'previous_price': previous_price,
                    'is_technical': is_technical,
                    'support_levels': support_levels[-3:] if len(support_levels) > 0 else [],
                    'resistance_levels': resistance_levels[-3:] if len(resistance_levels) > 0 else [],
                    'cache_performance': 'HIT' if cache_hit else 'MISS'
                }
                
                return alert
                    
        except Exception as e:
            logger.error(f'❌ Erreur scan {symbol}: {e}')
            
        return None
        
    def display_alert(self, alert: Dict):
        """Affiche une alerte formatée"""
        symbol = alert['symbol']
        catalyst = alert['catalyst']
        current_price = alert['current_price']
        previous_price = alert['previous_price']
        is_technical = alert.get('is_technical', False)
        
        change_pct = ((current_price - previous_price) / previous_price) * 100
        change_direction = "📈" if change_pct > 0 else "📉"
        
        # Type d'alerte
        alert_icon = "🔧" if is_technical else "🚨"
        alert_title = "BREAKOUT TECHNIQUE" if is_technical else "CATALYST DÉTECTÉ"
        
        print("\n" + "="*80)
        print(f"{alert_icon} {alert_title} - {datetime.now().strftime('%H:%M:%S')}")
        print("="*80)
        print(f"📊 Symbole: {symbol} ({alert['sector'].upper()})")
        print(f"💰 Prix: ${current_price:.2f} ({change_direction} {change_pct:+.2f}%)")
        print(f"🔍 Type: {catalyst.get('type', 'N/A').upper().replace('_', ' ')}")
        print(f"📝 Description: {catalyst.get('description', 'N/A')}")
        
        if is_technical:
            # Affichage spécial pour les breakouts
            level = catalyst.get('level', 0)
            direction = catalyst.get('direction', 'N/A')
            print(f"📏 Niveau: ${level:.2f}")
            print(f"➡️  Direction: {direction}")
            print(f"⚡ Signal: TECHNIQUE")
            
            # Afficher les niveaux proches
            if alert.get('support_levels'):
                supports = [f"${s:.2f}" for s in alert['support_levels']]
                print(f"🟢 Supports: {', '.join(supports)}")
            if alert.get('resistance_levels'):
                resistances = [f"${r:.2f}" for r in alert['resistance_levels']]
                print(f"🔴 Résistances: {', '.join(resistances)}")
        else:
            # Affichage pour les catalysts IA
            print(f"⭐ Fiabilité: {catalyst.get('reliability', 'N/A').upper()}")
            print(f"💼 Tradeable: {'✅ OUI' if catalyst.get('tradeable', False) else '❌ NON'}")
            print(f"🤖 Signal: INTELLIGENCE ARTIFICIELLE")
            
        print(f"📡 Source: {alert['provider']}")
        print("="*80)
        
        # Log pour historique
        signal_type = "TECHNIQUE" if is_technical else "IA"
        logger.info(f"{alert_icon} ALERTE {signal_type}: {symbol} {change_pct:+.2f}% - {catalyst.get('type', 'N/A')}")
        
        self.alerts_count += 1
        
    def scan_all_symbols(self):
        """Scan tous les symboles une fois"""
        # Vérifier d'abord si le marché est ouvert
        if not self.is_market_open():
            current_time = datetime.now(EST).strftime('%H:%M:%S EST')
            if not self.alerts_only:
                print(f"⚠️  Marché fermé ({current_time}) - En attente d'ouverture (9h30-16h EST)")
            return 0
            
        alerts_found = 0
        
        for symbol in self.symbols_config.keys():
            if self.stop_event.is_set():
                break
                
            logger.debug(f"🔍 Scan {symbol}...")
            alert = self.scan_symbol(symbol)
            
            if alert:
                self.display_alert(alert)
                alerts_found += 1
                
            # Petite pause entre les symboles
            time.sleep(1)
            
        return alerts_found
        
    def run_continuous_scan(self):
        """Lance le scan en continu"""
        logger.info("🚀 Démarrage du scanner continu...")
        logger.info(f"⏱️  Intervalle: {self.settings['interface']['refresh_seconds']} secondes")
        logger.info("📊 Appuyez sur Ctrl+C pour arrêter")
        
        self.is_running = True
        scan_count = 0
        
        while self.is_running and not self.stop_event.is_set():
            scan_count += 1
            start_time = time.time()
            
            if not self.alerts_only:
                print(f"\n🔄 Scan #{scan_count} - {datetime.now().strftime('%H:%M:%S')}")
                print("-" * 50)
            
            alerts_found = self.scan_all_symbols()
            
            scan_duration = time.time() - start_time
            
            if not self.alerts_only:
                if alerts_found == 0:
                    print(f"✅ Scan terminé - Aucune alerte ({scan_duration:.1f}s)")
                else:
                    print(f"🚨 Scan terminé - {alerts_found} alerte(s) détectée(s) ({scan_duration:.1f}s)")
                    
                print(f"📈 Total alertes depuis le début: {self.alerts_count}")
            
            # Attendre avant le prochain scan
            if not self.stop_event.wait(self.settings['interface']['refresh_seconds']):
                continue
            else:
                break
                
        logger.info("🛑 Scanner arrêté")
        
    def stop(self):
        """Arrête le scanner"""
        self.is_running = False
        self.stop_event.set()
        
    def cleanup_old_alerts(self):
        """Nettoie les anciennes alertes (plus de 24h)"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        keys_to_remove = []
        
        for key, timestamp in self.recent_alerts.items():
            if timestamp < cutoff_time:
                keys_to_remove.append(key)
                
        for key in keys_to_remove:
            del self.recent_alerts[key]

def signal_handler(signum, frame, scanner):
    """Gestionnaire de signal pour arrêt propre"""
    print("\n🛑 Arrêt demandé...")
    scanner.stop()

def main():
    print("📡 STOCK SCANNER - Surveillance Continue")
    print("=====================================")
    
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'config.txt'
    
    # Créer le scanner
    scanner = StockScanner(config_file)
    
    # Gestionnaire d'arrêt propre
    signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, scanner))
    signal.signal(signal.SIGTERM, lambda s, f: signal_handler(s, f, scanner))
    
    # Charger la configuration
    if not scanner.load_configuration():
        sys.exit(1)
        
    # Initialiser les services
    if not scanner.initialize_services():
        sys.exit(1)
        
    try:
        # Nettoyer les anciennes alertes au démarrage
        scanner.cleanup_old_alerts()
        
        # Lancer le scan continu
        scanner.run_continuous_scan()
        
    except KeyboardInterrupt:
        print("\n⌨️  Interruption détectée")
    except Exception as e:
        logger.exception(f"❌ Erreur fatale: {e}")
    finally:
        scanner.stop()
        print("✅ Scanner fermé proprement")

if __name__ == '__main__':
    main()