import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo
import json
import requests
import logging

logger = logging.getLogger(__name__)
EST = ZoneInfo("America/New_York")

def load_settings(settings_file: str = 'settings.json') -> Dict:
    """Charge les paramètres depuis le fichier JSON"""
    default_settings = {
        "historique": {"nb_bougies": 30},
        "interface": {"refresh_seconds": 60, "chart_height": 900},
        "catalyst": {"multiplier": 1.5, "min_threshold": 1.0, "news_days": 1, "recent_alerts_days": 7, "catalyst_count_days": 30, "catalyst_log_days": 365},
        "data": {"days_fetch": 150, "period_candles": 120, "cache_minutes": 1},
        "analysis": {"avg_candles": 30, "volume_min_candles": 5, "volume_spike_threshold": 1.5},
        "market": {"open_hour": 9, "open_minute": 30, "close_hour": 16, "close_minute": 0},
        "api": {"timeout_seconds": 30},
        "support_resistance": {"order": 5, "cluster_threshold": 0.02, "line_width": 2, "line_opacity": 0.6},
        "server": {"port": 8050, "host": "0.0.0.0", "debug": False},
        "chart": {"margin_top": 50, "margin_bottom": 50, "margin_left": 50, "margin_right": 50, "current_price_line_color": "yellow", "current_price_line_width": 2, "current_price_line_opacity": 0.8}
    }
    
    try:
        # Obtenir le chemin absolu pour debug
        abs_path = os.path.abspath(settings_file)
        print(f"🔍 DEBUG: Loading settings from: {abs_path}")

        with open(settings_file, 'r', encoding='utf-8') as f:
            settings = json.load(f)
            logger.info(f"✅ Settings chargés depuis {settings_file}")

            # DEBUG: Afficher la section debug
            if 'debug' in settings:
                print(f"🔍 DEBUG: settings.json contains debug section: {settings['debug']}")
            return settings
    except FileNotFoundError:
        logger.warning(f"⚠️  {settings_file} introuvable, utilisation valeurs par défaut")
        return default_settings
    except Exception as e:
        logger.warning(f"⚠️  Erreur lecture {settings_file}: {e}, utilisation valeurs par défaut")
        return default_settings

class CatalystAnalyzer:
    def __init__(self, api_key: Optional[str] = None, settings: Optional[Dict] = None):
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.current_catalysts = []
        
        # Charger settings ou utiliser defaults
        self.settings = settings or load_settings()
        self.avg_candles = self.settings["analysis"]["avg_candles"]
        self.multiplier = self.settings["catalyst"]["multiplier"] 
        self.min_threshold = self.settings["catalyst"]["min_threshold"]
        
        self.average_move_pct = None  # mémoire pour la moyenne calculée
        logger.info(f"✅ Catalyst Analyzer initialisé avec avg_candles={self.avg_candles}, multiplier={self.multiplier}")

    def calculate_average_move(self, df) -> float:
        """
        Calcule la moyenne des variations absolues (%) sur les dernières `self.avg_candles` bougies.
        """
        if len(df) < 2:
            return 0.0
        
        recent_df = df[-self.avg_candles:]
        abs_moves = [
            abs((recent_df['Close'].iloc[i] - recent_df['Close'].iloc[i-1]) / recent_df['Close'].iloc[i-1] * 100)
            for i in range(1, len(recent_df))
            if recent_df['Close'].iloc[i-1] != 0
        ]
        self.average_move_pct = sum(abs_moves) / len(abs_moves) if abs_moves else 0.0
        return self.average_move_pct
        
    def is_market_open(self, dt: Optional[datetime] = None) -> bool:
        """Vérifie si le marché est ouvert"""
        dt = dt or datetime.now(EST)
        
        # Fix: S'assurer que dt a le bon timezone
        if dt.tzinfo is None:
            dt = EST.localize(dt)
        elif dt.tzinfo != EST:
            dt = dt.astimezone(EST)
            
        weekday = dt.weekday()
        if weekday > 4:  # Weekend
            return False
            
        market_open = dt.replace(hour=self.settings["market"]["open_hour"], minute=self.settings["market"]["open_minute"], second=0, microsecond=0)
        market_close = dt.replace(hour=self.settings["market"]["close_hour"], minute=self.settings["market"]["close_minute"], second=0, microsecond=0)
        
        return market_open <= dt <= market_close

    def detect_significant_move(self, df) -> Tuple[bool, float, str]:
        """Détecte un mouvement significatif basé sur la moyenne historique"""
        if len(df) < 2:
            return False, 0.0, "NONE"
        
        current_price = df['Close'].iloc[-1]
        previous_price = df['Close'].iloc[-2]
        
        # Protection division par zéro
        if previous_price == 0:
            logger.warning("Prix précédent = 0, impossible de calculer %")
            return False, 0.0, "NONE"
            
        change_pct = ((current_price - previous_price) / previous_price) * 100
        
        # Calcul du seuil dynamique basé sur la moyenne
        avg_move = self.calculate_average_move(df)
        dynamic_threshold = avg_move * self.multiplier
        
        # Seuil minimum pour éviter les fausses alertes sur actions très stables
        dynamic_threshold = max(dynamic_threshold, self.min_threshold)
        
        is_significant = abs(change_pct) >= dynamic_threshold
        direction = "UP" if change_pct > 0 else "DOWN"
        
        logger.info(f"Mouvement: {change_pct:.2f}%, Moyenne: {avg_move:.2f}%, Seuil: {dynamic_threshold:.2f}%, Significatif: {is_significant}")
        
        return is_significant, change_pct, direction

    def get_news_context(self, symbol, days=1) -> List[str]:
        """Récupère les news récentes d’un symbole via yfinance"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            news = ticker.news
            if not news:
                return []

            cutoff = datetime.now(EST) - timedelta(days=days)
            recent_news = []
            for item in news[:5]:
                try:
                    news_date = datetime.fromtimestamp(item.get('providerPublishTime', 0), EST)
                    if news_date >= cutoff:
                        recent_news.append(item.get('title', 'No title'))
                except:
                    continue
            return recent_news
        except Exception as e:
            print(f"⚠️  Erreur récupération news pour {symbol}: {e}")
            return []

    def analyze_catalyst_with_ai(self, symbol, change_pct, sector, news, volume_spike) -> Dict:
        """Analyse un catalyseur via l'API Claude"""
        # Si pas d'API key, passer directement au fallback
        if not self.api_key:
            return self._fallback_analysis(change_pct, volume_spike, len(news))
            
        try:
            news_text = "\n".join(news) if news else "Aucune news récente"
            prompt = f"""Analyse ce mouvement boursier et retourne UNIQUEMENT un JSON valide:

Symbole: {symbol}
Secteur: {sector}  
Mouvement: {change_pct:+.2f}%
Volume élevé: {volume_spike}
News récentes: {news_text}

Retourne un JSON avec cette structure exacte:
{{
    "type": "earnings|technical|contract|macro|fda|acquisition|crypto_move|sector_rotation|short_squeeze|unknown",
    "description": "description courte max 80 caractères",
    "reliability": "high|medium|low", 
    "tradeable": true|false,
    "reasoning": "explication rapide"
}}
"""
            response = requests.post(
                'https://api.anthropic.com/v1/messages',
                headers={
                    'Content-Type': 'application/json',
                    'anthropic-version': '2023-06-01'
                },
                json={
                    'model': 'claude-sonnet-4-20250514',
                    'max_tokens': 1000,
                    'messages': [{'role': 'user', 'content': prompt}]
                },
                timeout=self.settings["api"]["timeout_seconds"]
            )

            if response.status_code == 200:
                try:
                    ai_response = response.json()['content'][0]['text']
                    # Clean JSON fences
                    if ai_response.startswith('```json'):
                        ai_response = ai_response[7:]
                    if ai_response.startswith('```'):
                        ai_response = ai_response[3:]
                    if ai_response.endswith('```'):
                        ai_response = ai_response.rsplit('\n', 1)[0]
                    catalyst_data = json.loads(ai_response)
                    return catalyst_data
                except json.JSONDecodeError:
                    print(f"⚠️  Erreur parsing JSON Claude pour {symbol}")
                    return self._fallback_analysis(change_pct, volume_spike, len(news))
            else:
                print(f"⚠️  Erreur API Claude ({response.status_code}) pour {symbol}")
                return self._fallback_analysis(change_pct, volume_spike, len(news))
        except Exception as e:
            print(f"⚠️  Exception API Claude pour {symbol}: {e}")
            return self._fallback_analysis(change_pct, volume_spike, len(news))

    def _fallback_analysis(self, change_pct, volume_spike, news_count) -> Dict:
        """Fallback si IA indisponible"""
        if abs(change_pct) > 10 and volume_spike:
            catalyst_type = "earnings" if news_count > 0 else "technical"
            reliability = "medium"
        elif volume_spike and news_count > 0:
            catalyst_type = "unknown"
            reliability = "low"
        else:
            catalyst_type = "technical"
            reliability = "low"

        return {
            "type": catalyst_type,
            "description": f"Mouvement {abs(change_pct):.1f}% sans analyse AI",
            "reliability": reliability,
            "tradeable": False,
            "reasoning": "Analyse de fallback - API indisponible"
        }

    def analyze_symbol(self, symbol, df, sector) -> Optional[Dict]:
        """Analyse complète d'un symbole sur la dernière bougie"""
        if df.empty:
            return None

        # On n'analyse la bougie courante que si le marché est ouvert
        if not self.is_market_open():
            print(f"⚠️ Marché fermé, analyse de la bougie courante ignorée pour {symbol}")
            return None
        
        # Détection du mouvement sur la dernière bougie (seuil dynamique)
        is_significant, change_pct, direction = self.detect_significant_move(df)
        if not is_significant:
            return None

        # Volume spike
        volume_spike = False
        volume_ratio = 0.0
        try:
            if len(df) >= self.settings["analysis"]["volume_min_candles"] and 'Volume' in df.columns:
                current_vol = df['Volume'].iloc[-1]
                avg_vol = df['Volume'].iloc[-5:].mean()
                volume_ratio = current_vol / avg_vol if avg_vol > 0 else 0.0
                volume_spike = volume_ratio >= self.settings["analysis"]["volume_spike_threshold"]
        except:
            pass

        # News
        news = self.get_news_context(symbol, days=self.settings["catalyst"]["news_days"])

        # Analyse IA
        catalyst_analysis = self.analyze_catalyst_with_ai(symbol, change_pct, sector, news, volume_spike)

        result = {
            "symbol": symbol,
            "date": datetime.now(EST).strftime('%Y-%m-%d'),
            "timestamp": datetime.now(EST).isoformat(),
            "change_pct": round(change_pct, 2),
            "direction": direction,
            "sector": sector,
            "volume_spike": bool(volume_spike),
            "volume_ratio": round(float(volume_ratio), 2),
            "news_count": len(news),
            "catalyst": catalyst_analysis
        }

        # Ajout mémoire
        exists = any(
            c['symbol'] == result['symbol'] and c['date'] == result['date']
            for c in self.current_catalysts
        )
        if not exists:
            self.current_catalysts.append(result)
            # Supprimé - alerte finale affichée par scanner
        return result

    def analyze_all_candles(self, symbol, df, sector) -> None:
        """Analyse backtest historique des dernières bougies"""
        if len(df) < 2:
            return

        # Calculer la moyenne des mouvements pour ce symbole
        avg_move = self.calculate_average_move(df)
        dynamic_threshold = max(avg_move * self.multiplier, self.min_threshold)
        
        logger.info(f"📊 {symbol}: Moyenne={avg_move:.2f}%, Seuil dynamique={dynamic_threshold:.2f}%")

        days_back = self.settings["historique"]["nb_bougies"]
        start_idx = max(1, len(df) - days_back)
        found_count = 0
        for i in range(start_idx, len(df)):
            candle_date = df.index[i].date()
            current_price = df['Close'].iloc[i]
            prev_price = df['Close'].iloc[i - 1]
            
            if prev_price == 0:
                continue
                
            change_pct = ((current_price - prev_price) / prev_price) * 100

            if abs(change_pct) < dynamic_threshold:
                continue

            # Volume spike
            volume_spike = False
            try:
                if i >= self.settings["analysis"]["volume_min_candles"] and 'Volume' in df.columns:
                    current_vol = df['Volume'].iloc[i]
                    avg_vol = df['Volume'].iloc[i-5:i].mean()
                    volume_spike = current_vol / avg_vol >= self.settings["analysis"]["volume_spike_threshold"] if avg_vol > 0 else False
            except:
                pass

            # Fallback AI pour historique
            catalyst_analysis = self._fallback_analysis(change_pct, volume_spike, 0)

            result = {
                "symbol": symbol,
                "date": candle_date.strftime('%Y-%m-%d'),
                "timestamp": df.index[i].isoformat(),
                "change_pct": round(change_pct, 2),
                "direction": "UP" if change_pct > 0 else "DOWN",
                "sector": sector,
                "volume_spike": volume_spike,
                "volume_ratio": 1.0,
                "news_count": 0,
                "catalyst": catalyst_analysis
            }

            self.current_catalysts.append(result)
            found_count += 1
            print(f" {symbol} {candle_date}: {change_pct:+.1f}% {catalyst_analysis['type']}")

        print(f"   ✅ {symbol}: {found_count} catalyseurs trouvés")

    def get_recent_alerts(self, days: Optional[int] = None) -> List[Dict]:
        """Retourne catalyseurs mémoire session filtrés par date"""
        days = days or self.settings["catalyst"]["recent_alerts_days"]
        cutoff_date = (datetime.now(EST) - timedelta(days=days)).strftime('%Y-%m-%d')
        filtered = [c for c in self.current_catalysts if c['date'] >= cutoff_date]
        return sorted(filtered, key=lambda x: x['timestamp'], reverse=True)


def integrate_with_dashboard(symbol, df, analyzer, sector_map) -> Optional[Dict]:
    """Fonction d'intégration pour dashboard avec seuil dynamique"""
    if analyzer is None:
        return None
    sector = sector_map.get(symbol, "unknown")
    try:
        return analyzer.analyze_symbol(symbol, df, sector)
    except Exception as e:
        print(f"❌ Erreur intégration dashboard pour {symbol}: {e}")
        return None
