#!/usr/bin/env python3
"""
Backtest du système de détection de catalyseurs
Teste les 30 derniers jours pour voir quels catalyseurs auraient été détectés
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from catalyst_analyzer import CatalystAnalyzer
from data_providers import MultiSourceDataProvider

def load_sector_mapping(file='sector_mapping.txt'):
    """Load sector mapping from file"""
    sector_map = {}
    
    try:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    symbol = parts[0].upper()
                    sector = parts[1].lower()
                    sector_map[symbol] = sector
    except FileNotFoundError:
        print(f"⚠️  Fichier {file} introuvable")
        return {}
    except Exception as e:
        print(f"⚠️  Erreur lecture sector mapping: {e}")
        return {}
    
    return sector_map

def parse_config(config_file='config.txt'):
    """Parse configuration file"""
    symbols_config = {}
    
    try:
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 4:
                    symbol = parts[0].upper()
                    provider = parts[4].lower() if len(parts) >= 5 else 'auto'
                    
                    symbols_config[symbol] = {
                        'min_rvol': float(parts[1]),
                        'min_beta': float(parts[2]),
                        'min_rvol_change': float(parts[3]),
                        'provider': provider
                    }
    except Exception as e:
        print(f"❌ Erreur lecture config: {e}")
        return {}
    
    return symbols_config

def get_historical_data(symbol, data_provider, days=90):
    """Récupère 90 jours de données historiques"""
    print(f"📥 Téléchargement de {days} jours pour {symbol}...")
    
    try:
        df, provider_used = data_provider.fetch_data(symbol, days, days)
        if df is None or df.empty:
            print(f"❌ Aucune donnée pour {symbol}")
            return None, None
        
        print(f"✅ {len(df)} bougies récupérées pour {symbol} via {provider_used}")
        return df, provider_used
        
    except Exception as e:
        print(f"❌ Erreur téléchargement {symbol}: {e}")
        return None, None

def calculate_metrics_for_date(df, date_index, min_beta, min_rvol, min_rvol_change):
    """Calcule les métriques pour une date spécifique (comme dans le dashboard)"""
    
    # On a besoin de 21 jours de données avant la date pour calculer les métriques
    if date_index < 21:
        return None
    
    # Créer un subset jusqu'à cette date
    subset_df = df.iloc[:date_index+1].copy()
    
    # Initialise les colonnes
    subset_df['Beta'] = 0.0
    subset_df['RVol'] = 0.0
    subset_df['RVol_Change'] = 0.0
    subset_df['Alert'] = False
    subset_df['RVol_Signal'] = ''
    
    # On ne calcule que pour le dernier point (date_index)
    i = date_index
    
    # --- Beta calculation (simulé, on utilise une approximation) ---
    if len(subset_df) >= 21:
        returns = subset_df['Close'].pct_change().tail(20)
        if len(returns.dropna()) > 10:
            # Approximation du beta (corrélation avec le marché simulée)
            volatility = returns.std()
            subset_df.loc[subset_df.index[i], 'Beta'] = min(max(volatility * 50, 0.5), 3.0)
    
    # --- RVol calculation ---
    if i >= 21:
        avg_volume = np.mean(subset_df.iloc[i-21:i-1]['Volume'])
        current_volume = subset_df.iloc[i]['Volume']
        
        if avg_volume > 0:
            rvol = current_volume / avg_volume
            subset_df.loc[subset_df.index[i], 'RVol'] = rvol
            
            # RVol change
            if i > 21:
                prev_rvol = subset_df.iloc[i-1]['RVol'] if 'RVol' in subset_df.columns else 1.0
                rvol_change = rvol - prev_rvol
                subset_df.loc[subset_df.index[i], 'RVol_Change'] = rvol_change
                
                # Alert detection
                if subset_df.iloc[i]['Beta'] >= min_beta and rvol >= min_rvol:
                    subset_df.loc[subset_df.index[i], 'Alert'] = True
    
    return subset_df

def backtest_symbol(symbol, sector, config, data_provider, analyzer):
    """Backtest un symbole sur les 30 derniers jours"""
    
    print(f"\n🎯 BACKTEST {symbol} ({sector})")
    print("=" * 50)
    
    # 1. Charge 90 jours de données
    df, provider_used = get_historical_data(symbol, data_provider, days=90)
    if df is None:
        return []
    
    min_beta = config['min_beta']
    min_rvol = config['min_rvol'] 
    min_rvol_change = config['min_rvol_change']
    
    catalysts_found = []
    
    # 2. Commence le scan à partir de 30 jours en arrière
    start_date_index = len(df) - 30  # 30 jours en arrière
    end_date_index = len(df) - 1     # Hier
    
    print(f"📊 Période de test: {df.index[start_date_index].strftime('%Y-%m-%d')} → {df.index[end_date_index].strftime('%Y-%m-%d')}")
    print(f"🔍 Scanning {end_date_index - start_date_index + 1} jours...")
    
    # 3. Scan jour par jour
    for day_index in range(start_date_index, end_date_index + 1):
        current_date = df.index[day_index]
        days_ago = len(df) - 1 - day_index
        
        # Calcule les métriques jusqu'à cette date
        subset_df = calculate_metrics_for_date(df, day_index, min_beta, min_rvol, min_rvol_change)
        
        if subset_df is None:
            continue
        
        # Test de détection de catalyseur
        is_significant, change_pct, direction = analyzer.detect_significant_move(subset_df, threshold=5.0)
        
        if is_significant:
            # Simule l'analyse complète (sans API car c'est du backtest)
            volume_spike = subset_df['RVol'].iloc[-1] >= 2.0 if 'RVol' in subset_df.columns else False
            
            catalyst_data = {
                "symbol": symbol,
                "date": current_date.strftime("%Y-%m-%d"),
                "days_ago": days_ago,
                "change_pct": round(change_pct, 2),
                "direction": direction,
                "sector": sector,
                "volume_spike": volume_spike,
                "rvol": round(subset_df['RVol'].iloc[-1], 2) if 'RVol' in subset_df.columns else 0,
                "beta": round(subset_df['Beta'].iloc[-1], 2) if 'Beta' in subset_df.columns else 0,
                "price": round(subset_df['Close'].iloc[-1], 2),
                "alert": subset_df['Alert'].iloc[-1] if 'Alert' in subset_df.columns else False
            }
            
            catalysts_found.append(catalyst_data)
            
            # 4. Print résultat immédiat
            emoji = "🚀" if direction == "UP" else "📉"
            alert_text = "🚨" if catalyst_data["alert"] else "⚪"
            
            print(f"{emoji} {current_date.strftime('%Y-%m-%d')} (J-{days_ago:2d}): {change_pct:+6.1f}% | ${catalyst_data['price']:7.2f} | RVol:{catalyst_data['rvol']:4.1f} | Beta:{catalyst_data['beta']:4.1f} {alert_text}")
    
    return catalysts_found

def main():
    """Main backtest execution"""
    
    print("🔬 BACKTEST SYSTÈME CATALYSEURS")
    print("=" * 60)
    print("📅 Analyse des 30 derniers jours")
    print("🎯 Seuil de détection: mouvements >5%")
    print("=" * 60)
    
    # Load configuration
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'config.txt'
    symbols_config = parse_config(config_file)
    sector_map = load_sector_mapping('sector_mapping.txt')
    
    if not symbols_config:
        print("❌ Aucune configuration trouvée!")
        sys.exit(1)
    
    # Initialize providers (backtest mode forcé)
    try:
        data_provider = MultiSourceDataProvider(backtest_mode=True)
        analyzer = CatalystAnalyzer("backtest_history.json")
        print(f"✅ Providers: {', '.join([p.__class__.__name__ for p in data_provider.available_providers])}")
    except Exception as e:
        print(f"❌ Erreur initialisation: {e}")
        sys.exit(1)
    
    # Backtest each symbol
    all_catalysts = []
    
    for symbol in symbols_config.keys():
        sector = sector_map.get(symbol, "unknown")
        config = symbols_config[symbol]
        
        try:
            catalysts = backtest_symbol(symbol, sector, config, data_provider, analyzer)
            all_catalysts.extend(catalysts)
        except Exception as e:
            print(f"❌ Erreur backtest {symbol}: {e}")
            continue
    
    # Summary
    print(f"\n📊 RÉSUMÉ DU BACKTEST")
    print("=" * 60)
    print(f"🎯 Total catalyseurs détectés: {len(all_catalysts)}")
    
    if all_catalysts:
        # Statistiques
        up_moves = [c for c in all_catalysts if c['direction'] == 'UP']
        down_moves = [c for c in all_catalysts if c['direction'] == 'DOWN']
        
        print(f"🚀 Mouvements haussiers: {len(up_moves)}")
        print(f"📉 Mouvements baissiers: {len(down_moves)}")
        
        avg_move = np.mean([abs(c['change_pct']) for c in all_catalysts])
        max_move = max([abs(c['change_pct']) for c in all_catalysts])
        
        print(f"📈 Mouvement moyen: {avg_move:.1f}%")
        print(f"🎯 Plus gros mouvement: {max_move:.1f}%")
        
        # Par secteur
        by_sector = {}
        for c in all_catalysts:
            sector = c['sector']
            if sector not in by_sector:
                by_sector[sector] = []
            by_sector[sector].append(c)
        
        print(f"\n📋 Par secteur:")
        for sector, catalysts in by_sector.items():
            avg_sector = np.mean([abs(c['change_pct']) for c in catalysts])
            print(f"  {sector}: {len(catalysts)} catalyseurs (avg: {avg_sector:.1f}%)")
        
        # Top 5 plus gros mouvements
        print(f"\n🏆 TOP 5 MOUVEMENTS:")
        sorted_catalysts = sorted(all_catalysts, key=lambda x: abs(x['change_pct']), reverse=True)
        for i, c in enumerate(sorted_catalysts[:5], 1):
            emoji = "🚀" if c['direction'] == 'UP' else "📉"
            print(f"  {i}. {c['symbol']} {c['date']}: {c['change_pct']:+.1f}% {emoji}")
    
    else:
        print("ℹ️  Aucun catalyseur détecté sur la période")
    
    print(f"\n✅ Backtest terminé!")

if __name__ == "__main__":
    main()