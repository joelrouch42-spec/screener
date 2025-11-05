#!/usr/bin/env python3
"""
Backtest du scanner - Simule le scanner sur données historiques
Teste EXACTEMENT ce que scanner.py détecte : breakouts + catalyseurs
"""

import sys
from datetime import datetime, timedelta
import logging

# Import du scanner et ses dépendances
from scanner import StockScanner, detect_scanner_breakouts
from data_providers import MultiSourceDataProvider
from catalyst_analyzer import CatalystAnalyzer, load_settings
from tabs import SupportResistance

# Setup logging
logging.basicConfig(
    level=logging.WARNING,  # Réduire le bruit
    format='%(message)s'
)
logger = logging.getLogger(__name__)

def backtest_symbol(symbol, sector, data_provider, catalyst_analyzer, settings, days_back=30):
    """
    Backtest un symbole sur les N derniers jours
    Simule exactement ce que fait scan_symbol()
    """

    print(f"\n🎯 BACKTEST {symbol} ({sector})")
    print("=" * 70)

    # 1. Charger TOUTES les données historiques (minimum 120 jours)
    try:
        df_all, provider_used = data_provider.fetch_data(
            symbol,
            days=150,
            period=150
        )
        if df_all is None or len(df_all) < days_back + 30:
            print(f"❌ Pas assez de données pour {symbol}")
            return []
    except Exception as e:
        print(f"❌ Erreur chargement données {symbol}: {e}")
        return []

    print(f"✅ {len(df_all)} bougies chargées via {provider_used}")

    # 2. Définir la période de test (30 derniers jours)
    start_index = len(df_all) - days_back
    end_index = len(df_all) - 1

    start_date = df_all.index[start_index]
    end_date = df_all.index[end_index]
    print(f"📊 Période: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')} ({days_back} jours)")

    alerts_found = []

    # 3. Scanner jour par jour
    for day_idx in range(start_index, end_index + 1):
        # Données jusqu'à ce jour (simule le scanner en temps réel)
        df_until_today = df_all.iloc[:day_idx+1].copy()
        current_date = df_all.index[day_idx]
        days_ago = len(df_all) - 1 - day_idx

        # Minimum 30 bougies pour calculer S/R
        if len(df_until_today) < 30:
            continue

        try:
            # A. Calculer Support/Resistance (comme dans le scanner)
            sr = SupportResistance()
            order = settings["support_resistance"]["order"]
            support_levels, resistance_levels, _, _ = sr.find_levels(df_until_today, order)

            # B. Détecter breakout technique (comme scan_symbol)
            breakout_info = None
            if len(support_levels) > 0 or len(resistance_levels) > 0:
                df_last_2 = df_until_today.tail(2)
                if len(df_last_2) >= 2:
                    breakout_info = detect_scanner_breakouts(
                        df_last_2,
                        support_levels,
                        resistance_levels,
                        settings
                    )

            # C. Détecter catalyseur (comme scan_symbol)
            catalyst_info = None
            if len(df_until_today) >= 2:
                try:
                    catalyst_info = catalyst_analyzer.analyze_symbol(
                        symbol,
                        df_until_today,
                        sector
                    )
                except Exception as e:
                    logger.debug(f"Catalyst analysis failed: {e}")

            # D. Priorité: Catalyst > Breakout (comme scanner)
            alert_info = catalyst_info or breakout_info

            if alert_info:
                alert_type = alert_info.get('type', 'unknown')
                is_technical = alert_type in ['resistance_breakout', 'support_breakdown']

                current_price = float(df_until_today['Close'].iloc[-1])
                previous_price = float(df_until_today['Close'].iloc[-2])
                change_pct = ((current_price - previous_price) / previous_price) * 100

                alert = {
                    'symbol': symbol,
                    'date': current_date.strftime('%Y-%m-%d'),
                    'days_ago': days_ago,
                    'type': alert_type,
                    'is_technical': is_technical,
                    'price': current_price,
                    'change_pct': change_pct,
                    'alert_info': alert_info
                }

                alerts_found.append(alert)

                # Afficher immédiatement
                emoji = "⚡" if is_technical else "🔥"
                type_label = alert_type.replace('_', ' ').upper()
                print(f"{emoji} {current_date.strftime('%Y-%m-%d')} (J-{days_ago:2d}): {type_label:25s} | ${current_price:7.2f} ({change_pct:+5.1f}%)")

        except Exception as e:
            logger.debug(f"Error on {current_date}: {e}")
            continue

    return alerts_found

def main():
    """Main backtest"""

    print("🔬 BACKTEST SCANNER")
    print("=" * 70)
    print("📅 Simule le scanner sur les 30 derniers jours")
    print("🎯 Détecte: Breakouts techniques + Catalyseurs")
    print("=" * 70)

    # 1. Charger la config
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'config.txt'

    scanner = StockScanner(config_file)
    if not scanner.load_configuration():
        print("❌ Erreur chargement configuration")
        sys.exit(1)

    print(f"✅ Config chargée: {len(scanner.symbols_config)} symboles")

    # 2. Initialiser services
    data_provider = MultiSourceDataProvider()
    settings = load_settings('settings.json')
    catalyst_analyzer = CatalystAnalyzer(settings=settings)

    # IMPORTANT: Bypasser la vérification du marché pour le backtest
    catalyst_analyzer.is_market_open = lambda dt=None: True
    print("⚠️  Mode BACKTEST: Vérification marché désactivée")

    # 3. Backtest chaque symbole
    all_alerts = []

    for symbol in scanner.symbols_config.keys():
        sector = scanner.sector_map.get(symbol, 'unknown')

        try:
            alerts = backtest_symbol(
                symbol,
                sector,
                data_provider,
                catalyst_analyzer,
                settings,
                days_back=30
            )
            all_alerts.extend(alerts)
        except Exception as e:
            print(f"❌ Erreur backtest {symbol}: {e}")
            continue

    # 4. Résumé
    print(f"\n📊 RÉSUMÉ")
    print("=" * 70)
    print(f"🎯 Total alertes détectées: {len(all_alerts)}")

    if all_alerts:
        technical = [a for a in all_alerts if a['is_technical']]
        catalyst = [a for a in all_alerts if not a['is_technical']]

        print(f"⚡ Breakouts techniques: {len(technical)}")
        print(f"🔥 Catalyseurs: {len(catalyst)}")

        # Top 5
        print(f"\n🏆 TOP 5 ALERTES:")
        sorted_alerts = sorted(all_alerts, key=lambda x: abs(x['change_pct']), reverse=True)
        for i, a in enumerate(sorted_alerts[:5], 1):
            emoji = "⚡" if a['is_technical'] else "🔥"
            print(f"  {i}. {a['symbol']} {a['date']}: {a['type']:20s} {emoji} ({a['change_pct']:+.1f}%)")
    else:
        print("ℹ️  Aucune alerte détectée")

    print(f"\n✅ Backtest terminé!")

if __name__ == "__main__":
    main()
