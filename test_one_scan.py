#!/usr/bin/env python3
"""
Test rapide - Un seul scan puis arrêt
Fonctionne 24/7 même marché fermé
"""

import sys
sys.path.insert(0, '/home/user/screener/venv/lib/python3.12/site-packages')

from scanner import StockScanner
from datetime import datetime

# Monkey patch pour ignorer les heures de marché
StockScanner.is_market_open = lambda self, dt=None: True

def test_single_scan():
    """Lance un seul scan de test"""

    config_file = sys.argv[1] if len(sys.argv) > 1 else 'config_test.txt'

    print("🧪 TEST - UN SEUL SCAN")
    print("=" * 80)
    print(f"Config: {config_file}")
    print(f"Heure: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("⚠️  Mode TEST : Marché toujours ouvert")
    print("=" * 80)
    print()

    # Créer le scanner
    scanner = StockScanner(config_file)

    # Charger config
    if not scanner.load_configuration():
        print("❌ Erreur chargement config")
        return

    # Initialiser services
    if not scanner.initialize_services():
        print("❌ Erreur initialisation services")
        return

    print(f"\n🔍 Scan de {len(scanner.symbols_config)} symboles...")
    print("-" * 80)

    # Scan tous les symboles
    alerts_found = scanner.scan_all_symbols()

    print()
    print("=" * 80)
    print("📊 RÉSULTATS")
    print("=" * 80)

    if alerts_found > 0:
        print(f"🚨 {alerts_found} alerte(s) détectée(s)")
    else:
        print("✅ Aucune alerte (conditions normales)")

    # Afficher les métriques
    stats = scanner.metrics.get_stats()
    print(f"\n📈 Métriques:")
    print(f"   Symboles scannés: {len(scanner.symbols_config)}")
    print(f"   API calls: {stats['total_api_calls']}")
    print(f"   Erreurs: {stats['total_errors']}")
    print(f"   Cache hit rate: {stats['cache_hit_rate']:.1f}%")

    print()
    print("=" * 80)
    print("💡 Pour un scan continu, utilisez: ./run_test_24_7.sh")
    print("=" * 80)

if __name__ == '__main__':
    try:
        test_single_scan()
    except KeyboardInterrupt:
        print("\n⌨️  Interrompu")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
