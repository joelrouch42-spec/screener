#!/usr/bin/env python3
"""
Scanner en MODE TEST - Ignore les heures de marché
Permet de tester le scanner même quand le marché est fermé
"""

import sys
sys.path.insert(0, '/home/user/screener/venv/lib/python3.12/site-packages')

# Monkey patch pour désactiver la vérification du marché
import scanner

# Sauvegarder la fonction originale
_original_is_market_open = scanner.StockScanner.is_market_open

# Remplacer par une fonction qui retourne toujours True
def always_open(self, dt=None):
    """Mode TEST - Marché toujours ouvert"""
    return True

# Appliquer le patch
scanner.StockScanner.is_market_open = always_open

print("🧪 MODE TEST ACTIVÉ")
print("=" * 80)
print("⚠️  La vérification des heures de marché est DÉSACTIVÉE")
print("   Le scanner va fonctionner 24/7, même hors heures de marché")
print("=" * 80)
print()

# Lancer le scanner normalement
if __name__ == '__main__':
    scanner.main()
