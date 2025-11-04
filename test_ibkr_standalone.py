#!/usr/bin/env python3
"""
Test standalone IBKR - Connexion directe et lecture d'une valeur
"""

import sys
import time
from ibkr_provider import IBKRProvider

def test_ibkr_standalone():
    print("🧪 === TEST STANDALONE IBKR ===")
    print()
    
    # Test 1: Connexion
    print("1️⃣ Test de connexion...")
    ibkr = IBKRProvider(client_id=5)  # Client ID différent pour éviter conflits
    
    if not ibkr.connect():
        print("❌ ÉCHEC: Impossible de se connecter à IBKR")
        return False
    
    print("✅ Connexion IBKR réussie")
    print()
    
    # Test 2: Lecture de données AAPL
    print("2️⃣ Test lecture données AAPL...")
    try:
        df = ibkr.get_historical_data("AAPL", 5)
        if df is not None and not df.empty:
            print("✅ Données AAPL récupérées:")
            print(f"   Nombre de lignes: {len(df)}")
            print(f"   Dernière date: {df.index[-1]}")
            print(f"   Dernier prix: ${df['Close'].iloc[-1]:.2f}")
            print()
            print("Dernières données:")
            print(df.tail(2))
        else:
            print("❌ ÉCHEC: Aucune donnée récupérée pour AAPL")
            return False
    except Exception as e:
        print(f"❌ ERREUR lors de la récupération AAPL: {e}")
        return False
    
    print()
    
    # Test 3: Lecture de données VSME
    print("3️⃣ Test lecture données VSME...")
    try:
        df = ibkr.get_historical_data("VSME", 5)
        if df is not None and not df.empty:
            print("✅ Données VSME récupérées:")
            print(f"   Nombre de lignes: {len(df)}")
            print(f"   Dernière date: {df.index[-1]}")
            print(f"   Dernier prix: ${df['Close'].iloc[-1]:.2f}")
            print()
            print("Dernières données:")
            print(df.tail(2))
        else:
            print("❌ ÉCHEC: Aucune donnée récupérée pour VSME")
            print("ℹ️  VSME pourrait ne pas être disponible sur IBKR")
    except Exception as e:
        print(f"❌ ERREUR lors de la récupération VSME: {e}")
        print("ℹ️  VSME pourrait ne pas être disponible sur IBKR")
    
    print()
    
    # Test 4: Prix live AAPL
    print("4️⃣ Test prix live AAPL...")
    try:
        price = ibkr.get_live_price("AAPL")
        if price and price > 0:
            print(f"✅ Prix live AAPL: ${price:.2f}")
        else:
            print("⚠️  Prix live non disponible (abonnement requis)")
    except Exception as e:
        print(f"❌ ERREUR prix live: {e}")
    
    print()
    
    # Déconnexion
    print("5️⃣ Déconnexion...")
    ibkr.disconnect()
    print("✅ Déconnecté d'IBKR")
    
    print()
    print("🎉 TEST TERMINÉ")
    return True

if __name__ == "__main__":
    success = test_ibkr_standalone()
    sys.exit(0 if success else 1)