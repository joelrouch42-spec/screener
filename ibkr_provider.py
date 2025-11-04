#!/usr/bin/env python3
"""
Interactive Brokers Data Provider
Utilise ib_insync pour récupérer des données temps réel d'IBKR
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple, List
import nest_asyncio
from ib_insync import IB, Stock, Forex, util
import threading
import time

# Allow nested event loops
nest_asyncio.apply()

# Global lock pour sérialiser les requêtes IBKR
_ibkr_lock = threading.Lock()

class IBKRProvider:
    """
    Fournisseur de données Interactive Brokers
    """
    
    def __init__(self, host='127.0.0.1', port=7497, client_id=2):
        """
        Initialise la connexion IBKR
        
        Args:
            host: Adresse IP d'IB Gateway/TWS (défaut: localhost)
            port: Port d'IB Gateway (7497) ou TWS (7496)
            client_id: ID client unique
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = None
        self.connected = False
        print(f"🔌 IBKRProvider configuré: {host}:{port} (client_id={client_id})")
    
    def connect(self) -> bool:
        """
        Connecte à IB Gateway/TWS
        
        Returns:
            True si connexion réussie, False sinon
        """
        try:
            if self.ib is None:
                self.ib = IB()
            
            if not self.ib.isConnected():
                self.ib.connect(self.host, self.port, clientId=self.client_id)
                self.connected = True
                print(f"✅ Connecté à IBKR {self.host}:{self.port}")
                return True
            else:
                self.connected = True
                return True
                
        except Exception as e:
            print(f"❌ Erreur connexion IBKR: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Déconnecte d'IBKR"""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            self.connected = False
            print("🔌 Déconnecté d'IBKR")
    
    def is_available(self) -> bool:
        """
        Vérifie si IBKR est disponible en testant la connexion
        
        Returns:
            True si connexion possible, False sinon
        """
        try:
            # Test rapide de connexion
            if self.ib is None:
                self.ib = IB()
            
            if not self.ib.isConnected():
                self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=3)
                self.connected = True
                print("✅ IBKR connecté pour test")
                return True
            else:
                self.connected = True
                return True
                
        except Exception as e:
            print(f"⚠️  IBKR non disponible: {e}")
            self.connected = False
            return False
    
    def create_contract(self, symbol: str):
        """
        Crée un contrat IB pour le symbole
        
        Args:
            symbol: Symbole du titre (ex: AAPL, EURUSD)
            
        Returns:
            Contract IB approprié
        """
        # Pour les actions US par défaut
        if len(symbol) <= 5 and symbol.isalpha():
            return Stock(symbol, 'SMART', 'USD')
        
        # Pour les forex (ex: EURUSD)
        if len(symbol) == 6 and symbol.isalpha():
            base = symbol[:3]
            quote = symbol[3:]
            return Forex(f"{base}{quote}")
        
        # Par défaut, action US
        return Stock(symbol, 'SMART', 'USD')
    
    def get_historical_data(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """
        Récupère les données historiques
        
        Args:
            symbol: Symbole du titre
            days: Nombre de jours d'historique
            
        Returns:
            DataFrame avec colonnes OHLCV ou None
        """
        # Utilise le lock global pour sérialiser les requêtes IBKR
        with _ibkr_lock:
            print(f"🔒 IBKR: Lock acquis pour {symbol}")
            
            if not self.connect():
                print(f"❌ IBKR connexion échouée pour {symbol}")
                return None
            
            try:
                print(f"🔍 IBKR: Recherche contrat pour {symbol}...")
                contract = self.create_contract(symbol)
                
                # Qualifie le contrat avec timeout agressif
                print(f"🔄 IBKR: Qualification contrat {symbol}...")
                qualified_contracts = self.ib.qualifyContracts(contract)
                
                if not qualified_contracts:
                    print(f"❌ IBKR: Contrat non trouvé pour {symbol} (timeout)")
                    return None
                
                contract = qualified_contracts[0]
                print(f"✅ IBKR: Contrat qualifié pour {symbol}")
                
                # Récupère les données historiques avec timeout ultra-court
                short_duration = "5 D"  # Juste 5 jours
                bar_size = "1 day"
                
                print(f"📥 IBKR: Demande données historiques {symbol} (5 jours)...")
                
                bars = self.ib.reqHistoricalData(
                    contract,
                    endDateTime='',
                    durationStr=short_duration,
                    barSizeSetting=bar_size,
                    whatToShow='TRADES',
                    useRTH=True,
                    formatDate=1,
                    timeout=1  # Timeout de 1 seconde seulement
                )
                
                if not bars:
                    print(f"❌ IBKR: Aucune donnée historique pour {symbol}")
                    return None
                
                # Convertit en DataFrame
                df = util.df(bars)
                
                if df.empty:
                    print(f"❌ IBKR: DataFrame vide pour {symbol}")
                    return None
                
                # Renomme les colonnes pour correspondre au format attendu
                df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High', 
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                
                # Set l'index sur la date
                df.index = pd.to_datetime(df['date'])
                df = df.drop('date', axis=1)
                
                # Assure que les colonnes requises existent
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in required_cols:
                    if col not in df.columns:
                        df[col] = 0.0
                
                print(f"✅ IBKR: {len(df)} barres récupérées pour {symbol}")
                return df[required_cols]
                
            except Exception as e:
                print(f"❌ IBKR erreur données {symbol}: {e}")
                return None
    
    def get_live_price(self, symbol: str) -> Optional[float]:
        """
        Récupère le prix temps réel
        
        Args:
            symbol: Symbole du titre
            
        Returns:
            Prix actuel ou None
        """
        if not self.connect():
            return None
        
        try:
            contract = self.create_contract(symbol)
            
            # Qualifie le contrat avec timeout rapide
            qualified_contracts = self.ib.qualifyContracts(contract)
            if not qualified_contracts:
                return None
            
            contract = qualified_contracts[0]
            
            # Demande les données de marché différées (disponibles gratuitement)
            ticker = self.ib.reqMktData(contract, snapshot=True)
            self.ib.sleep(2)  # Attend plus longtemps pour les données différées
            
            # Récupère le prix
            price = None
            if ticker.last and ticker.last > 0:
                price = float(ticker.last)
            elif ticker.close and ticker.close > 0:
                price = float(ticker.close)
            elif ticker.bid and ticker.ask:
                price = float((ticker.bid + ticker.ask) / 2)
            
            # Annule la souscription rapidement
            try:
                self.ib.cancelMktData(contract)
            except:
                pass  # Ignore les erreurs d'annulation
            
            return price
            
        except Exception as e:
            print(f"❌ Erreur prix live IBKR {symbol}: {e}")
            return None
    
    def fetch_data(self, symbol: str, days: int, cache_minutes: int = 5) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Interface principale pour récupérer les données (compatible avec le dashboard)
        
        Args:
            symbol: Symbole du titre
            days: Nombre de jours d'historique
            cache_minutes: Non utilisé (pour compatibilité)
            
        Returns:
            (DataFrame, nom_du_provider)
        """
        try:
            # Tentative rapide IBKR
            df = self.get_historical_data(symbol, days)
            if df is not None:
                return df, "IBKRProvider"
            else:
                print(f"⚠️  IBKR: Données vides pour {symbol}, échec")
                raise Exception("Données IBKR vides")
        except Exception as e:
            print(f"❌ IBKR échec pour {symbol}: {e}")
            # Force le fallback en retournant None
            raise e


# Test rapide si lancé directement
if __name__ == "__main__":
    print("🧪 Test IBKRProvider")
    
    # Crée le provider
    ibkr = IBKRProvider()
    
    # Test de connexion
    if ibkr.connect():
        print("✅ Connexion IBKR réussie")
        
        # Test données historiques
        df, provider = ibkr.fetch_data("AAPL", 30)
        if df is not None:
            print(f"✅ Données AAPL: {len(df)} jours")
            print(df.tail())
        
        # Test prix live
        price = ibkr.get_live_price("AAPL")
        if price:
            print(f"✅ Prix live AAPL: ${price}")
        
        ibkr.disconnect()
    else:
        print("❌ Impossible de se connecter à IBKR")
        print("💡 Vérifiez que TWS ou IB Gateway est lancé et configuré pour accepter les connexions API")