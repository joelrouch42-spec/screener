#!/usr/bin/env python3
"""
Multi-source data provider with fallback system
Mode backtest: Yahoo Finance -> CSV cache in data/
Mode réel: IBKR direct
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from abc import ABC, abstractmethod
from pathlib import Path

# Import IBKR provider
try:
    from ibkr_provider import IBKRProvider
    IBKR_AVAILABLE = True
except ImportError:
    print("⚠️  IBKR provider non disponible (ib_insync non installé)")
    IBKR_AVAILABLE = False

# Cache directory
CACHE_DIR = Path(__file__).parent / "data"
CACHE_DIR.mkdir(exist_ok=True)


def get_cache_path(symbol: str) -> Path:
    """Retourne le chemin du fichier CSV cache pour un symbole avec date"""
    today = datetime.now().strftime('%Y-%m-%d')
    return CACHE_DIR / f"{today}_{symbol}.csv"


def find_latest_cache(symbol: str) -> Path:
    """Trouve le fichier cache le plus récent pour un symbole"""
    # Chercher tous les fichiers qui matchent le pattern *_SYMBOL.csv
    cache_files = list(CACHE_DIR.glob(f"*_{symbol}.csv"))

    if not cache_files:
        return None

    # Trier par date de modification (plus récent en premier)
    cache_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return cache_files[0]


def load_from_cache(symbol: str, days: int = 150) -> pd.DataFrame:
    """
    Charge les données depuis le cache CSV (cherche le plus récent)
    Vérifie UNIQUEMENT que le fichier est daté d'aujourd'hui

    Args:
        symbol: Symbole de l'action
        days: Non utilisé (gardé pour compatibilité API)

    Returns:
        DataFrame ou None si pas de cache valide
    """
    # Chercher le fichier cache le plus récent
    cache_file = find_latest_cache(symbol)

    if not cache_file or not cache_file.exists():
        return None

    try:
        # Extraire la date du nom du fichier (format: YYYY-MM-DD_SYMBOL.csv)
        filename = cache_file.stem  # Nom sans extension
        file_date_str = filename.split('_')[0]  # Extraire YYYY-MM-DD
        today_str = datetime.now().strftime('%Y-%m-%d')

        # Vérifier si le fichier est daté d'aujourd'hui
        if file_date_str != today_str:
            print(f"📅 Cache obsolète pour {symbol}: fichier du {file_date_str}, aujourd'hui {today_str}")
            return None

        # Charger le fichier - pas de vérification du nombre de lignes
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)

        print(f"✅ Cache HIT pour {symbol}: {len(df)} lignes (fichier: {cache_file.name})")
        return df

    except Exception as e:
        print(f"❌ Erreur lecture cache {symbol}: {e}")
        return None


def save_to_cache(symbol: str, df: pd.DataFrame):
    """
    Sauvegarde les données dans le cache CSV

    Args:
        symbol: Symbole de l'action
        df: DataFrame à sauvegarder
    """
    if df is None or df.empty:
        return

    cache_file = get_cache_path(symbol)

    try:
        # Sauvegarder avec l'index (Date)
        df.to_csv(cache_file)
        print(f"💾 Cache SAVED pour {symbol}: {len(df)} lignes -> {cache_file}")
    except Exception as e:
        print(f"❌ Erreur sauvegarde cache {symbol}: {e}")


class DataProvider(ABC):
    """Abstract base class for data providers"""
    
    @abstractmethod
    def fetch_data(self, symbol, days=80, period=60):
        """Fetch historical data"""
        pass
    
    @abstractmethod
    def get_live_price(self, symbol):
        """Get current live price"""
        pass
    
    @abstractmethod
    def is_available(self):
        """Check if provider is available"""
        pass


class PolygonProvider(DataProvider):
    """Polygon.io data provider - Best for real-time"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or "4GjYKrVOnvXqYx4Kz500Yxiz4kjN5nfF"
        self.base_url = "https://api.polygon.io"
        
    def is_available(self):
        """Check if API key is valid"""
        if not self.api_key or self.api_key == "YOUR_POLYGON_API_KEY":
            return False
        # Quick test
        try:
            url = f"{self.base_url}/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-02?apiKey={self.api_key}"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def fetch_data(self, symbol, days=80, period=60):
        """Fetch historical daily data from Polygon"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'apiKey': self.api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data.get('status') != 'OK' or not data.get('results'):
            raise ValueError(f"Polygon: No data for {symbol}")
        
        # Convert to DataFrame
        results = data['results']
        df = pd.DataFrame(results)
        df['Date'] = pd.to_datetime(df['t'], unit='ms')
        df = df.set_index('Date')
        df = df.rename(columns={
            'o': 'Open',
            'h': 'High',
            'l': 'Low',
            'c': 'Close',
            'v': 'Volume'
        })
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(period)
        
        return df
    
    def get_live_price(self, symbol):
        """Get real-time last trade"""
        url = f"{self.base_url}/v2/last/trade/{symbol}"
        params = {'apiKey': self.api_key}
        
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        
        if data.get('status') == 'OK' and data.get('results'):
            return data['results']['p']  # price
        return None


class AlphaVantageProvider(DataProvider):
    """Alpha Vantage data provider - Good for OTC stocks"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or "FC2Y5S2ZDA66X6VM"
        self.base_url = "https://www.alphavantage.co/query"
        
    def is_available(self):
        """Check if API key is valid"""
        if not self.api_key or self.api_key == "YOUR_ALPHAVANTAGE_API_KEY":
            return False
        try:
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': 'AAPL',
                'apikey': self.api_key
            }
            response = requests.get(self.base_url, params=params, timeout=5)
            data = response.json()
            return 'Time Series (Daily)' in data
        except:
            return False
    
    def fetch_data(self, symbol, days=80, period=60):
        """Fetch historical daily data from Alpha Vantage"""
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': 'full',
            'apikey': self.api_key
        }
        
        response = requests.get(self.base_url, params=params, timeout=10)
        data = response.json()
        
        if 'Time Series (Daily)' not in data:
            raise ValueError(f"Alpha Vantage: No data for {symbol}")
        
        # Convert to DataFrame
        ts = data['Time Series (Daily)']
        df = pd.DataFrame.from_dict(ts, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        })
        df = df.astype(float)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(period)
        
        return df
    
    def get_live_price(self, symbol):
        """Get current quote"""
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        response = requests.get(self.base_url, params=params, timeout=5)
        data = response.json()
        
        if 'Global Quote' in data and '05. price' in data['Global Quote']:
            return float(data['Global Quote']['05. price'])
        return None


class YahooFinanceProvider(DataProvider):
    """Yahoo Finance data provider - Fallback option"""
    
    def is_available(self):
        """Yahoo Finance is always available"""
        return True
    
    def fetch_data(self, symbol, days=80, period=60):
        """Fetch historical data from Yahoo Finance"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start_date, end=end_date, interval="1d")
        
        if hist.empty:
            raise ValueError(f"Yahoo Finance: No data for {symbol}")
        
        hist = hist.tail(period)
        df = hist[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        return df
    
    def get_live_price(self, symbol):
        """Get current price from Yahoo"""
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return info.get('currentPrice') or info.get('regularMarketPrice')


class MultiSourceDataProvider:
    """
    Orchestrates multiple data providers with two modes:
    - backtest_mode=True: Yahoo Finance + CSV cache in data/
    - backtest_mode=False: IBKR direct (real-time)
    """

    def __init__(self, backtest_mode=True, polygon_key=None, alphavantage_key=None, debug=False):
        print(f"🔧 MultiSourceDataProvider.__init__ appelé avec backtest_mode={backtest_mode}")
        self.debug = debug
        self.backtest_mode = backtest_mode
        self.providers = []

        if backtest_mode:
            # Mode backtest : Yahoo uniquement (+ cache CSV)
            print("📊 MODE BACKTEST: Yahoo Finance + CSV cache")
            self.providers = [YahooFinanceProvider()]
        else:
            # Mode réel : IBKR uniquement
            print("🔴 MODE RÉEL: IBKR direct")
            if IBKR_AVAILABLE:
                self.providers = [IBKRProvider(client_id=4)]
            else:
                raise RuntimeError("IBKR non disponible. Installez ib_insync: pip install ib_insync")

        # Test which providers are available
        self.available_providers = []
        for provider in self.providers:
            if provider.is_available():
                self.available_providers.append(provider)
                if self.debug:
                    print(f"✅ {provider.__class__.__name__} disponible")
            else:
                if self.debug:
                    print(f"❌ {provider.__class__.__name__} non disponible")
    
    def fetch_data(self, symbol, days=80, period=60):
        """
        Fetch data with two strategies:
        - backtest_mode: CSV cache first, then Yahoo + save to cache
        - real mode: IBKR direct
        """
        # MODE BACKTEST : Essayer cache CSV d'abord
        if self.backtest_mode:
            cached_df = load_from_cache(symbol)
            if cached_df is not None:
                # Cache hit ! Retourner directement
                print(f"⏭️  SKIPPED download for {symbol}: using cached data from today")
                return cached_df.tail(period), YahooFinanceProvider()

            # Cache miss : télécharger depuis Yahoo et sauvegarder
            print(f"📥 DOWNLOADING {symbol} from Yahoo Finance (no valid cache)...")
            try:
                yahoo_provider = YahooFinanceProvider()
                df = yahoo_provider.fetch_data(symbol, days=days, period=period)

                # Sauvegarder dans le cache (on sauvegarde plus que period pour futurs usages)
                # Télécharger plus de données pour le cache
                print(f"💾 Fetching full year of data for {symbol} to populate cache...")
                full_df = yahoo_provider.fetch_data(symbol, days=365, period=365)
                save_to_cache(symbol, full_df)

                return df, yahoo_provider
            except Exception as e:
                raise ValueError(f"Impossible de télécharger {symbol} depuis Yahoo: {e}")

        # MODE RÉEL : IBKR direct (pas de cache)
        else:
            last_error = None
            for provider in self.available_providers:
                try:
                    if self.debug:
                        print(f"🔄 Tentative {provider.__class__.__name__}...")
                    df = provider.fetch_data(symbol, days, period)
                    if self.debug:
                        print(f"✅ Données récupérées via {provider.__class__.__name__}")
                    return df, provider
                except Exception as e:
                    if self.debug:
                        print(f"❌ {provider.__class__.__name__} échoué: {e}")
                    last_error = e
                    continue

            raise ValueError(f"Aucune source de données disponible. Dernière erreur: {last_error}")
    
    def get_live_price(self, symbol, preferred_provider=None):
        """Get live price, try preferred provider first then fallback"""
        # Try preferred provider first if specified
        if preferred_provider:
            try:
                price = preferred_provider.get_live_price(symbol)
                if price:
                    if self.debug:
                        print(f"✅ Prix live via {preferred_provider.__class__.__name__}: ${price}")
                    return price
                else:
                    if self.debug:
                        print(f"⚠️  {preferred_provider.__class__.__name__} n'a pas retourné de prix")
            except Exception as e:
                if self.debug:
                    print(f"❌ {preferred_provider.__class__.__name__} live price error: {e}")
        
        # Fallback to all available providers
        for provider in self.available_providers:
            if provider == preferred_provider:  # Already tried
                continue
            try:
                price = provider.get_live_price(symbol)
                if price:
                    if self.debug:
                        print(f"✅ Prix live via {provider.__class__.__name__}: ${price}")
                    return price
                else:
                    if self.debug:
                        print(f"⚠️  {provider.__class__.__name__} n'a pas retourné de prix")
            except Exception as e:
                if self.debug:
                    print(f"❌ {provider.__class__.__name__} live price error: {e}")
        
        if self.debug:
            print(f"❌ Aucun provider n'a retourné de prix live pour {symbol}")
        return None


# Test function
if __name__ == "__main__":
    # Test with your API keys
    provider = MultiSourceDataProvider(
        polygon_key="YOUR_POLYGON_KEY",  # Replace with real key
        alphavantage_key="YOUR_ALPHAVANTAGE_KEY"  # Replace with real key
    )
    
    # Test fetch
    try:
        df, used_provider = provider.fetch_data("VSME", days=80, period=60)
        print(f"\n✅ Récupéré {len(df)} bougies")
        print(f"Dernière bougie:\n{df.tail(1)}")
        
        # Test live price
        live = provider.get_live_price("VSME", used_provider)
        print(f"\n💰 Prix live: ${live}")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")