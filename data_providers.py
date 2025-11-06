#!/usr/bin/env python3
"""
Multi-source data provider with fallback system
Tries: IBKR -> Polygon -> Alpha Vantage -> Yahoo Finance
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import yfinance as yf
from abc import ABC, abstractmethod

# Timezone
EST = ZoneInfo("America/New_York")

# Import IBKR provider
try:
    from ibkr_provider import IBKRProvider
    IBKR_AVAILABLE = True
except ImportError:
    print("⚠️  IBKR provider non disponible (ib_insync non installé)")
    IBKR_AVAILABLE = False


class DataProvider(ABC):
    """Abstract base class for data providers"""
    
    @abstractmethod
    def fetch_data(self, symbol, days=80, period=60, end_date=None):
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
    
    def fetch_data(self, symbol, days=80, period=60, end_date=None):
        """Fetch historical daily data from Polygon"""
        if end_date is None:
            end_date = datetime.now(EST)
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
    
    def fetch_data(self, symbol, days=80, period=60, end_date=None):
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
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        # Filter by end_date if specified (backtest mode)
        if end_date is not None:
            df = df[df.index <= end_date]

        df = df.tail(period)

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
    
    def fetch_data(self, symbol, days=80, period=60, end_date=None):
        """Fetch historical data from Yahoo Finance"""
        ticker = yf.Ticker(symbol)

        if end_date is None:
            # Normal mode: EST timezone
            end_date = datetime.now(EST)
            start_date = end_date - timedelta(days=days)
            hist = ticker.history(start=start_date, end=end_date, interval="1d")
        else:
            # Backtest mode: use provided end_date (yfinance end is exclusive, add 1 day)
            start_date = end_date - timedelta(days=days)
            hist = ticker.history(start=start_date, end=end_date + timedelta(days=1), interval="1d")
        
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
    """Orchestrates multiple data providers with fallback"""

    def __init__(self, polygon_key=None, alphavantage_key=None, use_ibkr=True, debug=False):
        self.debug = debug
        self.providers = []

        # IBKR only - no fallback
        if not IBKR_AVAILABLE:
            raise RuntimeError("IBKR provider is not available. Please install ibapi: pip install ibapi")

        try:
            self.providers.append(IBKRProvider())
            if self.debug:
                print("🔌 IBKRProvider configuré (mode exclusif)")
        except Exception as e:
            raise RuntimeError(f"Impossible d'initialiser IBKR: {e}")

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
    
    def fetch_data(self, symbol, days=80, period=60, end_date=None):
        """Try each provider until one succeeds"""
        last_error = None

        for provider in self.available_providers:
            try:
                if self.debug:
                    print(f"🔄 Tentative {provider.__class__.__name__}...")
                df = provider.fetch_data(symbol, days, period, end_date=end_date)
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