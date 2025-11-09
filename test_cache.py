#!/usr/bin/env python3
"""
Test script to populate the CSV cache in data/
"""

from data_providers import MultiSourceDataProvider

print("🧪 Testing backtest mode with CSV cache...")
print("=" * 60)

provider = MultiSourceDataProvider(backtest_mode=True, debug=True)

# Download data for symbols from config
symbols = ['VSME', 'NVDA', 'AAPL', 'TSLA', 'MSFT']

for symbol in symbols:
    print(f"\n📥 Downloading {symbol}...")
    try:
        df, used = provider.fetch_data(symbol, days=365, period=120)
        print(f"✅ Got {len(df)} candles for {symbol}")
    except Exception as e:
        print(f"❌ Failed for {symbol}: {e}")

print("\n" + "=" * 60)
print("✅ Done! Check data/ folder for CSV files")
