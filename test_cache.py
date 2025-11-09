#!/usr/bin/env python3
"""
Test script to populate the CSV cache in data/
Reads symbols from config file
"""

import sys
from data_providers import MultiSourceDataProvider

def parse_config(config_file='scanner_config.txt'):
    """Parse config file and return list of symbols"""
    symbols = []
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 1:
                    symbols.append(parts[0].upper())
    except Exception as e:
        print(f"❌ Error reading config file: {e}")
        sys.exit(1)
    return symbols

# Get config file from command line or use default
config_file = sys.argv[1] if len(sys.argv) > 1 else 'scanner_config.txt'

print(f"🧪 Testing backtest mode with CSV cache...")
print(f"📋 Reading symbols from: {config_file}")
print("=" * 60)

# Parse config to get symbols
symbols = parse_config(config_file)
if not symbols:
    print("❌ No symbols found in config file!")
    sys.exit(1)

print(f"📊 Found {len(symbols)} symbols: {', '.join(symbols)}\n")

# Initialize provider in backtest mode
provider = MultiSourceDataProvider(backtest_mode=True, debug=True)

# Download data for each symbol
for symbol in symbols:
    print(f"\n📥 Downloading {symbol}...")
    try:
        df, used = provider.fetch_data(symbol, days=365, period=120)
        print(f"✅ Got {len(df)} candles for {symbol}")
    except Exception as e:
        print(f"❌ Failed for {symbol}: {e}")

print("\n" + "=" * 60)
print("✅ Done! Check data/ folder for CSV files")
