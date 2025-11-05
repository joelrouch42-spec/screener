#!/usr/bin/env python3
import yfinance as yf
from datetime import datetime, timedelta

ticker = yf.Ticker('VSME')
end = datetime.now()
start = end - timedelta(days=20)

hist = ticker.history(start=start, end=end, interval='1d')
print(f'Total bougies: {len(hist)}')
print('\nToutes les dates (derniers 15 jours):')
for date, row in hist.tail(15).iterrows():
    print(f'{date.strftime("%Y-%m-%d %A")}: Close=${row["Close"]:.2f} Vol={int(row["Volume"]):,}')
