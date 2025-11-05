#!/usr/bin/env python3
import yfinance as yf
from datetime import datetime, timedelta

print("="*80)
print("TEST: MSFT (action liquide qui trade tous les jours)")
print("="*80)

ticker = yf.Ticker('MSFT')
end = datetime.now()
start = end - timedelta(days=20)

hist = ticker.history(start=start, end=end, interval='1d')
print(f'\nTotal bougies MSFT: {len(hist)}')
print('\nToutes les dates MSFT (derniers 15 jours):')
for date, row in hist.tail(15).iterrows():
    print(f'{date.strftime("%Y-%m-%d %A")}: Close=${row["Close"]:.2f} Vol={int(row["Volume"]):,}')

print("\n" + "="*80)
print("TEST: VSME (penny stock)")
print("="*80)

ticker2 = yf.Ticker('VSME')
hist2 = ticker2.history(start=start, end=end, interval='1d')
print(f'\nTotal bougies VSME: {len(hist2)}')
print('\nToutes les dates VSME (derniers 15 jours):')
for date, row in hist2.tail(15).iterrows():
    print(f'{date.strftime("%Y-%m-%d %A")}: Close=${row["Close"]:.2f} Vol={int(row["Volume"]):,}')

# Vérifier si le 4 novembre existe
print("\n" + "="*80)
print("VÉRIFICATION: Est-ce que le 4 novembre 2025 existe?")
print("="*80)
nov_4_2025 = datetime(2025, 11, 4)
msft_nov4 = hist[hist.index.date == nov_4_2025.date()]
vsme_nov4 = hist2[hist2.index.date == nov_4_2025.date()]

print(f"\nMSFT le 2025-11-04: {'OUI' if len(msft_nov4) > 0 else 'NON - MANQUANT'}")
print(f"VSME le 2025-11-04: {'OUI' if len(vsme_nov4) > 0 else 'NON - MANQUANT'}")

if len(msft_nov4) == 0:
    print("\n⚠️  Le 4 novembre manque même pour MSFT - c'est Yahoo qui ne fournit pas cette date!")
    print("Raisons possibles: jour férié, marché fermé, ou données pas encore publiées")
