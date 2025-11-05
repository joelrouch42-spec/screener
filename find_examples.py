#!/usr/bin/env python3
"""
Script pour trouver des exemples de breakouts dans les données récentes
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def find_breakout_examples():
    symbols = ['NVDA', 'TSLA', 'COIN', 'PLTR', 'AMD', 'AAPL', 'MSFT']
    
    print("🔍 Recherche d'exemples de breakouts dans les derniers jours...")
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            # 3 jours de données horaires
            data = ticker.history(period='3d', interval='1h')
            
            if len(data) < 10:
                continue
                
            # Calculer les changements
            data['pct_change'] = data['Close'].pct_change() * 100
            data['vol_ratio'] = data['Volume'] / data['Volume'].rolling(5).mean()
            
            # Chercher mouvements significatifs avec volume
            significant = data[
                (abs(data['pct_change']) >= 1.5) & 
                (data['vol_ratio'] >= 1.3) &
                (~data['pct_change'].isna()) &
                (~data['vol_ratio'].isna())
            ].tail(5)
            
            if not significant.empty:
                print(f"\n📊 === {symbol} ===")
                for idx, row in significant.iterrows():
                    direction = "📈" if row['pct_change'] > 0 else "📉"
                    print(f"  {idx.strftime('%m-%d %H:%M')} - ${row['Close']:.2f} {direction} {row['pct_change']:+.2f}% - Vol: {row['vol_ratio']:.1f}x")
                    
                    # Calculer si c'est près des highs/lows récents
                    recent_data = data.loc[:idx].tail(20)
                    if len(recent_data) >= 10:
                        recent_high = recent_data['High'].max()
                        recent_low = recent_data['Low'].min()
                        
                        if row['Close'] >= recent_high * 0.99:
                            print(f"    💡 Proche du récent high ${recent_high:.2f}")
                        elif row['Close'] <= recent_low * 1.01:
                            print(f"    💡 Proche du récent low ${recent_low:.2f}")
                            
        except Exception as e:
            print(f"❌ Erreur {symbol}: {e}")

if __name__ == "__main__":
    find_breakout_examples()