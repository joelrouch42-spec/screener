#!/usr/bin/env python3
"""
Test rapide pour afficher les 5 dernières bougies d'un symbole
Utile pour vérifier que les data providers fonctionnent
"""

import sys
sys.path.insert(0, '/home/user/screener/venv/lib/python3.12/site-packages')

from data_providers import MultiSourceDataProvider
from datetime import datetime
from zoneinfo import ZoneInfo

EST = ZoneInfo("America/New_York")

def test_latest_candles(symbol='AAPL', num_candles=5):
    """Récupère et affiche les dernières bougies d'un symbole"""

    print(f"🔍 Récupération des {num_candles} dernières bougies pour {symbol}")
    print("=" * 80)

    # Initialiser le data provider
    provider = MultiSourceDataProvider(debug=True)

    # Récupérer les données (2 jours pour avoir au moins 5 bougies)
    df, source = provider.fetch_data(symbol, days=2, period=5)

    if df is None or len(df) == 0:
        print(f"❌ Aucune donnée disponible pour {symbol}")
        return

    print(f"✅ Source de données: {source}")
    print(f"📊 Total de bougies récupérées: {len(df)}")
    print()

    # Afficher les dernières bougies
    last_candles = df.tail(num_candles)

    print(f"📈 Les {num_candles} dernières bougies (5 min):")
    print("-" * 80)

    for idx, row in last_candles.iterrows():
        timestamp = idx.strftime('%Y-%m-%d %H:%M:%S')

        # Calculer le changement %
        change = ((row['Close'] - row['Open']) / row['Open']) * 100
        direction = "📈" if change > 0 else "📉"

        print(f"\n{timestamp}")
        print(f"  Open:   ${row['Open']:.2f}")
        print(f"  High:   ${row['High']:.2f}")
        print(f"  Low:    ${row['Low']:.2f}")
        print(f"  Close:  ${row['Close']:.2f}")
        if 'Volume' in row:
            volume_str = f"{row['Volume']:,.0f}" if row['Volume'] > 0 else "N/A"
            print(f"  Volume: {volume_str}")
        print(f"  Change: {direction} {change:+.2f}%")

    print()
    print("=" * 80)

    # Info sur la dernière bougie
    latest = df.iloc[-1]
    latest_time = df.index[-1].strftime('%H:%M:%S')

    print(f"\n💡 Dernière bougie à {latest_time} EST:")
    print(f"   Prix actuel: ${latest['Close']:.2f}")

    # Vérifier si c'est récent
    now = datetime.now(EST)
    age = now - df.index[-1].tz_convert(EST)
    age_minutes = age.total_seconds() / 60

    if age_minutes < 10:
        print(f"   ✅ Données récentes (il y a {age_minutes:.1f} min)")
    else:
        print(f"   ⚠️  Données anciennes (il y a {age_minutes:.0f} min)")
        if now.weekday() > 4:
            print("   💡 Weekend - marché fermé")
        else:
            current_hour = now.hour
            if current_hour < 9 or current_hour >= 16:
                print("   💡 Hors heures de marché (9h30-16h EST)")

if __name__ == '__main__':
    # Symbole par défaut ou passé en argument
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'
    num_candles = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    test_latest_candles(symbol, num_candles)
