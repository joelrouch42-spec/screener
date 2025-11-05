#!/usr/bin/env python3
"""
Démo : Affichage des 5 dernières bougies avec données simulées
Montre comment le scanner traite les données de marché
"""

import sys
sys.path.insert(0, '/home/user/screener/venv/lib/python3.12/site-packages')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

EST = ZoneInfo("America/New_York")

def create_sample_candles(symbol='AAPL', num_candles=5, base_price=180.0):
    """Crée des bougies simulées pour démonstration"""

    # Partir d'une heure récente
    now = datetime.now(EST)
    # Ajuster pour être pendant les heures de marché
    base_time = now.replace(hour=14, minute=0, second=0, microsecond=0)

    dates = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []

    current_price = base_price

    for i in range(num_candles):
        # Timestamp (5 minutes par bougie)
        timestamp = base_time + timedelta(minutes=5*i)
        dates.append(timestamp)

        # Prix avec variation aléatoire
        open_price = current_price
        change = np.random.uniform(-0.5, 0.5)  # ±0.5%
        close_price = open_price * (1 + change/100)

        high_price = max(open_price, close_price) * (1 + np.random.uniform(0, 0.2)/100)
        low_price = min(open_price, close_price) * (1 - np.random.uniform(0, 0.2)/100)

        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        closes.append(close_price)
        volumes.append(np.random.randint(100000, 500000))

        current_price = close_price

    # Créer le DataFrame
    df = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes
    }, index=pd.DatetimeIndex(dates, name='Datetime'))

    return df

def display_candles(symbol, df):
    """Affiche les bougies formatées"""

    print(f"🔍 DEMO - Analyse des bougies pour {symbol}")
    print("=" * 80)
    print(f"📊 Nombre de bougies: {len(df)}")
    print(f"📅 Période: {df.index[0].strftime('%Y-%m-%d %H:%M')} → {df.index[-1].strftime('%Y-%m-%d %H:%M')}")
    print()

    print("📈 Détail des bougies (5 min):")
    print("-" * 80)

    for idx, row in df.iterrows():
        timestamp = idx.strftime('%Y-%m-%d %H:%M:%S')

        # Calculer le changement %
        change = ((row['Close'] - row['Open']) / row['Open']) * 100
        direction = "📈" if change > 0 else "📉"
        direction_text = "HAUSSE" if change > 0 else "BAISSE"

        print(f"\n⏰ {timestamp}")
        print(f"  Open:   ${row['Open']:.2f}")
        print(f"  High:   ${row['High']:.2f}")
        print(f"  Low:    ${row['Low']:.2f}")
        print(f"  Close:  ${row['Close']:.2f}")
        print(f"  Volume: {row['Volume']:,.0f}")
        print(f"  {direction} Change: {change:+.2f}% ({direction_text})")

    print()
    print("=" * 80)

    # Analyse de la tendance
    total_change = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
    trend_icon = "📈" if total_change > 0 else "📉"
    trend_text = "HAUSSIÈRE" if total_change > 0 else "BAISSIÈRE"

    print(f"\n💡 Analyse:")
    print(f"   Prix initial:  ${df['Close'].iloc[0]:.2f}")
    print(f"   Prix final:    ${df['Close'].iloc[-1]:.2f}")
    print(f"   {trend_icon} Tendance:    {trend_text} ({total_change:+.2f}%)")
    print(f"   Plus haut:     ${df['High'].max():.2f}")
    print(f"   Plus bas:      ${df['Low'].min():.2f}")
    print(f"   Volume total:  {df['Volume'].sum():,.0f}")

    # Détection de patterns simples
    print(f"\n🔍 Patterns détectés:")

    # Momentum
    last_3_changes = [(df['Close'].iloc[i] - df['Close'].iloc[i-1]) / df['Close'].iloc[i-1] * 100
                      for i in range(-3, 0)]
    avg_momentum = sum(last_3_changes) / len(last_3_changes)

    if avg_momentum > 0.1:
        print(f"   ✅ Momentum positif ({avg_momentum:.2f}%)")
    elif avg_momentum < -0.1:
        print(f"   ⚠️  Momentum négatif ({avg_momentum:.2f}%)")
    else:
        print(f"   ➡️  Consolidation ({avg_momentum:.2f}%)")

    # Volume
    avg_volume = df['Volume'].mean()
    last_volume = df['Volume'].iloc[-1]
    volume_ratio = last_volume / avg_volume

    if volume_ratio > 1.5:
        print(f"   🔥 Volume élevé (+{(volume_ratio-1)*100:.0f}% vs moyenne)")
    elif volume_ratio < 0.5:
        print(f"   💤 Volume faible (-{(1-volume_ratio)*100:.0f}% vs moyenne)")
    else:
        print(f"   ➡️  Volume normal")

def test_scanner_logic(symbol, df):
    """Teste la logique du scanner (comme dans scanner.py)"""

    print("\n" + "=" * 80)
    print("🤖 SIMULATION DU SCANNER")
    print("=" * 80)

    if len(df) < 2:
        print("❌ Pas assez de données (besoin de 2+ bougies)")
        return

    # Dernière et précédente bougie
    current = df.iloc[-1]
    previous = df.iloc[-2]

    print(f"\n📊 Analyse de la dernière bougie:")
    print(f"   Bougie précédente: ${previous['Close']:.2f}")
    print(f"   Bougie actuelle:   ${current['Close']:.2f}")

    # Calcul du changement
    change_pct = ((current['Close'] - previous['Close']) / previous['Close']) * 100

    print(f"   Changement: {change_pct:+.2f}%")

    # Simulation des seuils du scanner
    MIN_MOVE = 1.0  # Seuil configuré dans le scanner

    if abs(change_pct) >= MIN_MOVE:
        direction = "HAUSSE" if change_pct > 0 else "BAISSE"
        print(f"\n   🚨 SIGNAL: {direction} significative détectée!")
        print(f"   Mouvement de {abs(change_pct):.2f}% (seuil: {MIN_MOVE}%)")
    else:
        print(f"\n   ✅ Pas de signal (mouvement < {MIN_MOVE}%)")

    # Volume
    if len(df) >= 5:
        avg_volume = df['Volume'].iloc[-5:].mean()
        volume_spike = current['Volume'] / avg_volume

        print(f"\n📈 Volume:")
        print(f"   Actuel:  {current['Volume']:,.0f}")
        print(f"   Moyenne: {avg_volume:,.0f}")
        print(f"   Ratio:   {volume_spike:.2f}x")

        if volume_spike >= 1.5:
            print(f"   🔥 SPIKE de volume détecté!")

if __name__ == '__main__':
    # Paramètres
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'
    num_candles = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    base_price = float(sys.argv[3]) if len(sys.argv) > 3 else 180.0

    # Générer les données
    df = create_sample_candles(symbol, num_candles, base_price)

    # Afficher
    display_candles(symbol, df)

    # Tester la logique du scanner
    test_scanner_logic(symbol, df)

    print("\n" + "=" * 80)
    print("💡 NOTE: Ces données sont SIMULÉES pour démonstration")
    print("   Pour des données réelles, configure les clés API dans .env")
    print("=" * 80)
