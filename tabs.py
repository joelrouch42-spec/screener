#!/usr/bin/env python3
"""
Multi-Symbol Watchlist Dashboard with Tabs
Corrected / hardened version
- Robust datetime index handling
- Defensive copies to avoid SettingWithCopyWarning
- Safer support/resistance detection (order auto-adjust)
- Logging instead of print
- Better error handling in config loading
- Cache protection and fallbacks for missing analyzers/providers
- Average movement (%) display added
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from scipy.signal import argrelextrema

# Local imports (must exist)
from data_providers import MultiSourceDataProvider
from catalyst_analyzer import CatalystAnalyzer, integrate_with_dashboard, load_settings

# Timezone
EST = ZoneInfo("America/New_York")

# ----------------------
# Logging
# ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ----------------------
# Cache System for S/R Levels
# ----------------------
class LevelsCache:
    """Cache des niveaux support/résistance pour éviter les recalculs constants"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = timedelta(hours=4)  # Cache 4h par défaut
        
    def get_levels(self, symbol: str):
        """Récupère les niveaux en cache s'ils sont valides"""
        if symbol not in self.cache:
            return None

        cache_entry = self.cache[symbol]
        now = datetime.now(EST)

        # Vérifier si le cache n'est pas expiré
        if now - cache_entry['timestamp'] > self.cache_duration:
            logger.info(f"🗑️  Cache expiré pour {symbol}")
            del self.cache[symbol]
            return None

        logger.info(f"📦 Cache HIT pour {symbol}")
        return cache_entry
        
    def set_levels(self, symbol: str, support_levels: List[float], resistance_levels: List[float],
                   df: pd.DataFrame, last_price: float):
        """Sauvegarde les niveaux en cache"""
        self.cache[symbol] = {
            'timestamp': datetime.now(EST),
            'support_levels': support_levels.copy() if isinstance(support_levels, list) else list(support_levels),
            'resistance_levels': resistance_levels.copy() if isinstance(resistance_levels, list) else list(resistance_levels),
            'last_price': last_price,
            'data_length': len(df),
            'price_range': {
                'high': float(df['High'].max()),
                'low': float(df['Low'].min())
            }
        }
        logger.info(f"💾 Cache SAVED pour {symbol} - {len(support_levels)} supports, {len(resistance_levels)} résistances")
        
    def should_refresh_levels(self, symbol: str, current_price: float, price_change_threshold: float = 0.05):
        """Détermine si on doit recalculer les niveaux (prix a beaucoup bougé)"""
        if symbol not in self.cache:
            return True
            
        cache_entry = self.cache[symbol]
        last_price = cache_entry['last_price']
        
        # Si le prix a bougé de plus de 5%, recalculer
        price_change = abs(current_price - last_price) / last_price
        if price_change > price_change_threshold:
            logger.info(f"📈 Prix {symbol} a bougé de {price_change:.1%}, refresh niveaux")
            return True
            
        return False
        
    def clear_cache(self):
        """Vide tout le cache"""
        self.cache.clear()
        logger.info("🗑️  Cache complètement vidé")

# Instance globale du cache
LEVELS_CACHE = LevelsCache()

# ----------------------
# API pour Scanner
# ----------------------
def get_latest_data_with_cached_levels(symbol: str, data_provider, settings: Dict = None) -> Dict:
    """
    API optimisée pour le scanner : récupère la dernière bougie + niveaux S/R en cache
    
    Returns:
        {
            'symbol': str,
            'latest_candle': pd.Series,
            'support_levels': List[float], 
            'resistance_levels': List[float],
            'provider_used': str,
            'cache_hit': bool
        }
    """
    try:
        # 1. Récupérer seulement la dernière bougie
        chart_data = ChartData(symbol, data_provider, days=2, period=2)
        latest_candle = chart_data.fetch_latest_candle()
        
        if latest_candle is None:
            logger.warning(f"Impossible de récupérer la dernière bougie pour {symbol}")
            return None
            
        current_price = float(latest_candle['Close'])
        
        # 2. Vérifier le cache des niveaux S/R
        cache_entry = LEVELS_CACHE.get_levels(symbol)
        cache_hit = False
        
        if cache_entry is None or LEVELS_CACHE.should_refresh_levels(symbol, current_price):
            # Recalculer les niveaux (besoin de l'historique complet)
            logger.info(f"🔄 Scanner: Recalcul niveaux S/R pour {symbol}")
            
            chart_data_full = ChartData(symbol, data_provider)
            chart_data_full.fetch_data()
            df_full = chart_data_full.df
            
            sr = SupportResistance()
            order = settings.get("support_resistance", {}).get("order", 5) if settings else 5
            support_levels, resistance_levels, _, _ = sr.find_levels(df_full, order)
            
            # Sauvegarder en cache
            LEVELS_CACHE.set_levels(symbol, support_levels, resistance_levels, df_full, current_price)
        else:
            # Utiliser les niveaux S/R en cache
            logger.info(f"📦 Scanner: Cache HIT S/R pour {symbol}")
            support_levels = cache_entry['support_levels']
            resistance_levels = cache_entry['resistance_levels']
            cache_hit = True
        
        return {
            'symbol': symbol,
            'latest_candle': latest_candle,
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'provider_used': chart_data.used_provider,
            'cache_hit': cache_hit
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur API scanner pour {symbol}: {e}")
        return None

# ----------------------
# Classes & helpers
# ----------------------
class ChartData:
    def __init__(self, symbol: str, data_provider, days: Optional[int] = None, period: Optional[int] = None):
        self.symbol = symbol
        
        # Utiliser settings globaux si paramètres non fournis
        if days is None or period is None:
            settings = SETTINGS if 'SETTINGS' in globals() and SETTINGS else {
                "data": {"days_fetch": 150, "period_candles": 120}
            }
            self.days = days or settings["data"]["days_fetch"]
            self.period = period or settings["data"]["period_candles"]
        else:
            self.days = days
            self.period = period
            
        self.df: Optional[pd.DataFrame] = None
        self.data_provider = data_provider
        self.used_provider: Optional[str] = None

    def fetch_data(self) -> pd.DataFrame:
        """Download stock/market data and normalize index/columns."""
        logger.info(" [%s] Téléchargement des données...", self.symbol)

        # Use provider to obtain a DataFrame and provider name
        df, used_provider = self.data_provider.fetch_data(self.symbol, self.days, self.period)

        if df is None or df.empty:
            raise ValueError(f"Aucune donnée récupérée pour {self.symbol}")

        # Make a copy before mutating
        df = df.copy()

        # Ensure a DatetimeIndex
        try:
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'Date' in df.columns:
                    df.index = pd.to_datetime(df['Date'], errors='coerce')
                    df.drop(columns=['Date'], inplace=True)
                else:
                    df.index = pd.to_datetime(df.index, errors='coerce')
            df = df.sort_index()
        except Exception as e:
            raise ValueError(f"Erreur conversion index datetime pour {self.symbol}: {e}")

        # Ensure numeric price columns and drop invalid rows
        for col in ('Open', 'High', 'Low', 'Close', 'Volume'):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])

        if len(df) < 2:
            raise ValueError(f"Trop peu de données ({len(df)} bougies, minimum 2 requis).")

        self.df = df
        self.used_provider = used_provider
        logger.info("✅ [%s] %d bougies récupérées (Provider: %s)", self.symbol, len(df), used_provider)
        return self.df
    
    def fetch_latest_candle(self) -> Optional[pd.Series]:
        """Récupère seulement la dernière bougie (optimisé pour cache)"""
        try:
            # Récupérer seulement les 2 dernières bougies
            df, used_provider = self.data_provider.fetch_data(self.symbol, days=2, period=2)
            
            if df is None or df.empty:
                logger.warning(f"Aucune donnée récente pour {self.symbol}")
                return None
                
            # Normaliser l'index
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'Date' in df.columns:
                    df.index = pd.to_datetime(df['Date'], errors='coerce')
                    df.drop(columns=['Date'], inplace=True)
                else:
                    df.index = pd.to_datetime(df.index, errors='coerce')
            df = df.sort_index()
            
            # Nettoyer les données
            for col in ('Open', 'High', 'Low', 'Close', 'Volume'):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
            
            if len(df) == 0:
                return None
                
            # Retourner la dernière bougie
            latest_candle = df.iloc[-1].copy()
            latest_candle.name = df.index[-1]  # Préserver le timestamp
            
            self.used_provider = used_provider
            return latest_candle
            
        except Exception as e:
            logger.error(f"Erreur récupération dernière bougie {self.symbol}: {e}")
            return None

    def calculate_metrics(self) -> pd.DataFrame:
        """Add/initialize columns used later in the pipeline."""
        if self.df is None:
            raise RuntimeError("Données non chargées pour calcul des métriques")

        df = self.df.copy()

        if 'Alert' not in df.columns:
            df['Alert'] = False
        if 'Breakout' not in df.columns:
            df['Breakout'] = ''
        if 'Breakout_Level' not in df.columns:
            df['Breakout_Level'] = 0.0

        self.df = df
        return self.df


class SupportResistance:
    """Support & resistance detection utilities."""

    @staticmethod
    def find_levels(df: pd.DataFrame, order: Optional[int] = None) -> Tuple[List[float], List[float], np.ndarray, np.ndarray]:
        # Utiliser settings si order non fourni
        if order is None:
            order = SETTINGS.get("support_resistance", {}).get("order", 5) if 'SETTINGS' in globals() and SETTINGS else 5
            
        highs = df['High'].values
        lows = df['Low'].values
        n = len(df)

        if order < 1:
            order = 1
        if n <= (2 * order):
            order = max(1, (n - 1) // 2)

        resistance_idx = argrelextrema(highs, np.greater, order=order)[0]
        support_idx = argrelextrema(lows, np.less, order=order)[0]

        resistance_levels = highs[resistance_idx] if resistance_idx.size else np.array([])
        support_levels = lows[support_idx] if support_idx.size else np.array([])

        def cluster_levels(levels: np.ndarray, threshold: Optional[float] = None) -> List[float]:
            if threshold is None:
                threshold = SETTINGS.get("support_resistance", {}).get("cluster_threshold", 0.02) if 'SETTINGS' in globals() and SETTINGS else 0.02
                
            if len(levels) == 0:
                return []
            levels_sorted = sorted(levels)
            clusters: List[float] = []
            current_cluster = [levels_sorted[0]]
            for level in levels_sorted[1:]:
                denom = current_cluster[-1] if current_cluster[-1] != 0 else 1.0
                if abs(level - current_cluster[-1]) / denom < threshold:
                    current_cluster.append(level)
                else:
                    clusters.append(float(np.mean(current_cluster)))
                    current_cluster = [level]
            clusters.append(float(np.mean(current_cluster)))
            return clusters

        support_clusters = cluster_levels(support_levels)
        resistance_clusters = cluster_levels(resistance_levels)

        return support_clusters, resistance_clusters, support_idx, resistance_idx

    @staticmethod
    def detect_breakouts(df: pd.DataFrame, support_levels: List[float], resistance_levels: List[float]) -> pd.DataFrame:
        df = df.copy()
        if 'Breakout' not in df.columns:
            df['Breakout'] = ''
        if 'Breakout_Level' not in df.columns:
            df['Breakout_Level'] = 0.0

        for i in range(1, len(df)):
            current_high = float(df['High'].iat[i])
            current_low = float(df['Low'].iat[i])
            prev_close = float(df['Close'].iat[i - 1])

            for resistance in resistance_levels:
                if prev_close < resistance and current_high > resistance:
                    df.iat[i, df.columns.get_loc('Breakout')] = 'RESISTANCE'
                    df.iat[i, df.columns.get_loc('Breakout_Level')] = float(resistance)
                    break

            for support in support_levels:
                if prev_close > support and current_low < support:
                    df.iat[i, df.columns.get_loc('Breakout')] = 'SUPPORT'
                    df.iat[i, df.columns.get_loc('Breakout_Level')] = float(support)
                    break

        return df


# ----------------------
# Chart creation
# ----------------------
def create_chart(symbol: str, df: pd.DataFrame, support_levels: List[float], resistance_levels: List[float]) -> go.Figure:
    fig = make_subplots(rows=1, cols=1, subplot_titles=(f'{symbol}',))

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Prix',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )

    try:
        for support in support_levels:
            fig.add_hline(y=support, line_dash="dash", line_color="green", 
                          line_width=SETTINGS.get("support_resistance", {}).get("line_width", 2), 
                          opacity=SETTINGS.get("support_resistance", {}).get("line_opacity", 0.6), row=1, col=1)
        for resistance in resistance_levels:
            fig.add_hline(y=resistance, line_dash="dash", line_color="red", 
                          line_width=SETTINGS.get("support_resistance", {}).get("line_width", 2), 
                          opacity=SETTINGS.get("support_resistance", {}).get("line_opacity", 0.6), row=1, col=1)
        
        # Ligne jaune pour le prix actuel
        current_price = float(df['Close'].iloc[-1])
        fig.add_hline(y=current_price, line_dash="solid", 
                      line_color=SETTINGS.get("chart", {}).get("current_price_line_color", "yellow"), 
                      line_width=SETTINGS.get("chart", {}).get("current_price_line_width", 2), 
                      opacity=SETTINGS.get("chart", {}).get("current_price_line_opacity", 0.8), row=1, col=1)
        
    except Exception:
        for support in support_levels:
            fig.add_shape(type='line', x0=df.index[0], x1=df.index[-1], y0=support, y1=support, line=dict(dash='dash'))
        for resistance in resistance_levels:
            fig.add_shape(type='line', x0=df.index[0], x1=df.index[-1], y0=resistance, y1=resistance, line=dict(dash='dash'))
        
        # Ligne jaune pour le prix actuel (fallback)
        current_price = float(df['Close'].iloc[-1])
        fig.add_shape(type='line', x0=df.index[0], x1=df.index[-1], y0=current_price, y1=current_price, 
                      line=dict(
                          color=SETTINGS.get("chart", {}).get("current_price_line_color", "yellow"),
                          width=SETTINGS.get("chart", {}).get("current_price_line_width", 2)
                      ))

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=SETTINGS.get("interface", {}).get("chart_height", 900),
        template='plotly_dark',
        hovermode='x unified',
        showlegend=True,
        margin=dict(
            t=SETTINGS.get("chart", {}).get("margin_top", 50),
            b=SETTINGS.get("chart", {}).get("margin_bottom", 50),
            l=SETTINGS.get("chart", {}).get("margin_left", 50),
            r=SETTINGS.get("chart", {}).get("margin_right", 50)
        )
    )

    fig.update_yaxes(title_text="Prix ($)")
    fig.update_xaxes(title_text="Date")

    return fig


# ----------------------
# Globals & config
# ----------------------
SYMBOLS_CONFIG: Dict[str, dict] = {}
DATA_PROVIDER = None
# SYMBOL_DATA: Dict[str, dict] = {}  # CACHE SUPPRIMÉ
CATALYST_ANALYZER = None
SECTOR_MAP: Dict[str, str] = {}
SETTINGS = {}  # Configuration globale

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])


# ----------------------
# Layout
# ----------------------
def create_layout(watchlist: List[str]):
    tabs = []
    for symbol in watchlist:
        tabs.append(
            dbc.Tab(
                label=symbol,
                tab_id=f"tab-{symbol}",
                label_style={"color": "#ffffff"},
                active_label_style={"color": "#26a69a"}
            )
        )

    return dbc.Container([
        dbc.Row([dbc.Col([html.H3(" Watchlist Dashboard", className="text-center mb-3")])]),
        dbc.Row([dbc.Col([dbc.Tabs(children=tabs, id="symbol-tabs", active_tab=f"tab-{watchlist[0]}" if watchlist else None)])], className="mb-3"),
        # Stats
dbc.Row([
    dbc.Col(dbc.Card(dbc.CardBody([
        html.H5(" Prix actuel", className="card-title"),
        html.H3(id='current-price', className="text-success")
    ])), width=2),
    dbc.Col(dbc.Card(dbc.CardBody([
        html.H5(" Close précédent", className="card-title"),
        html.H3(id='prev-close', className="text-info")
    ])), width=2),
    dbc.Col(dbc.Card(dbc.CardBody([
        html.H5("%Δ Change", className="card-title"),
        html.H3(id='price-change', className="text-warning")
    ])), width=2),
    dbc.Col(dbc.Card(dbc.CardBody([
        html.H5("Δ% Moy/Seuil", className="card-title"),
        html.H3(id='avg-move', className="text-warning")  # <-- couleur changée
    ])), width=2),
    dbc.Col(dbc.Card(dbc.CardBody([
        html.H5(" Catalyseurs", className="card-title"),
        html.H3(id='alerts-count', className="text-warning")
    ])), width=2),
    dbc.Col(dbc.Card(dbc.CardBody([
        html.H5("📊 Cata Log", className="card-title"),
        html.Div(id='cata-log', className="text-info", style={'maxHeight': '100px', 'overflowY': 'auto', 'fontSize': '16px'})
    ])), width=2),
], className="mb-3"),
        dbc.Row([dbc.Col(dcc.Graph(id='symbol-chart', style={'height': '900px'}))]),
        dbc.Row([dbc.Col([html.P(id='last-update', className="text-muted text-center"), html.P(" Auto-refresh toutes les 60 secondes", className="text-center text-sm")])]),
        dcc.Interval(id='interval-component', interval=SETTINGS.get("interface", {}).get("refresh_seconds", 60) * 1000, n_intervals=0)
    ], fluid=True, style={'backgroundColor': '#1a1a1a', 'minHeight': '100vh', 'padding': '20px'})


# ----------------------
# Catalyst log helper
# ----------------------
def create_catalyst_log(symbol: str, days: Optional[int] = None):
    if not CATALYST_ANALYZER or not hasattr(CATALYST_ANALYZER, 'get_recent_alerts'):
        return html.Div([html.P(" Catalyst Analyzer non disponible", style={'color': '#888', 'fontStyle': 'italic', 'margin': '0'})])

    try:
        days = days or SETTINGS.get("catalyst", {}).get("catalyst_log_days", 365)
        recent_catalysts = CATALYST_ANALYZER.get_recent_alerts(days=days)
        symbol_catalysts = [c for c in recent_catalysts if c.get('symbol') == symbol]
    except Exception as e:
        logger.exception("Erreur récupération catalysts pour %s: %s", symbol, e)
        return html.Div([html.P("❌ Erreur lors de la récupération des catalyseurs", style={'color': '#ff6666', 'fontStyle': 'italic', 'margin': '0'})])

    if not symbol_catalysts:
        return html.Div([html.P(" Aucun catalyseur détecté ces 7 derniers jours", style={'color': '#888', 'fontStyle': 'italic', 'margin': '0'})])

    symbol_catalysts.sort(key=lambda x: x.get('timestamp', 0), reverse=True)

    entries = []
    for c in symbol_catalysts:
        change_pct = c.get('change_pct', 0.0)
        cat_type = c.get('catalyst', {}).get('type', 'unknown')
        date_str = c.get('date', '')
        try:
            movement_date = datetime.strptime(date_str, '%Y-%m-%d').strftime('%d/%m')
        except Exception:
            movement_date = date_str[-5:]

        entry = html.Div([
            html.Span(f"{movement_date}: ", style={'color': '#888', 'fontSize': '0.9em'}),
            html.Span(f"{change_pct:+.1f}% ", style={'color': '#00ff00' if change_pct > 0 else '#ff4444', 'fontWeight': 'bold'}),
            html.Span(f"{cat_type}", style={'color': '#ccc'})
        ], style={'marginBottom': '4px', 'fontSize': '0.9em'})
        entries.append(entry)

    return html.Div(entries)


# ----------------------
# Callback
# ----------------------
@app.callback(
    [Output('symbol-chart', 'figure'),
     Output('current-price', 'children'),
     Output('prev-close', 'children'),
     Output('price-change', 'children'),
     Output('avg-move', 'children'),   # <-- ajouté
     Output('alerts-count', 'children'),
     Output('cata-log', 'children'),
     Output('last-update', 'children')],
    [Input('symbol-tabs', 'active_tab'), Input('interval-component', 'n_intervals')]
)

def update_display(active_tab: Optional[str], n: int):
    if not active_tab or not isinstance(active_tab, str) or not active_tab.startswith('tab-'):
        empty_fig = go.Figure()
        empty_fig.update_layout(template='plotly_dark')
        return empty_fig, "-", "-", "-", "-", "-", "-", "-"

    symbol = active_tab.replace('tab-', '')
    config = SYMBOLS_CONFIG.get(symbol)
    if not config:
        empty_fig = go.Figure()
        empty_fig.update_layout(template='plotly_dark', title=f"Config manquante pour {symbol}")
        return empty_fig, "Erreur", "Erreur", "Erreur", "-", "Erreur", html.Div("Erreur"), f"Config manquante"

    try:
        # SYSTÈME HYBRIDE : Historique complet + Cache S/R
        chart_data = ChartData(symbol, DATA_PROVIDER)
        
        # 1. TOUJOURS charger l'historique complet (pour le graphique)
        chart_data.fetch_data()
        chart_data.calculate_metrics()
        df = chart_data.df
        logger.info(f"📊 {symbol}: {len(df)} bougies récupérées (demandé: {SETTINGS['data']['period_candles']})")
        
        current_price = float(df['Close'].iloc[-1])
        
        # 2. Vérifier le cache des niveaux S/R
        cache_entry = LEVELS_CACHE.get_levels(symbol)
        
        if cache_entry is None or LEVELS_CACHE.should_refresh_levels(symbol, current_price):
            # Recalculer les niveaux S/R
            logger.info(f"🔄 Recalcul niveaux S/R pour {symbol}")
            sr = SupportResistance()
            support_levels, resistance_levels, _, _ = sr.find_levels(df)
            
            # Sauvegarder en cache
            LEVELS_CACHE.set_levels(symbol, support_levels, resistance_levels, df, current_price)
        else:
            # Utiliser les niveaux S/R en cache
            logger.info(f"📦 Cache HIT S/R pour {symbol}")
            support_levels = cache_entry['support_levels']
            resistance_levels = cache_entry['resistance_levels']
        
        # 3. Détecter les breakouts avec les niveaux (cache ou recalculés)
        sr = SupportResistance()
        df = sr.detect_breakouts(df, support_levels, resistance_levels)

        # ANALYSE CATALYST TEMPS RÉEL
        catalyst_result = None
        if CATALYST_ANALYZER:
            try:
                catalyst_result = integrate_with_dashboard(symbol, df, CATALYST_ANALYZER, SECTOR_MAP)
                if catalyst_result:
                    logger.info(f'✅ Catalyseur détecté pour {symbol}: {catalyst_result.get("catalyst", {}).get("type")}')
            except Exception as e:
                logger.exception(f'❌ Erreur intégration catalyst pour {symbol}: {e}')

        last_fetch_time = datetime.now(EST).strftime('%H:%M:%S EST')

        live_price = float(df['Close'].iat[-1])
        prev_close = float(df['Close'].iat[-2]) if len(df) >= 2 else live_price
        price_change_pct = ((live_price - prev_close) / prev_close * 100) if prev_close > 0 else 0.0

        # Moyenne mouvements absolus + seuil dynamique (MÊME CALCUL QUE ANALYZER)
        if len(df) > 1 and CATALYST_ANALYZER:
            avg_move = CATALYST_ANALYZER.calculate_average_move(df)
            multiplier = SETTINGS["catalyst"]["multiplier"]
            threshold = max(avg_move * multiplier, SETTINGS["catalyst"]["min_threshold"])
            avg_move_text = f"{avg_move:.2f}% / {threshold:.2f}%"
        else:
            avg_move_text = "-"

        catalyst_count = len([c for c in (CATALYST_ANALYZER.get_recent_alerts(days=SETTINGS["catalyst"]["catalyst_count_days"]) if CATALYST_ANALYZER else []) if c.get('symbol') == symbol])
        catalyst_log_content = create_catalyst_log(symbol, days=SETTINGS["catalyst"]["catalyst_log_days"])
        fig = create_chart(symbol, df, support_levels, resistance_levels)

        return (
            fig,
            f"${live_price:.2f}",
            f"${prev_close:.2f}",
            f"{price_change_pct:+.2f}%",
            avg_move_text,
            str(catalyst_count),
            catalyst_log_content,  # cata-log content
            f"Dernière mise à jour: {last_fetch_time}"
        )

    except Exception as e:
        logger.exception('❌ Erreur callback [%s]: %s', symbol, e)
        empty_fig = go.Figure()
        empty_fig.update_layout(template='plotly_dark', title=f"❌ Erreur {symbol}: {str(e)}")
        return empty_fig, "Erreur", "Erreur", "Erreur", "-", "Erreur", html.Div("Erreur"), f"Erreur: {str(e)}"


# ----------------------
# Config helpers
# ----------------------
def load_sector_mapping(file: str = 'sector_mapping.txt') -> Dict[str, str]:
    sector_map: Dict[str, str] = {}
    try:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    sector_map[parts[0].upper()] = parts[1].lower()
    except Exception:
        logger.warning('⚠️  Erreur lecture sector mapping')
    return sector_map


def parse_config(config_file: str = 'config.txt') -> Dict[str, dict]:
    symbols_config: Dict[str, dict] = {}
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 1:
                    symbol = parts[0].upper()
                    provider = parts[1].lower() if len(parts) >= 2 else 'auto'
                    symbols_config[symbol] = {'provider': provider}
    except Exception:
        logger.exception('❌ Erreur lecture config')
        raise
    return symbols_config


# ----------------------
# Main
# ----------------------
def main():
    global SYMBOLS_CONFIG, DATA_PROVIDER, CATALYST_ANALYZER, SECTOR_MAP, SETTINGS

    # Charger les settings en premier
    SETTINGS = load_settings('settings.json')
    logger.info(f"📋 Settings: refresh={SETTINGS['interface']['refresh_seconds']}s, multiplier={SETTINGS['catalyst']['multiplier']}")

    config_file = sys.argv[1] if len(sys.argv) > 1 else 'config.txt'
    SYMBOLS_CONFIG = parse_config(config_file)
    if not SYMBOLS_CONFIG:
        logger.error('❌ Aucun symbole dans la configuration!')
        sys.exit(1)

    watchlist = list(SYMBOLS_CONFIG.keys())
    SECTOR_MAP = load_sector_mapping('sector_mapping.txt')

    # Déterminer le mode : replay_mode = backtest_mode
    backtest_mode = SETTINGS.get('debug', {}).get('replay_mode', True)

    try:
        DATA_PROVIDER = MultiSourceDataProvider(backtest_mode=backtest_mode)
    except Exception:
        logger.exception('❌ ERREUR INITIALISATION DATA PROVIDER')
        sys.exit(1)

    try:
        CATALYST_ANALYZER = CatalystAnalyzer(settings=SETTINGS)
        
        # ANALYSE HISTORIQUE AU DÉMARRAGE
        if CATALYST_ANALYZER:
            logger.info('🔍 DEBUT ANALYSE HISTORIQUE...')
            for symbol in watchlist:
                logger.info(f'  Analyse {symbol}...')
                sector = SECTOR_MAP.get(symbol, 'unknown')
                try:
                    df, provider_used = DATA_PROVIDER.fetch_data(
                        symbol, 
                        days=SETTINGS["data"]["days_fetch"], 
                        period=SETTINGS["data"]["period_candles"]
                    )
                    if df is not None and len(df) > 1:
                        CATALYST_ANALYZER.analyze_all_candles(symbol, df, sector)
                    else:
                        logger.warning(f'  ⚠️  Pas assez de données pour {symbol}')
                except Exception as e:
                    logger.exception(f'⚠️  ERREUR ANALYSE {symbol}: {e}')
            logger.info('✅ FIN ANALYSE HISTORIQUE')
        
    except Exception:
        logger.exception('⚠️  Erreur initialisation Catalyst Analyzer')
        CATALYST_ANALYZER = None

    app.layout = create_layout(watchlist)
    logger.info('\n Ouvrir: http://127.0.0.1:8050')
    app.run(
        debug=SETTINGS["server"]["debug"], 
        host=SETTINGS["server"]["host"], 
        port=SETTINGS["server"]["port"]
    )


if __name__ == '__main__':
    main()
