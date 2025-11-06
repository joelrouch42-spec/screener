#!/usr/bin/env python3
"""
Stock Scanner - Surveillance continue sans GUI
Détecte les catalyseurs et affiche les alertes en temps réel
Version CLI du système de détection de catalyseurs

REFACTORED VERSION with:
- Thread safety
- Rate limiting
- Parallel scanning
- Better error handling
- Input validation
- Performance optimizations
- Metrics tracking
"""

import logging
import sys
import time
import json
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from threading import Thread, Event, Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

# Local imports
from data_providers import MultiSourceDataProvider
from catalyst_analyzer import CatalystAnalyzer, load_settings
from tabs import get_latest_data_with_cached_levels

# ----------------------
# Constants
# ----------------------
EST = ZoneInfo("America/New_York")

# Default settings
DEFAULT_MIN_MOVE_PERCENT = 1.0
DEFAULT_VOLUME_SPIKE_THRESHOLD = 1.5
DEFAULT_MAX_WORKERS = 5
DEFAULT_RATE_LIMIT_CALLS = 50
DEFAULT_RATE_LIMIT_PERIOD = 60.0  # seconds
DEFAULT_SYMBOL_DELAY = 0.5  # seconds between symbols

# Validation patterns
SYMBOL_PATTERN = re.compile(r'^[A-Z]{1,5}$')
VALID_PROVIDERS = {'auto', 'polygon', 'ibkr', 'yahoo'}

# Market hours
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

# Alert cooldowns
TECHNICAL_ALERT_COOLDOWN = timedelta(minutes=30)
CATALYST_ALERT_COOLDOWN = timedelta(hours=2)
ALERT_CLEANUP_INTERVAL = timedelta(hours=24)

# ----------------------
# Logging
# ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------------
# Rate Limiter
# ----------------------
class RateLimiter:
    """Rate limiter to prevent API throttling"""

    def __init__(self, max_calls: int = DEFAULT_RATE_LIMIT_CALLS,
                 period: float = DEFAULT_RATE_LIMIT_PERIOD):
        """
        Args:
            max_calls: Maximum number of calls allowed in the period
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.calls: List[float] = []
        self.lock = Lock()

    def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded"""
        with self.lock:
            now = time.time()

            # Remove old calls outside the window
            self.calls = [t for t in self.calls if now - t < self.period]

            # If at limit, wait until oldest call expires
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0]) + 0.1
                if sleep_time > 0:
                    logger.debug(f"Rate limit reached, sleeping {sleep_time:.1f}s")
                    time.sleep(sleep_time)
                    now = time.time()
                    self.calls = [t for t in self.calls if now - t < self.period]

            # Record this call
            self.calls.append(now)

# ----------------------
# Input Validation
# ----------------------
def validate_symbol(symbol: str) -> bool:
    """Validate stock symbol format"""
    return bool(SYMBOL_PATTERN.match(symbol))

def validate_provider(provider: str) -> bool:
    """Validate data provider name"""
    return provider.lower() in VALID_PROVIDERS

def validate_config_path(path: str) -> str:
    """
    Validate and resolve config file path

    Args:
        path: Path to config file

    Returns:
        Absolute path to config file

    Raises:
        ValueError: If path is invalid
    """
    abs_path = os.path.abspath(path)

    if not os.path.isfile(abs_path):
        raise ValueError(f"Config file not found: {abs_path}")

    if not abs_path.endswith('.txt'):
        raise ValueError("Config file must be a .txt file")

    return abs_path

# ----------------------
# Support/Resistance Detection for Scanner
# ----------------------
def detect_scanner_breakouts(
    df: pd.DataFrame,
    support_levels: List[float],
    resistance_levels: List[float],
    settings: Dict = None,
    check_volume_spike: bool = True
) -> Optional[Dict]:
    """
    Detect support/resistance breakouts on latest candle - VERSION SCANNER

    Args:
        df: DataFrame with OHLCV data (minimum 2 rows)
        support_levels: List of support price levels
        resistance_levels: List of resistance price levels
        settings: Optional settings dict with breakout configuration
        check_volume_spike: If False, ignore volume spike requirement (for backtest)

    Returns:
        Dict with breakout info if detected, None otherwise
    """
    if len(df) < 2:
        return None

    # Current and previous candle
    current = df.iloc[-1]
    previous = df.iloc[-2]

    current_high = current['High']
    current_low = current['Low']
    current_close = current['Close']
    prev_close = previous['Close']

    # Guard against invalid prices
    if prev_close <= 0:
        logger.warning("Previous close price is invalid (<=0)")
        return None

    # Calculate % movement
    change_pct = ((current_close - prev_close) / prev_close) * 100

    # Configurable parameters
    if settings:
        min_move = settings.get("breakout", {}).get("min_move_percent", DEFAULT_MIN_MOVE_PERCENT)
        vol_threshold = settings.get("breakout", {}).get("volume_spike_threshold", DEFAULT_VOLUME_SPIKE_THRESHOLD)
    else:
        min_move = DEFAULT_MIN_MOVE_PERCENT
        vol_threshold = DEFAULT_VOLUME_SPIKE_THRESHOLD

    # Volume spike check (if available)
    volume_spike = False
    volume_ratio = 0.0
    if 'Volume' in df.columns and len(df) >= 6:
        current_vol = current['Volume']
        # Average of 5 candles BEFORE current (exclude current)
        avg_vol = df['Volume'].iloc[-6:-1].mean()
        volume_ratio = current_vol / avg_vol if avg_vol > 0 else 0.0
        volume_spike = volume_ratio >= vol_threshold

    # Debug logs (commented out for production)
    # logger.info(f"Breakout check: change={change_pct:.2f}%, vol_ratio={volume_ratio:.2f}x, spike={volume_spike}, S={len(support_levels)}, R={len(resistance_levels)}")
    # logger.info(f"  Prices: prev_close=${prev_close:.2f}, current: L=${current_low:.2f} H=${current_high:.2f} C=${current_close:.2f}")
    # if len(resistance_levels) > 0:
    #     logger.info(f"  Resistances: {[f'${r:.2f}' for r in resistance_levels[:3]]}")
    # if len(support_levels) > 0:
    #     logger.info(f"  Supports: {[f'${s:.2f}' for s in support_levels[:3]]}")

    # Check resistance breakouts
    for resistance in resistance_levels:
        if (prev_close < resistance and
            current_high > resistance and
            current_close > resistance and  # Close above as well
            change_pct >= min_move and  # Configurable threshold
            (not check_volume_spike or volume_spike)):  # Volume check (optional in backtest)
            return {
                'type': 'resistance_breakout',
                'level': float(resistance),
                'direction': 'UP',
                'description': f'Breakout résistance à ${resistance:.2f}'
            }

    # Check support breakdowns
    for support in support_levels:
        if (prev_close > support and
            current_low < support and
            current_close < support and  # Close below as well
            change_pct <= -min_move and  # Configurable threshold
            (not check_volume_spike or volume_spike)):  # Volume check (optional in backtest)
            return {
                'type': 'support_breakdown',
                'level': float(support),
                'direction': 'DOWN',
                'description': f'Breakdown support à ${support:.2f}'
            }

    return None

# ----------------------
# Metrics Tracker
# ----------------------
class MetricsTracker:
    """Track scanner performance metrics"""

    def __init__(self):
        self.lock = Lock()
        self.reset()

    def reset(self):
        """Reset all metrics"""
        with self.lock:
            self.total_scans = 0
            self.total_symbols_scanned = 0
            self.total_alerts = 0
            self.total_technical_alerts = 0
            self.total_catalyst_alerts = 0
            self.total_errors = 0
            self.total_api_calls = 0
            self.cache_hits = 0
            self.cache_misses = 0
            self.scan_times: List[float] = []
            self.start_time = time.time()

    def record_scan(self, duration: float, symbols_count: int):
        """Record a scan completion"""
        with self.lock:
            self.total_scans += 1
            self.total_symbols_scanned += symbols_count
            self.scan_times.append(duration)
            # Keep only last 100 scan times
            if len(self.scan_times) > 100:
                self.scan_times = self.scan_times[-100:]

    def record_alert(self, is_technical: bool):
        """Record an alert"""
        with self.lock:
            self.total_alerts += 1
            if is_technical:
                self.total_technical_alerts += 1
            else:
                self.total_catalyst_alerts += 1

    def record_error(self):
        """Record an error"""
        with self.lock:
            self.total_errors += 1

    def record_api_call(self):
        """Record an API call"""
        with self.lock:
            self.total_api_calls += 1

    def record_cache_hit(self, hit: bool):
        """Record cache hit or miss"""
        with self.lock:
            if hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1

    def get_stats(self) -> Dict:
        """Get current statistics"""
        with self.lock:
            uptime = time.time() - self.start_time
            avg_scan_time = sum(self.scan_times) / len(self.scan_times) if self.scan_times else 0
            cache_total = self.cache_hits + self.cache_misses
            cache_hit_rate = (self.cache_hits / cache_total * 100) if cache_total > 0 else 0

            return {
                'uptime_seconds': uptime,
                'total_scans': self.total_scans,
                'total_symbols_scanned': self.total_symbols_scanned,
                'total_alerts': self.total_alerts,
                'technical_alerts': self.total_technical_alerts,
                'catalyst_alerts': self.total_catalyst_alerts,
                'total_errors': self.total_errors,
                'total_api_calls': self.total_api_calls,
                'avg_scan_time': avg_scan_time,
                'cache_hit_rate': cache_hit_rate,
            }

# ----------------------
# Stock Scanner Class
# ----------------------
class StockScanner:
    """Main scanner class with parallel scanning and rate limiting"""

    def __init__(self, config_file: str = 'config.txt'):
        self.config_file = validate_config_path(config_file)
        self.symbols_config = {}
        self.sector_map = {}
        self.data_provider = None
        self.catalyst_analyzer = None
        self.settings = {}
        self.is_running = False
        self.stop_event = Event()

        # Thread safety
        self.alerts_lock = Lock()
        self.recent_alerts: Dict[str, datetime] = {}

        # Rate limiting
        self.rate_limiter = None

        # Metrics
        self.metrics = MetricsTracker()

    def is_market_open(self, dt: Optional[datetime] = None) -> bool:
        """
        Check if US market is open

        Args:
            dt: Datetime to check (defaults to now)

        Returns:
            True if market is open
        """
        # Backtest mode: always open
        if self.settings.get("backtest", {}).get("enabled", False):
            return True

        dt = dt or datetime.now(EST)

        # Weekend
        if dt.weekday() > 4:  # Monday=0, Friday=4
            return False

        # Market hours: 9:30-16:00 EST
        market_open = dt.replace(
            hour=MARKET_OPEN_HOUR,
            minute=MARKET_OPEN_MINUTE,
            second=0,
            microsecond=0
        )
        market_close = dt.replace(
            hour=MARKET_CLOSE_HOUR,
            minute=MARKET_CLOSE_MINUTE,
            second=0,
            microsecond=0
        )

        return market_open <= dt <= market_close

    def load_configuration(self) -> bool:
        """
        Load configuration from files

        Returns:
            True if successful
        """
        try:
            # Load settings
            self.settings = load_settings('settings.json')

            # Control printouts
            self.alerts_only = self.settings.get("logging", {}).get("alerts_only", False)

            if not self.alerts_only:
                logger.info(f"📋 Settings: refresh={self.settings['interface']['refresh_seconds']}s")

            # Initialize rate limiter from settings
            rate_limit_config = self.settings.get("rate_limiting", {})
            max_calls = rate_limit_config.get("max_calls_per_minute", DEFAULT_RATE_LIMIT_CALLS)
            period = rate_limit_config.get("period_seconds", DEFAULT_RATE_LIMIT_PERIOD)
            self.rate_limiter = RateLimiter(max_calls=max_calls, period=period)

            # Load symbols
            self.symbols_config = self._parse_config(self.config_file)
            if not self.symbols_config:
                logger.error('❌ No symbols in configuration!')
                return False

            # Load sector mapping
            self.sector_map = self._load_sector_mapping('sector_mapping.txt')

            if not self.alerts_only:
                logger.info(f"📊 {len(self.symbols_config)} symbols to monitor: {list(self.symbols_config.keys())}")

            return True

        except Exception as e:
            logger.exception(f'❌ Error loading configuration: {e}')
            return False

    def _parse_config(self, config_file: str) -> Dict[str, dict]:
        """
        Parse configuration file

        Args:
            config_file: Path to config file

        Returns:
            Dict mapping symbol to config
        """
        symbols_config = {}

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    parts = line.split()
                    if len(parts) >= 1:
                        symbol = parts[0].upper()
                        provider = parts[1].lower() if len(parts) >= 2 else 'auto'

                        # Validate symbol
                        if not validate_symbol(symbol):
                            logger.warning(f"Line {line_num}: Invalid symbol '{symbol}' (must be 1-5 uppercase letters)")
                            continue

                        # Validate provider
                        if not validate_provider(provider):
                            logger.warning(f"Line {line_num}: Invalid provider '{provider}' for {symbol}, using 'auto'")
                            provider = 'auto'

                        symbols_config[symbol] = {'provider': provider}

        except FileNotFoundError:
            logger.error(f'❌ Config file not found: {config_file}')
        except Exception as e:
            logger.exception(f'❌ Error reading config: {e}')

        return symbols_config

    def _load_sector_mapping(self, file: str) -> Dict[str, str]:
        """
        Load sector mapping file

        Args:
            file: Path to sector mapping file

        Returns:
            Dict mapping symbol to sector
        """
        sector_map = {}

        try:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    parts = line.split()
                    if len(parts) >= 2:
                        symbol = parts[0].upper()
                        sector = parts[1].lower()

                        if validate_symbol(symbol):
                            sector_map[symbol] = sector

        except FileNotFoundError:
            logger.warning(f'⚠️  Sector mapping file not found: {file}')
        except Exception as e:
            logger.error(f'⚠️  Error reading sector mapping: {e}', exc_info=True)

        return sector_map

    def initialize_services(self) -> bool:
        """
        Initialize data providers and analyzers

        Returns:
            True if successful
        """
        try:
            # Debug mode based on alerts_only setting
            debug_mode = not self.alerts_only
            self.data_provider = MultiSourceDataProvider(debug=debug_mode)

            if not self.alerts_only:
                logger.info("✅ Data provider initialized")

        except Exception as e:
            logger.exception('❌ ERROR INITIALIZING DATA PROVIDER')
            return False

        try:
            self.catalyst_analyzer = CatalystAnalyzer(settings=self.settings)

            if not self.alerts_only:
                logger.info("✅ Catalyst analyzer initialized")

            # Reduce log verbosity if alerts_only
            if self.alerts_only:
                logging.getLogger('tabs').setLevel(logging.WARNING)
                logging.getLogger('catalyst_analyzer').setLevel(logging.WARNING)

        except Exception as e:
            logger.error('⚠️  Error initializing Catalyst Analyzer', exc_info=True)
            self.catalyst_analyzer = None

        return True

    def scan_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Scan a single symbol and return alert if detected (OPTIMIZED VERSION)

        Args:
            symbol: Stock symbol to scan

        Returns:
            Alert dict if detected, None otherwise
        """
        try:
            # Check if market is open
            if not self.is_market_open():
                return None

            # Rate limiting
            if self.rate_limiter:
                self.rate_limiter.wait_if_needed()

            sector = self.sector_map.get(symbol, 'unknown')

            # OPTIMIZED API: Get latest candle + cached S/R levels
            data_result = get_latest_data_with_cached_levels(symbol, self.data_provider, self.settings)

            if data_result is None:
                logger.warning(f'⚠️  Unable to fetch data for {symbol}')
                self.metrics.record_error()
                return None

            self.metrics.record_api_call()
            self.metrics.record_cache_hit(data_result['cache_hit'])

            latest_candle = data_result['latest_candle']
            support_levels = data_result['support_levels']
            resistance_levels = data_result['resistance_levels']
            provider_used = data_result['provider_used']
            cache_hit = data_result['cache_hit']

            if cache_hit:
                logger.debug(f"📦 {symbol}: Cache HIT S/R")
            else:
                logger.debug(f"🔄 {symbol}: Recalculating S/R")

            # OPTIMIZATION: Fetch full data ONCE and reuse
            df_full = None
            previous_price = None
            current_price = None

            try:
                # Backtest mode: calculate end_date based on iteration
                end_date = None
                if self.settings.get("backtest", {}).get("enabled", False):
                    iteration = self.settings.get("backtest", {}).get("iteration", 0)
                    end_date = datetime.now(EST) - timedelta(days=10 + iteration)

                df_full, _ = self.data_provider.fetch_data(
                    symbol,
                    days=self.settings["data"]["days_fetch"],
                    period=self.settings["data"]["period_candles"],
                    end_date=end_date
                )
                self.metrics.record_api_call()

                if df_full is not None and len(df_full) >= 2:
                    # In backtest mode, use prices from historical data
                    # In normal mode, also use df_full for consistency
                    current_price = float(df_full['Close'].iloc[-1])
                    previous_price = float(df_full['Close'].iloc[-2])
                else:
                    logger.warning(f"Insufficient data for {symbol}")
                    return None

            except Exception as e:
                logger.error(f"Error fetching full data for {symbol}: {e}", exc_info=True)
                self.metrics.record_error()
                return None

            # Validate prices
            if current_price is None or current_price <= 0 or previous_price is None or previous_price <= 0:
                logger.warning(f"Invalid prices for {symbol}: current={current_price}, previous={previous_price}")
                return None

            # 1. Analyze technical breakouts with cached levels
            breakout_info = None
            backtest_mode = self.settings.get("backtest", {}).get("enabled", False)
            if (len(support_levels) > 0 or len(resistance_levels) > 0) and len(df_full) >= 6:
                try:
                    # Use last 6 candles (need 5 for volume average + current)
                    df_mini = df_full.tail(6)
                    # logger.info(f"[{symbol}] df_full={len(df_full)} rows, df_mini={len(df_mini)} rows, S={len(support_levels)}, R={len(resistance_levels)}")
                    # In backtest mode, don't require volume spike
                    breakout_info = detect_scanner_breakouts(
                        df_mini,
                        support_levels,
                        resistance_levels,
                        self.settings,
                        check_volume_spike=not backtest_mode
                    )
                except Exception as e:
                    logger.error(f"Error detecting breakout for {symbol}: {e}", exc_info=True)
                    self.metrics.record_error()

            # 2. Analyze with catalyst analyzer (needs full DataFrame)
            catalyst_info = None
            if self.catalyst_analyzer and df_full is not None and len(df_full) >= 2:
                try:
                    catalyst_info = self.catalyst_analyzer.analyze_symbol(symbol, df_full, sector)
                except Exception as e:
                    logger.error(f"Error analyzing catalyst for {symbol}: {e}", exc_info=True)
                    self.metrics.record_error()

            # Apply alert mode filter and collect all alerts
            alert_mode = self.settings.get("alerts", {}).get("mode", "all")
            alerts_to_process = []

            if alert_mode == "technical_only":
                # Only technical breakouts
                if breakout_info:
                    alerts_to_process.append(breakout_info)
            elif alert_mode == "catalyst_only":
                # Only catalysts
                if catalyst_info:
                    alerts_to_process.append(catalyst_info)
            else:
                # "all" mode: show both if both exist
                if breakout_info:
                    alerts_to_process.append(breakout_info)
                if catalyst_info:
                    alerts_to_process.append(catalyst_info)

            # Process all detected alerts
            created_alerts = []
            for alert_info in alerts_to_process:
                # Determine alert type
                alert_type = alert_info.get('type', 'unknown')
                is_technical = alert_type in ['resistance_breakout', 'support_breakdown']

                # Avoid duplicate alerts (thread-safe)
                alert_key = f"{symbol}_{alert_type}"

                # In backtest mode, use the date of the last candle
                if self.settings.get("backtest", {}).get("enabled", False):
                    now = df_full.index[-1].to_pydatetime()
                    if now.tzinfo is None:
                        now = now.replace(tzinfo=EST)
                else:
                    now = datetime.now(EST)

                # Check if recent alert (last 30min for technical, 2h for catalyst)
                cooldown = TECHNICAL_ALERT_COOLDOWN if is_technical else CATALYST_ALERT_COOLDOWN

                with self.alerts_lock:
                    if alert_key in self.recent_alerts:
                        time_diff = now - self.recent_alerts[alert_key]
                        if time_diff < cooldown:
                            logger.debug(f"Skipping duplicate alert for {alert_key}")
                            continue  # Skip this alert, process next one

                    # Record this alert
                    self.recent_alerts[alert_key] = now

                # Create alert
                alert = {
                    'symbol': symbol,
                    'sector': sector,
                    'timestamp': now.isoformat(),
                    'catalyst': alert_info,
                    'provider': provider_used,
                    'current_price': current_price,
                    'previous_price': previous_price,
                    'is_technical': is_technical,
                    'support_levels': support_levels[-3:] if len(support_levels) > 0 else [],
                    'resistance_levels': resistance_levels[-3:] if len(resistance_levels) > 0 else [],
                    'cache_performance': 'HIT' if cache_hit else 'MISS'
                }

                # Record metrics
                self.metrics.record_alert(is_technical)

                created_alerts.append(alert)

            # Return all created alerts (list, may be empty)
            return created_alerts if created_alerts else None

        except Exception as e:
            logger.error(f'❌ Error scanning {symbol}: {e}', exc_info=True)
            self.metrics.record_error()

        return None

    def display_alert(self, alert: Dict):
        """
        Display formatted alert

        Args:
            alert: Alert dictionary
        """
        symbol = alert['symbol']
        catalyst = alert['catalyst']
        current_price = alert['current_price']
        previous_price = alert['previous_price']
        is_technical = alert.get('is_technical', False)

        # Safe division
        if previous_price > 0:
            change_pct = ((current_price - previous_price) / previous_price) * 100
        else:
            change_pct = 0.0

        change_direction = "📈" if change_pct > 0 else "📉"

        # Alert type
        alert_icon = "🔧" if is_technical else "🚨"
        alert_title = "BREAKOUT TECHNIQUE" if is_technical else "CATALYST DÉTECTÉ"

        # Use alert timestamp (important for backtest)
        alert_timestamp = datetime.fromisoformat(alert['timestamp'])

        print("\n" + "="*80)
        print(f"{alert_icon} {alert_title} - {alert_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        print(f"📊 Symbole: {symbol} ({alert['sector'].upper()})")
        print(f"💰 Prix: ${current_price:.2f} ({change_direction} {change_pct:+.2f}%)")
        print(f"🔍 Type: {catalyst.get('type', 'N/A').upper().replace('_', ' ')}")
        print(f"📝 Description: {catalyst.get('description', 'N/A')}")

        if is_technical:
            # Special display for breakouts
            level = catalyst.get('level', 0)
            direction = catalyst.get('direction', 'N/A')
            print(f"📏 Niveau: ${level:.2f}")
            print(f"➡️  Direction: {direction}")
            print(f"⚡ Signal: TECHNIQUE")

            # Display nearby levels
            if alert.get('support_levels'):
                supports = [f"${s:.2f}" for s in alert['support_levels']]
                print(f"🟢 Supports: {', '.join(supports)}")
            if alert.get('resistance_levels'):
                resistances = [f"${r:.2f}" for r in alert['resistance_levels']]
                print(f"🔴 Résistances: {', '.join(resistances)}")
        else:
            # Display for AI catalysts
            print(f"⭐ Fiabilité: {catalyst.get('reliability', 'N/A').upper()}")
            print(f"💼 Tradeable: {'✅ OUI' if catalyst.get('tradeable', False) else '❌ NON'}")
            print(f"🤖 Signal: INTELLIGENCE ARTIFICIELLE")

        print(f"📡 Source: {alert['provider']}")
        print("="*80)

        # Log for history
        signal_type = "TECHNIQUE" if is_technical else "IA"
        logger.info(f"{alert_icon} ALERTE {signal_type}: {symbol} {change_pct:+.2f}% - {catalyst.get('type', 'N/A')}")

    def scan_all_symbols(self) -> int:
        """
        Scan all symbols in parallel

        Returns:
            Number of alerts found
        """
        # Check market hours first
        if not self.is_market_open():
            current_time = datetime.now(EST).strftime('%H:%M:%S EST')
            if not self.alerts_only:
                print(f"⚠️  Marché fermé ({current_time}) - En attente d'ouverture ({MARKET_OPEN_HOUR}:{MARKET_OPEN_MINUTE:02d}-{MARKET_CLOSE_HOUR}:{MARKET_CLOSE_MINUTE:02d} EST)")
            return 0

        alerts_found = 0

        # Get max workers from settings or use default
        max_workers = self.settings.get("scanning", {}).get("max_workers", DEFAULT_MAX_WORKERS)

        # Parallel scanning with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all scan jobs
            future_to_symbol = {
                executor.submit(self.scan_symbol, symbol): symbol
                for symbol in self.symbols_config.keys()
            }

            # Process results as they complete
            for future in as_completed(future_to_symbol):
                if self.stop_event.is_set():
                    # Cancel remaining futures
                    for f in future_to_symbol:
                        f.cancel()
                    break

                symbol = future_to_symbol[future]

                try:
                    alerts = future.result()

                    if alerts:
                        # alerts is a list (can contain 1 or 2 alerts)
                        for alert in alerts:
                            self.display_alert(alert)
                            alerts_found += 1

                except Exception as e:
                    logger.error(f"❌ Error processing {symbol}: {e}", exc_info=True)
                    self.metrics.record_error()

        return alerts_found

    def run_continuous_scan(self):
        """Run continuous scanning loop (or single scan in backtest mode)"""
        # Check if backtest mode
        backtest_enabled = self.settings.get("backtest", {}).get("enabled", False)

        if backtest_enabled:
            iteration = self.settings.get("backtest", {}).get("iteration", 0)
            logger.info(f"🔍 BACKTEST MODE - Single scan (iteration {iteration})")
        else:
            logger.info("🚀 Starting continuous scanner...")
            logger.info(f"⏱️  Interval: {self.settings['interface']['refresh_seconds']} seconds")
            logger.info(f"🔧 Max workers: {self.settings.get('scanning', {}).get('max_workers', DEFAULT_MAX_WORKERS)}")
            logger.info("📊 Press Ctrl+C to stop")

        self.is_running = True
        scan_count = 0
        last_cleanup_time = time.time()
        cleanup_interval_seconds = ALERT_CLEANUP_INTERVAL.total_seconds()

        while self.is_running and not self.stop_event.is_set():
            scan_count += 1
            start_time = time.time()

            if not self.alerts_only:
                print(f"\n🔄 Scan #{scan_count} - {datetime.now(EST).strftime('%Y-%m-%d %H:%M:%S')}")
                print("-" * 50)

            alerts_found = self.scan_all_symbols()

            scan_duration = time.time() - start_time

            # Record metrics
            self.metrics.record_scan(scan_duration, len(self.symbols_config))

            if not self.alerts_only:
                if alerts_found == 0:
                    print(f"✅ Scan completed - No alerts ({scan_duration:.1f}s)")
                else:
                    print(f"🚨 Scan completed - {alerts_found} alert(s) detected ({scan_duration:.1f}s)")

                # Display metrics periodically
                if scan_count % 10 == 0:
                    self._display_metrics()

            # Periodic cleanup of old alerts
            current_time = time.time()
            if current_time - last_cleanup_time > cleanup_interval_seconds:
                self.cleanup_old_alerts()
                last_cleanup_time = current_time

            # BACKTEST MODE: Stop after one scan
            if backtest_enabled:
                logger.info("✅ Backtest scan completed - Exiting")
                break

            # Wait before next scan (interruptible)
            if self.stop_event.wait(self.settings['interface']['refresh_seconds']):
                break  # Stop requested

        logger.info("🛑 Scanner stopped")

    def _display_metrics(self):
        """Display current metrics"""
        stats = self.metrics.get_stats()

        print("\n" + "="*80)
        print("📊 METRICS")
        print("="*80)
        print(f"⏱️  Uptime: {stats['uptime_seconds']/3600:.1f}h")
        print(f"🔄 Total scans: {stats['total_scans']}")
        print(f"📈 Total symbols scanned: {stats['total_symbols_scanned']}")
        print(f"🚨 Total alerts: {stats['total_alerts']} (Technical: {stats['technical_alerts']}, Catalyst: {stats['catalyst_alerts']})")
        print(f"❌ Total errors: {stats['total_errors']}")
        print(f"📡 Total API calls: {stats['total_api_calls']}")
        print(f"⚡ Avg scan time: {stats['avg_scan_time']:.1f}s")
        print(f"📦 Cache hit rate: {stats['cache_hit_rate']:.1f}%")
        print("="*80)

    def stop(self):
        """Stop the scanner"""
        self.is_running = False
        self.stop_event.set()

    def cleanup_old_alerts(self):
        """Clean up old alerts (older than 24h)"""
        cutoff_time = datetime.now() - ALERT_CLEANUP_INTERVAL

        with self.alerts_lock:
            keys_to_remove = [
                key for key, timestamp in self.recent_alerts.items()
                if timestamp < cutoff_time
            ]

            for key in keys_to_remove:
                del self.recent_alerts[key]

            if keys_to_remove and not self.alerts_only:
                logger.info(f"🧹 Cleaned up {len(keys_to_remove)} old alerts")

# ----------------------
# Signal Handling
# ----------------------
def make_signal_handler(scanner: StockScanner):
    """
    Create signal handler with scanner context

    Args:
        scanner: StockScanner instance

    Returns:
        Signal handler function
    """
    def handler(signum, frame):
        """Handle shutdown signals"""
        print("\n🛑 Stop requested...")
        scanner.stop()

    return handler

# ----------------------
# Main Entry Point
# ----------------------
def main():
    """Main entry point"""
    print("📡 STOCK SCANNER - Surveillance Continue")
    print("=" * 80)
    print("REFACTORED VERSION with:")
    print("  ✅ Thread safety")
    print("  ✅ Rate limiting")
    print("  ✅ Parallel scanning")
    print("  ✅ Better error handling")
    print("  ✅ Input validation")
    print("  ✅ Performance optimizations")
    print("  ✅ Metrics tracking")
    print("=" * 80)

    # Get config file from args or use default
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'config.txt'

    try:
        # Create scanner
        scanner = StockScanner(config_file)

        # Setup signal handlers
        handler = make_signal_handler(scanner)
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

        # Load configuration
        if not scanner.load_configuration():
            sys.exit(1)

        # Initialize services
        if not scanner.initialize_services():
            sys.exit(1)

        # Clean up old alerts at startup
        scanner.cleanup_old_alerts()

        # Run continuous scan
        scanner.run_continuous_scan()

    except ValueError as e:
        logger.error(f"❌ Configuration error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⌨️  Keyboard interrupt detected")
    except Exception as e:
        logger.exception(f"❌ Fatal error: {e}")
        sys.exit(1)
    finally:
        if 'scanner' in locals():
            scanner.stop()
        print("✅ Scanner shut down cleanly")

if __name__ == '__main__':
    main()
