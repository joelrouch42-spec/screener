#!/usr/bin/env python3
"""
IBKR Data Provider for Stock Scanner
Uses Interactive Brokers API (ibapi) to fetch real-time and historical data
Based on working test code
"""

import time
import threading
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import pytz
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract, ContractDetails

# Timezone
EST = ZoneInfo("America/New_York")


class IBKRDataProvider(EWrapper, EClient):
    """
    Interactive Brokers data provider
    Fetches historical OHLCV data using IB API
    """

    def __init__(self, host="127.0.0.1", port=4002, client_id=1):
        """
        Initialize IBKR provider

        Args:
            host: IB Gateway/TWS host (default: 127.0.0.1)
            port: IB Gateway/TWS port (4002 for paper, 7497 for live TWS)
            client_id: Unique client ID
        """
        EClient.__init__(self, self)

        self.host = host
        self.port = port
        self.client_id = client_id

        # Data storage
        self.bars = []
        self.valid_contract = None
        self.error_occurred = False
        self.error_message = ""

        # Threading events
        self.contract_event = threading.Event()
        self.data_event = threading.Event()
        self.connected_event = threading.Event()

        # Connection thread
        self.api_thread = None

    def nextValidId(self, orderId):
        """Callback: Connection established"""
        super().nextValidId(orderId)
        self.connected_event.set()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        """Callback: Error handling"""
        # Non-critical information codes
        if errorCode in [2104, 2106, 2158, 2108, 2174, 2107]:
            return

        # Critical errors
        if errorCode in [200, 502, 504]:  # No security definition, connection errors
            self.error_occurred = True
            self.error_message = f"IBKR Error {errorCode}: {errorString}"
            self.contract_event.set()
            self.data_event.set()

    def contractDetails(self, reqId: int, contractDetails: ContractDetails):
        """Callback: Contract details received"""
        self.valid_contract = contractDetails.contract

    def contractDetailsEnd(self, reqId: int):
        """Callback: Contract search completed"""
        self.contract_event.set()

    def historicalData(self, reqId, bar):
        """Callback: Historical bar received"""
        if bar.barCount == 0:
            return

        self.bars.append({
            'Date': bar.date,
            'Open': bar.open,
            'High': bar.high,
            'Low': bar.low,
            'Close': bar.close,
            'Volume': bar.volume
        })

    def historicalDataEnd(self, reqId, start, end):
        """Callback: Historical data completed"""
        self.data_event.set()

    def connect_and_run(self):
        """Connect to IB Gateway/TWS and run message loop"""
        try:
            self.connect(self.host, self.port, self.client_id)
            self.run()
        except Exception as e:
            self.error_occurred = True
            self.error_message = f"Connection failed: {e}"
            self.connected_event.set()

    def fetch_data(self, symbol, days=80, period=60, end_date=None):
        """
        Fetch historical data from IBKR

        Args:
            symbol: Stock symbol (e.g., "AAPL")
            days: Not used (IBKR uses duration string)
            period: Number of candles to return
            end_date: End date for historical data (datetime object with timezone)

        Returns:
            DataFrame with OHLCV data
        """
        # Reset state
        self.bars = []
        self.valid_contract = None
        self.error_occurred = False
        self.error_message = ""
        self.contract_event.clear()
        self.data_event.clear()
        self.connected_event.clear()

        # Start connection in separate thread
        self.api_thread = threading.Thread(target=self.connect_and_run, daemon=True)
        self.api_thread.start()

        # Wait for connection
        if not self.connected_event.wait(timeout=10):
            raise ValueError(f"IBKR: Connection timeout for {symbol}")

        if self.error_occurred:
            raise ValueError(self.error_message)

        try:
            # Step 1: Search for contract
            search_contract = Contract()
            search_contract.symbol = symbol
            search_contract.secType = "STK"
            search_contract.exchange = "SMART"
            search_contract.currency = "USD"

            self.reqContractDetails(100, search_contract)

            # Wait for contract details
            if not self.contract_event.wait(timeout=10):
                raise ValueError(f"IBKR: Contract search timeout for {symbol}")

            if self.error_occurred:
                raise ValueError(self.error_message)

            if not self.valid_contract:
                raise ValueError(f"IBKR: No contract found for {symbol}")

            # Step 2: Request historical data
            # Format end date for IBKR (YYYYMMDD HH:MM:SS UTC)
            if end_date is None:
                utc_now = datetime.now(pytz.utc)
            else:
                # Convert end_date to UTC
                if end_date.tzinfo is None:
                    utc_now = EST.localize(end_date).astimezone(pytz.utc)
                else:
                    utc_now = end_date.astimezone(pytz.utc)

            # IBKR format: YYYYMMDD HH:MM:SS UTC
            end_datetime_str = utc_now.strftime("%Y%m%d %H:%M:%S") + " UTC"

            # Calculate duration to get approximately 'period' bars
            # Request more days to ensure we get enough data (account for weekends/holidays)
            duration_days = max(period + 60, 180)
            duration_str = f"{duration_days} D"

            self.reqHistoricalData(
                2,  # reqId
                self.valid_contract,
                end_datetime_str,
                duration_str,
                "1 day",
                "TRADES",
                0,  # useRTH (0 = all data, 1 = regular trading hours only)
                1,  # formatDate (1 = string format)
                False,  # keepUpToDate
                []  # chartOptions
            )

            # Wait for historical data
            if not self.data_event.wait(timeout=30):
                raise ValueError(f"IBKR: Historical data timeout for {symbol}")

            if self.error_occurred:
                raise ValueError(self.error_message)

            if not self.bars:
                raise ValueError(f"IBKR: No historical data for {symbol}")

            # Convert bars to DataFrame
            df = pd.DataFrame(self.bars)

            # Parse dates (IBKR returns string dates like "20231015")
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

            # Ensure columns are in correct order
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

            # Return last 'period' candles
            df = df.tail(period)

            return df

        finally:
            # Always disconnect
            self.disconnect()
            if self.api_thread and self.api_thread.is_alive():
                self.api_thread.join(timeout=2)

    def get_live_price(self, symbol):
        """Get current live price (not implemented yet)"""
        raise NotImplementedError("IBKR live price not implemented yet")

    def is_available(self):
        """Check if IBKR is available"""
        try:
            # Try to connect briefly
            test_event = threading.Event()

            def test_connect():
                try:
                    self.connect(self.host, self.port, self.client_id + 100)
                    test_event.set()
                    time.sleep(1)
                    self.disconnect()
                except:
                    pass

            test_thread = threading.Thread(target=test_connect, daemon=True)
            test_thread.start()

            return test_event.wait(timeout=5)
        except:
            return False


class IBKRProvider(IBKRDataProvider):
    """Alias for compatibility"""
    pass


# Test if run directly
if __name__ == "__main__":
    print("🧪 Testing IBKR Provider...")

    provider = IBKRProvider()

    try:
        print("\n📊 Fetching MSFT data...")
        df = provider.fetch_data("MSFT", days=150, period=120)

        if df is not None and not df.empty:
            print(f"✅ Success! Received {len(df)} candles")
            print("\nLast 5 candles:")
            print(df.tail())
            print(f"\nLatest close: ${df['Close'].iloc[-1]:.2f}")
        else:
            print("❌ No data received")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure IB Gateway or TWS is running on port 4002")
