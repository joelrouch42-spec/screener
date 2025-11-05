"""
Market utilities module - Common market functions
"""
from datetime import datetime
from typing import Optional, Dict
from zoneinfo import ZoneInfo

EST = ZoneInfo("America/New_York")

def is_market_open(dt: Optional[datetime] = None, settings: Dict = None) -> bool:
    """
    Check if US market is open
    
    Args:
        dt: Datetime to check (default: now)
        settings: Settings dict with market hours
        
    Returns:
        bool: True if market is open
    """
    dt = dt or datetime.now(EST)
    
    # Default market hours if no settings
    if not settings:
        settings = {
            "market": {
                "open_hour": 9,
                "open_minute": 30,
                "close_hour": 16,
                "close_minute": 0
            }
        }
    
    # Ensure correct timezone
    if dt.tzinfo is None:
        dt = EST.localize(dt)
    elif dt.tzinfo != EST:
        dt = dt.astimezone(EST)
    
    # Weekend check
    if dt.weekday() > 4:  # Saturday=5, Sunday=6
        return False
    
    # Market hours from settings
    market_open = dt.replace(
        hour=settings["market"]["open_hour"],
        minute=settings["market"]["open_minute"],
        second=0,
        microsecond=0
    )
    market_close = dt.replace(
        hour=settings["market"]["close_hour"],
        minute=settings["market"]["close_minute"],
        second=0,
        microsecond=0
    )
    
    return market_open <= dt <= market_close


def get_next_market_open(dt: Optional[datetime] = None, settings: Dict = None) -> datetime:
    """
    Get the next market open time
    
    Args:
        dt: Current datetime (default: now)
        settings: Settings dict with market hours
        
    Returns:
        datetime: Next market open time
    """
    dt = dt or datetime.now(EST)
    
    # Default market hours if no settings
    if not settings:
        settings = {
            "market": {
                "open_hour": 9,
                "open_minute": 30,
                "close_hour": 16,
                "close_minute": 0
            }
        }
    
    # Ensure correct timezone
    if dt.tzinfo is None:
        dt = EST.localize(dt)
    elif dt.tzinfo != EST:
        dt = dt.astimezone(EST)
    
    # Get next weekday
    while dt.weekday() > 4:  # Skip weekend
        dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        dt += timedelta(days=1)
    
    # Set market open time
    market_open = dt.replace(
        hour=settings["market"]["open_hour"],
        minute=settings["market"]["open_minute"],
        second=0,
        microsecond=0
    )
    
    # If already past today's market open, go to next day
    if dt >= market_open:
        dt += timedelta(days=1)
        # Skip weekend if needed
        while dt.weekday() > 4:
            dt += timedelta(days=1)
        market_open = dt.replace(
            hour=settings["market"]["open_hour"],
            minute=settings["market"]["open_minute"],
            second=0,
            microsecond=0
        )
    
    return market_open


def format_market_time(dt: datetime) -> str:
    """Format datetime for market display"""
    if dt.tzinfo != EST:
        dt = dt.astimezone(EST)
    return dt.strftime('%H:%M:%S EST')
