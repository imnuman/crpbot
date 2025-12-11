"""
News Event Filter - Avoid trading around high-impact economic events.

This filter prevents trading during:
- 30 minutes before high-impact events
- 15 minutes after high-impact events
- During known high-volatility windows (NFP, FOMC, etc.)

Usage:
    from libs.hydra.ftmo_bots.news_filter import NewsFilter, should_avoid_news

    filter = NewsFilter()
    if filter.should_avoid(symbol="XAUUSD"):
        logger.warning("Avoiding trade due to upcoming news event")
        return None

Phase 3 Implementation - 2025-12-11
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set
from loguru import logger


@dataclass
class NewsEvent:
    """Represents a scheduled economic news event."""
    event_time: datetime
    event_name: str
    currency: str
    impact: str  # "high", "medium", "low"
    affected_symbols: List[str] = field(default_factory=list)


# High impact events that cause significant volatility
HIGH_IMPACT_KEYWORDS = [
    "Non-Farm Payrolls", "NFP",
    "FOMC", "Fed Interest Rate", "Federal Reserve",
    "CPI", "Consumer Price Index", "Inflation",
    "GDP",
    "ECB Interest Rate", "ECB Press Conference",
    "BOE Interest Rate", "Bank of England",
    "Unemployment Rate",
    "Retail Sales",
    "PMI",
]

# Currency to symbols mapping
CURRENCY_SYMBOLS = {
    "USD": ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "US30.cash", "US100.cash", "USDCAD", "USDCHF"],
    "EUR": ["EURUSD", "EURGBP", "EURJPY", "XAUUSD"],  # XAUUSD affected by EUR via inverse correlation
    "GBP": ["GBPUSD", "EURGBP", "GBPJPY"],
    "JPY": ["USDJPY", "EURJPY", "GBPJPY"],
    "AUD": ["AUDUSD", "AUDJPY"],
    "CAD": ["USDCAD", "CADJPY"],
    "CHF": ["USDCHF", "EURCHF"],
}


class NewsFilter:
    """
    Filter to avoid trading around high-impact news events.

    Configuration:
        - pre_event_minutes: Minutes before event to stop trading (default: 30)
        - post_event_minutes: Minutes after event to resume trading (default: 15)
        - cache_file: Path to cache file for events
    """

    def __init__(
        self,
        pre_event_minutes: int = 30,
        post_event_minutes: int = 15,
        cache_file: str = "data/hydra/ftmo/news_events.json",
    ):
        self.pre_event_minutes = pre_event_minutes
        self.post_event_minutes = post_event_minutes
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)

        self._events: List[NewsEvent] = []
        self._last_fetch: Optional[datetime] = None
        self._fetch_interval = timedelta(hours=1)  # Refresh every hour

        # Load cached events
        self._load_cache()

        logger.info(
            f"[NewsFilter] Initialized (pre={pre_event_minutes}m, post={post_event_minutes}m, "
            f"events={len(self._events)})"
        )

    def should_avoid(self, symbol: str) -> tuple[bool, Optional[str]]:
        """
        Check if trading should be avoided for a symbol due to upcoming news.

        Args:
            symbol: Trading symbol (e.g., "XAUUSD")

        Returns:
            Tuple of (should_avoid: bool, reason: Optional[str])
        """
        # Refresh events if needed
        self._maybe_refresh_events()

        now = datetime.now(timezone.utc)

        for event in self._events:
            # Check if this event affects the symbol
            if not self._affects_symbol(event, symbol):
                continue

            # Check if we're in the avoidance window
            window_start = event.event_time - timedelta(minutes=self.pre_event_minutes)
            window_end = event.event_time + timedelta(minutes=self.post_event_minutes)

            if window_start <= now <= window_end:
                # Determine position in window
                if now < event.event_time:
                    mins_until = (event.event_time - now).total_seconds() / 60
                    reason = f"News in {mins_until:.0f}m: {event.event_name} ({event.currency})"
                else:
                    mins_after = (now - event.event_time).total_seconds() / 60
                    reason = f"News {mins_after:.0f}m ago: {event.event_name} ({event.currency})"

                return True, reason

        return False, None

    def get_upcoming_events(self, symbol: str, hours: int = 24) -> List[NewsEvent]:
        """Get upcoming events that affect a symbol."""
        self._maybe_refresh_events()

        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours)

        upcoming = []
        for event in self._events:
            if event.event_time > now and event.event_time <= cutoff:
                if self._affects_symbol(event, symbol):
                    upcoming.append(event)

        return sorted(upcoming, key=lambda e: e.event_time)

    def add_event(self, event: NewsEvent):
        """Manually add an event (for testing or manual overrides)."""
        self._events.append(event)
        self._save_cache()

    def _affects_symbol(self, event: NewsEvent, symbol: str) -> bool:
        """Check if an event affects a trading symbol."""
        # Normalize symbol
        symbol_upper = symbol.upper()

        # Check direct currency match
        if event.currency in symbol_upper:
            return True

        # Check currency-to-symbols mapping
        affected = CURRENCY_SYMBOLS.get(event.currency, [])
        return symbol_upper in [s.upper() for s in affected]

    def _maybe_refresh_events(self):
        """Refresh events from storage if interval has passed."""
        now = datetime.now(timezone.utc)

        if self._last_fetch and (now - self._last_fetch) < self._fetch_interval:
            return  # Still fresh

        try:
            self._fetch_events_from_storage()
            self._last_fetch = now
        except Exception as e:
            logger.warning(f"[NewsFilter] Failed to refresh events: {e}")

    def _fetch_events_from_storage(self):
        """Fetch events from the knowledge base storage."""
        try:
            from libs.knowledge.storage import get_storage
            from libs.knowledge.base import ImpactLevel

            storage = get_storage()
            now = datetime.now(timezone.utc)

            # Get events for next 24 hours
            events = storage.get_events(
                start_date=now,
                end_date=now + timedelta(days=1),
                impact=ImpactLevel.HIGH,
            )

            # Convert to our format
            self._events = []
            for event in events:
                news_event = NewsEvent(
                    event_time=event.event_date,
                    event_name=event.event_name,
                    currency=event.currency,
                    impact=event.impact.value if hasattr(event.impact, 'value') else str(event.impact),
                    affected_symbols=CURRENCY_SYMBOLS.get(event.currency, []),
                )
                self._events.append(news_event)

            # Save to cache
            self._save_cache()
            logger.debug(f"[NewsFilter] Loaded {len(self._events)} events from storage")

        except Exception as e:
            logger.warning(f"[NewsFilter] Error fetching from storage: {e}")
            # Use static events as fallback
            self._add_known_recurring_events()

    def _add_known_recurring_events(self):
        """Add known recurring high-impact events as fallback."""
        now = datetime.now(timezone.utc)
        today = now.date()

        # NFP is always first Friday of the month at 13:30 UTC
        # Find first Friday of current month
        first_day = today.replace(day=1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday() + 7) % 7)

        if first_friday >= today:
            nfp_time = datetime.combine(first_friday, datetime.min.time().replace(hour=13, minute=30))
            nfp_time = nfp_time.replace(tzinfo=timezone.utc)

            if nfp_time > now:
                self._events.append(NewsEvent(
                    event_time=nfp_time,
                    event_name="Non-Farm Payrolls",
                    currency="USD",
                    impact="high",
                    affected_symbols=CURRENCY_SYMBOLS["USD"],
                ))

        logger.debug(f"[NewsFilter] Added {len(self._events)} recurring events as fallback")

    def _load_cache(self):
        """Load events from cache file."""
        if not self.cache_file.exists():
            return

        try:
            with open(self.cache_file) as f:
                data = json.load(f)

            self._events = []
            for item in data.get("events", []):
                event_time = datetime.fromisoformat(item["event_time"])
                # Skip past events
                if event_time > datetime.now(timezone.utc):
                    self._events.append(NewsEvent(
                        event_time=event_time,
                        event_name=item["event_name"],
                        currency=item["currency"],
                        impact=item["impact"],
                        affected_symbols=item.get("affected_symbols", []),
                    ))

            self._last_fetch = datetime.fromisoformat(data.get("last_fetch", "2000-01-01T00:00:00+00:00"))
            logger.debug(f"[NewsFilter] Loaded {len(self._events)} events from cache")

        except Exception as e:
            logger.warning(f"[NewsFilter] Error loading cache: {e}")

    def _save_cache(self):
        """Save events to cache file."""
        try:
            data = {
                "last_fetch": datetime.now(timezone.utc).isoformat(),
                "events": [
                    {
                        "event_time": event.event_time.isoformat(),
                        "event_name": event.event_name,
                        "currency": event.currency,
                        "impact": event.impact,
                        "affected_symbols": event.affected_symbols,
                    }
                    for event in self._events
                ],
            }

            with open(self.cache_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.warning(f"[NewsFilter] Error saving cache: {e}")


# Singleton instance
_instance: Optional[NewsFilter] = None


def get_news_filter() -> NewsFilter:
    """Get singleton NewsFilter instance."""
    global _instance
    if _instance is None:
        _instance = NewsFilter()
    return _instance


def should_avoid_news(symbol: str) -> tuple[bool, Optional[str]]:
    """
    Quick check if trading should be avoided due to news.

    Args:
        symbol: Trading symbol

    Returns:
        Tuple of (should_avoid, reason)
    """
    return get_news_filter().should_avoid(symbol)


if __name__ == "__main__":
    # Test the filter
    filter = NewsFilter()

    # Add a test event
    test_event = NewsEvent(
        event_time=datetime.now(timezone.utc) + timedelta(minutes=15),
        event_name="Test NFP",
        currency="USD",
        impact="high",
    )
    filter.add_event(test_event)

    # Test symbols
    for symbol in ["XAUUSD", "EURUSD", "GBPJPY"]:
        avoid, reason = filter.should_avoid(symbol)
        print(f"{symbol}: avoid={avoid}, reason={reason}")

    # Show upcoming events
    print("\nUpcoming events for XAUUSD:")
    for event in filter.get_upcoming_events("XAUUSD", hours=24):
        print(f"  {event.event_time}: {event.event_name} ({event.currency})")
