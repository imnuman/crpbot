"""
Economic calendar collector.

Sources:
- ForexFactory XML calendar
- Investing.com (scrape)
- EODHD API (if available)
"""

import asyncio
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from loguru import logger

try:
    import httpx
    from bs4 import BeautifulSoup
    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False
    logger.warning("httpx/beautifulsoup4 not installed - calendar collector disabled")

from ..base import (
    BaseCollector,
    KnowledgeItem,
    EconomicEvent,
    KnowledgeSource,
    ContentType,
    ImpactLevel,
)
from ..storage import get_storage


# Calendar sources
FOREX_FACTORY_CALENDAR = "https://www.forexfactory.com/calendar.xml"
INVESTING_CALENDAR = "https://www.investing.com/economic-calendar/"

# High impact events we care about
HIGH_IMPACT_EVENTS = [
    "Non-Farm Payrolls",
    "NFP",
    "FOMC",
    "Fed Interest Rate Decision",
    "CPI",
    "Consumer Price Index",
    "GDP",
    "Retail Sales",
    "Unemployment Rate",
    "PMI",
    "ECB Interest Rate",
    "BOE Interest Rate",
]

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


class EconomicCalendarCollector(BaseCollector):
    """Collector for economic calendar events."""

    def __init__(self):
        self._client = None

    @property
    def client(self):
        """Lazy initialization of HTTP client."""
        if self._client is None and SCRAPING_AVAILABLE:
            self._client = httpx.AsyncClient(
                headers={"User-Agent": USER_AGENT},
                timeout=30.0,
                follow_redirects=True,
            )
        return self._client

    def get_source_name(self) -> KnowledgeSource:
        return KnowledgeSource.FOREX_FACTORY

    def get_schedule(self) -> str:
        """Run daily at 00:00 UTC."""
        return "0 0 * * *"

    def get_max_items_per_run(self) -> int:
        return 500  # Calendar can have many events

    async def collect(self) -> List[KnowledgeItem]:
        """Collect economic calendar events."""
        if not self.client:
            logger.error("HTTP client not available")
            return []

        events = []

        # Try ForexFactory first
        try:
            ff_events = await self._collect_forex_factory()
            events.extend(ff_events)
            logger.info(f"Collected {len(ff_events)} events from ForexFactory")
        except Exception as e:
            logger.error(f"ForexFactory collection failed: {e}")

        # Convert events to knowledge items
        items = []
        for event in events:
            item = self._event_to_knowledge_item(event)
            if item:
                items.append(item)

        # Also save events to dedicated table
        storage = get_storage()
        for event in events:
            try:
                storage.save_event(event)
            except Exception as e:
                logger.debug(f"Error saving event: {e}")

        logger.info(f"Collected {len(items)} calendar items")
        return items

    async def _collect_forex_factory(self) -> List[EconomicEvent]:
        """Collect from ForexFactory XML calendar."""
        events = []

        try:
            response = await self.client.get(FOREX_FACTORY_CALENDAR)
            response.raise_for_status()

            # Parse XML
            root = ET.fromstring(response.text)

            for event_elem in root.findall(".//event"):
                try:
                    event = self._parse_ff_event(event_elem)
                    if event:
                        events.append(event)
                except Exception as e:
                    logger.debug(f"Error parsing FF event: {e}")

        except Exception as e:
            logger.error(f"Error fetching ForexFactory: {e}")

        return events

    def _parse_ff_event(self, elem: ET.Element) -> Optional[EconomicEvent]:
        """Parse ForexFactory XML event element."""
        try:
            # Get event details
            title = elem.findtext("title", "")
            currency = elem.findtext("country", "")
            date_str = elem.findtext("date", "")
            time_str = elem.findtext("time", "")
            impact_str = elem.findtext("impact", "")
            forecast = elem.findtext("forecast", "")
            previous = elem.findtext("previous", "")

            if not title or not date_str:
                return None

            # Parse date/time
            try:
                if time_str and time_str != "All Day":
                    dt_str = f"{date_str} {time_str}"
                    event_date = datetime.strptime(dt_str, "%m-%d-%Y %I:%M%p")
                else:
                    event_date = datetime.strptime(date_str, "%m-%d-%Y")
                event_date = event_date.replace(tzinfo=timezone.utc)
            except ValueError:
                event_date = datetime.now(timezone.utc)

            # Parse impact
            impact_map = {
                "High": ImpactLevel.HIGH,
                "Medium": ImpactLevel.MEDIUM,
                "Low": ImpactLevel.LOW,
            }
            impact = impact_map.get(impact_str, ImpactLevel.LOW)

            return EconomicEvent(
                event_date=event_date,
                currency=currency,
                event_name=title,
                impact=impact,
                forecast=forecast or None,
                previous=previous or None,
                source=KnowledgeSource.FOREX_FACTORY,
            )

        except Exception as e:
            logger.debug(f"Error parsing FF event: {e}")
            return None

    def _event_to_knowledge_item(self, event: EconomicEvent) -> Optional[KnowledgeItem]:
        """Convert economic event to knowledge item."""
        try:
            # Only create items for high/medium impact events
            if event.impact == ImpactLevel.LOW:
                return None

            # Create content
            content = f"""
Economic Event: {event.event_name}
Date: {event.event_date.strftime('%Y-%m-%d %H:%M')} UTC
Currency: {event.currency}
Impact: {event.impact.value.upper()}
Forecast: {event.forecast or 'N/A'}
Previous: {event.previous or 'N/A'}

Trading Implications:
- {event.currency} pairs will likely see increased volatility
- Consider reducing position sizes before this event
- Wait for news reaction before entering new trades
"""

            # Determine affected symbols
            currency_symbols = {
                "USD": ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "US30", "NAS100"],
                "EUR": ["EURUSD", "EURGBP", "EURJPY"],
                "GBP": ["GBPUSD", "EURGBP", "GBPJPY"],
                "JPY": ["USDJPY", "EURJPY", "GBPJPY"],
                "AUD": ["AUDUSD", "AUDJPY"],
                "CAD": ["USDCAD", "CADJPY"],
                "CHF": ["USDCHF", "EURCHF"],
            }

            symbols = currency_symbols.get(event.currency, [])

            item = KnowledgeItem(
                source=KnowledgeSource.FOREX_FACTORY,
                source_url=FOREX_FACTORY_CALENDAR,
                title=f"{event.currency} - {event.event_name}",
                content_type=ContentType.CALENDAR_EVENT,
                summary=f"{event.impact.value.upper()} impact event: {event.event_name} on {event.event_date.strftime('%Y-%m-%d')}",
                full_content=content,
                symbols=symbols,
                tags=["economic", "calendar", event.currency.lower(), event.impact.value],
                quality_score=0.9 if event.impact == ImpactLevel.HIGH else 0.7,
                source_created_at=event.event_date,
            )

            return item

        except Exception as e:
            logger.debug(f"Error converting event: {e}")
            return None

    async def get_upcoming_events(
        self,
        days: int = 7,
        currencies: Optional[List[str]] = None,
        min_impact: ImpactLevel = ImpactLevel.MEDIUM,
    ) -> List[EconomicEvent]:
        """Get upcoming economic events."""
        storage = get_storage()

        start_date = datetime.now(timezone.utc)
        end_date = start_date + timedelta(days=days)

        events = storage.get_events(
            start_date=start_date,
            end_date=end_date,
            impact=min_impact,
        )

        if currencies:
            events = [e for e in events if e.currency in currencies]

        return events

    async def get_events_affecting_symbol(
        self,
        symbol: str,
        days: int = 7,
    ) -> List[EconomicEvent]:
        """Get events that could affect a specific symbol."""
        all_events = await self.get_upcoming_events(days=days, min_impact=ImpactLevel.MEDIUM)

        affecting_events = []
        for event in all_events:
            if event.affects_symbol(symbol):
                affecting_events.append(event)

        return affecting_events

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()


async def run_calendar_collector() -> int:
    """Run the calendar collector and save results."""
    collector = EconomicCalendarCollector()
    storage = get_storage()

    log_id = storage.start_scrape_log(KnowledgeSource.FOREX_FACTORY, "calendar_collect")

    try:
        items = await collector.collect()

        saved = 0
        for item in items:
            try:
                storage.save_item(item)
                saved += 1
            except Exception as e:
                logger.error(f"Error saving item: {e}")

        storage.complete_scrape_log(log_id, "success", len(items), saved)
        logger.info(f"Calendar collection complete: {saved}/{len(items)} saved")
        return saved

    except Exception as e:
        storage.complete_scrape_log(log_id, "failed", 0, 0, str(e))
        logger.error(f"Calendar collection failed: {e}")
        raise

    finally:
        await collector.close()


if __name__ == "__main__":
    asyncio.run(run_calendar_collector())
