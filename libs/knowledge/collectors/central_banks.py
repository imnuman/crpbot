"""
Central Bank Calendar Collector.

Tracks major central bank meetings and rate decisions.
CRITICAL for avoiding volatile periods around:
- Fed (FOMC) - affects USD, Gold, Indices
- ECB - affects EUR pairs
- BOE - affects GBP pairs
- BOJ - affects JPY pairs
"""

import os
import re
import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from loguru import logger

try:
    import httpx
    from bs4 import BeautifulSoup
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False
    logger.warning("httpx/beautifulsoup4 not installed - Central bank collector disabled")

from ..base import (
    BaseCollector,
    KnowledgeItem,
    KnowledgeSource,
    ContentType,
)
from ..storage import get_storage


# Central Bank Calendar Sources (free)
CENTRAL_BANK_SOURCES = [
    "https://www.centralbanking.com/central-banks/monetary-policy/interest-rates",
    "https://www.fxstreet.com/economic-calendar/central-banks",
]

# Known 2024-2025 FOMC Meeting Dates
# Updated periodically - these are the critical ones
FOMC_DATES_2024 = [
    "2024-01-30", "2024-01-31",  # Jan meeting
    "2024-03-19", "2024-03-20",  # Mar meeting
    "2024-04-30", "2024-05-01",  # May meeting
    "2024-06-11", "2024-06-12",  # Jun meeting
    "2024-07-30", "2024-07-31",  # Jul meeting
    "2024-09-17", "2024-09-18",  # Sep meeting
    "2024-11-06", "2024-11-07",  # Nov meeting
    "2024-12-17", "2024-12-18",  # Dec meeting
]

FOMC_DATES_2025 = [
    "2025-01-28", "2025-01-29",  # Jan meeting
    "2025-03-18", "2025-03-19",  # Mar meeting
    "2025-05-06", "2025-05-07",  # May meeting
    "2025-06-17", "2025-06-18",  # Jun meeting
    "2025-07-29", "2025-07-30",  # Jul meeting
    "2025-09-16", "2025-09-17",  # Sep meeting
    "2025-11-04", "2025-11-05",  # Nov meeting
    "2025-12-16", "2025-12-17",  # Dec meeting
]

# ECB Meeting Dates 2024-2025 (usually Thursdays)
ECB_DATES_2024 = [
    "2024-01-25", "2024-03-07", "2024-04-11",
    "2024-06-06", "2024-07-18", "2024-09-12",
    "2024-10-17", "2024-12-12",
]

ECB_DATES_2025 = [
    "2025-01-30", "2025-03-06", "2025-04-17",
    "2025-06-05", "2025-07-24", "2025-09-11",
    "2025-10-30", "2025-12-18",
]

# BOE Meeting Dates
BOE_DATES_2024 = [
    "2024-02-01", "2024-03-21", "2024-05-09",
    "2024-06-20", "2024-08-01", "2024-09-19",
    "2024-11-07", "2024-12-19",
]

BOE_DATES_2025 = [
    "2025-02-06", "2025-03-20", "2025-05-08",
    "2025-06-19", "2025-08-07", "2025-09-18",
    "2025-11-06", "2025-12-18",
]

# BOJ Meeting Dates (Japan)
BOJ_DATES_2024 = [
    "2024-01-23", "2024-03-19", "2024-04-26",
    "2024-06-14", "2024-07-31", "2024-09-20",
    "2024-10-31", "2024-12-19",
]

BOJ_DATES_2025 = [
    "2025-01-24", "2025-03-14", "2025-05-01",
    "2025-06-13", "2025-07-31", "2025-09-19",
    "2025-10-31", "2025-12-19",
]


class CentralBankCollector(BaseCollector):
    """Collector for central bank meeting dates and rate decisions."""

    def __init__(self):
        self._client = None

    @property
    def client(self):
        """Lazy initialization of HTTP client."""
        if self._client is None and DEPS_AVAILABLE:
            self._client = httpx.AsyncClient(
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                },
                timeout=30.0,
                follow_redirects=True,
            )
        return self._client

    def get_source_name(self) -> KnowledgeSource:
        return KnowledgeSource.ECONOMIC

    def get_schedule(self) -> str:
        """Run daily at 00:00 UTC."""
        return "0 0 * * *"

    def get_max_items_per_run(self) -> int:
        return 20

    async def collect(self) -> List[KnowledgeItem]:
        """Collect upcoming central bank events."""
        items = []
        today = datetime.now(timezone.utc).date()

        # Generate items for next 30 days of meetings
        all_meetings = []

        # FOMC (Fed)
        for date_str in FOMC_DATES_2024 + FOMC_DATES_2025:
            meeting_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            if today <= meeting_date <= today + timedelta(days=30):
                all_meetings.append({
                    "bank": "Federal Reserve (FOMC)",
                    "date": meeting_date,
                    "currencies": ["USD"],
                    "affected_symbols": ["XAUUSD", "EURUSD", "GBPUSD", "US30", "NAS100"],
                    "volatility": "EXTREME",
                })

        # ECB
        for date_str in ECB_DATES_2024 + ECB_DATES_2025:
            meeting_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            if today <= meeting_date <= today + timedelta(days=30):
                all_meetings.append({
                    "bank": "European Central Bank (ECB)",
                    "date": meeting_date,
                    "currencies": ["EUR"],
                    "affected_symbols": ["EURUSD", "EURGBP", "EURJPY"],
                    "volatility": "HIGH",
                })

        # BOE
        for date_str in BOE_DATES_2024 + BOE_DATES_2025:
            meeting_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            if today <= meeting_date <= today + timedelta(days=30):
                all_meetings.append({
                    "bank": "Bank of England (BOE)",
                    "date": meeting_date,
                    "currencies": ["GBP"],
                    "affected_symbols": ["GBPUSD", "EURGBP", "GBPJPY"],
                    "volatility": "HIGH",
                })

        # BOJ
        for date_str in BOJ_DATES_2024 + BOJ_DATES_2025:
            meeting_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            if today <= meeting_date <= today + timedelta(days=30):
                all_meetings.append({
                    "bank": "Bank of Japan (BOJ)",
                    "date": meeting_date,
                    "currencies": ["JPY"],
                    "affected_symbols": ["USDJPY", "EURJPY", "GBPJPY"],
                    "volatility": "HIGH",
                })

        # Create knowledge items
        for meeting in all_meetings:
            item = self._create_meeting_item(meeting, today)
            if item:
                items.append(item)

        # Create summary item with all upcoming meetings
        if all_meetings:
            summary_item = self._create_summary_item(all_meetings, today)
            if summary_item:
                items.append(summary_item)

        logger.info(f"Collected {len(items)} central bank events")
        return items

    def _create_meeting_item(
        self,
        meeting: Dict[str, Any],
        today: datetime.date
    ) -> Optional[KnowledgeItem]:
        """Create KnowledgeItem for a single meeting."""
        try:
            bank = meeting["bank"]
            meeting_date = meeting["date"]
            days_until = (meeting_date - today).days
            volatility = meeting["volatility"]
            symbols = meeting["affected_symbols"]
            currencies = meeting["currencies"]

            # Determine action
            if days_until <= 1:
                action = "NO TRADING - Meeting in progress or imminent"
                risk_level = "CRITICAL"
            elif days_until <= 3:
                action = "REDUCE SIZE - Meeting approaching"
                risk_level = "HIGH"
            elif days_until <= 7:
                action = "CAUTION - Meeting this week"
                risk_level = "ELEVATED"
            else:
                action = "AWARE - Meeting next few weeks"
                risk_level = "NORMAL"

            summary = f"""{bank} Meeting: {meeting_date.strftime('%Y-%m-%d')}

Days Until: {days_until}
Affected: {', '.join(symbols)}
Risk Level: {risk_level}
Action: {action}
"""

            full_content = f"""Central Bank Meeting Alert

Bank: {bank}
Meeting Date: {meeting_date.strftime('%Y-%m-%d')}
Days Until: {days_until}
Volatility Impact: {volatility}

=== Affected Instruments ===

Currencies: {', '.join(currencies)}
Symbols: {', '.join(symbols)}

=== Trading Guidelines ===

Risk Level: {risk_level}
Recommended Action: {action}

1-3 Days Before:
- Reduce position sizes by 50%
- Tighten stop losses
- Close any marginal trades

Day Of Meeting:
- NO new positions on affected pairs
- Close existing positions before announcement
- Wait 30-60 mins after for dust to settle

Post-Meeting:
- Watch for directional bias
- Enter with normal sizing after volatility subsides

=== Historical Volatility ===

{bank} meetings typically cause:
- XAUUSD: 100-300 pip moves
- EURUSD/GBPUSD: 50-150 pip moves
- US30/NAS100: 1-3% swings

=== HYDRA Integration ===

# Add to trading bot before_trade check:
if days_until_central_bank <= 1:
    return None  # Skip trade
elif days_until_central_bank <= 3:
    position_size *= 0.5  # Half size

current_config = {{
    "bank": "{bank}",
    "date": "{meeting_date}",
    "days_until": {days_until},
    "risk_level": "{risk_level}",
    "affected_symbols": {symbols}
}}
"""

            item = KnowledgeItem(
                source=KnowledgeSource.ECONOMIC,
                source_url="https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
                title=f"{bank}: {meeting_date.strftime('%b %d')} ({days_until}d)",
                content_type=ContentType.NEWS,
                summary=summary,
                full_content=full_content,
                quality_score=0.95,
            )

            item.symbols = symbols
            item.tags = [
                "central_bank", bank.split()[0].lower(), volatility.lower(),
                risk_level.lower(), "monetary_policy"
            ]

            return item

        except Exception as e:
            logger.debug(f"Error creating meeting item: {e}")
            return None

    def _create_summary_item(
        self,
        meetings: List[Dict[str, Any]],
        today: datetime.date
    ) -> Optional[KnowledgeItem]:
        """Create summary item with all upcoming meetings."""
        try:
            # Sort by date
            sorted_meetings = sorted(meetings, key=lambda x: x["date"])

            summary_lines = ["Upcoming Central Bank Meetings (Next 30 Days):\n"]
            for m in sorted_meetings:
                days = (m["date"] - today).days
                summary_lines.append(
                    f"• {m['bank']}: {m['date'].strftime('%b %d')} ({days}d) - {m['volatility']}"
                )

            summary = "\n".join(summary_lines)

            full_content = f"""Central Bank Meeting Calendar

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

=== Upcoming Meetings ===

"""
            for m in sorted_meetings:
                days = (m["date"] - today).days
                full_content += f"""
{m['bank']}
- Date: {m['date'].strftime('%Y-%m-%d')}
- Days Until: {days}
- Volatility: {m['volatility']}
- Affected: {', '.join(m['affected_symbols'])}
"""

            full_content += """

=== Risk Management Protocol ===

CRITICAL (0-1 days): NO TRADING on affected pairs
HIGH (2-3 days): 50% position size, tight stops
ELEVATED (4-7 days): Normal size, wider awareness
NORMAL (8+ days): Standard trading

=== Quick Reference ===

FOMC (Fed) → USD, Gold, Indices
ECB → EUR pairs
BOE → GBP pairs
BOJ → JPY pairs (watch for surprise moves)
"""

            # Collect all affected symbols
            all_symbols = set()
            for m in sorted_meetings:
                all_symbols.update(m["affected_symbols"])

            item = KnowledgeItem(
                source=KnowledgeSource.ECONOMIC,
                source_url="https://www.centralbanking.com/",
                title=f"Central Bank Calendar: {len(sorted_meetings)} meetings upcoming",
                content_type=ContentType.NEWS,
                summary=summary,
                full_content=full_content,
                quality_score=0.95,
            )

            item.symbols = list(all_symbols)
            item.tags = ["central_bank", "calendar", "fomc", "ecb", "boe", "boj"]

            return item

        except Exception as e:
            logger.debug(f"Error creating summary item: {e}")
            return None

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()


async def run_central_bank_collector() -> int:
    """Run the central bank collector."""
    collector = CentralBankCollector()
    storage = get_storage()

    log_id = storage.start_scrape_log(KnowledgeSource.ECONOMIC, "central_banks")

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
        logger.info(f"Central bank collection: {saved}/{len(items)} saved")
        return saved

    except Exception as e:
        storage.complete_scrape_log(log_id, "failed", 0, 0, str(e))
        logger.error(f"Central bank collection failed: {e}")
        raise

    finally:
        await collector.close()


if __name__ == "__main__":
    asyncio.run(run_central_bank_collector())
