"""
Broker Sentiment Data Collector.

Collects retail trader positioning from multiple brokers.
This is GOLD for contrarian trading - retail traders are wrong 70%+ of the time.

Sources:
- Myfxbook Community Outlook (free)
- DailyFX Sentiment (free)
- IG Client Sentiment (free)
"""

import os
import re
import asyncio
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from loguru import logger

try:
    import httpx
    from bs4 import BeautifulSoup
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False
    logger.warning("httpx/beautifulsoup4 not installed - Broker sentiment disabled")

from ..base import (
    BaseCollector,
    KnowledgeItem,
    KnowledgeSource,
    ContentType,
)
from ..storage import get_storage


# Myfxbook Community Outlook (most comprehensive)
MYFXBOOK_URL = "https://www.myfxbook.com/community/outlook"

# DailyFX Client Sentiment
DAILYFX_URL = "https://www.dailyfx.com/sentiment"

# Our target symbols
TARGET_SYMBOLS = {
    "EURUSD": ["EUR/USD", "EURUSD", "eurusd"],
    "GBPUSD": ["GBP/USD", "GBPUSD", "gbpusd"],
    "XAUUSD": ["XAU/USD", "XAUUSD", "Gold", "GOLD"],
    "USDJPY": ["USD/JPY", "USDJPY", "usdjpy"],
    "AUDUSD": ["AUD/USD", "AUDUSD", "audusd"],
    "US30": ["US30", "Dow Jones", "DOW", "DJI"],
    "NAS100": ["NAS100", "NASDAQ", "US100", "NDX"],
    "BTCUSD": ["BTC/USD", "BTCUSD", "Bitcoin"],
}


class BrokerSentimentCollector(BaseCollector):
    """Collector for retail broker sentiment data."""

    def __init__(self):
        self._client = None

    @property
    def client(self):
        """Lazy initialization of HTTP client."""
        if self._client is None and DEPS_AVAILABLE:
            self._client = httpx.AsyncClient(
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                },
                timeout=30.0,
                follow_redirects=True,
            )
        return self._client

    def get_source_name(self) -> KnowledgeSource:
        return KnowledgeSource.SENTIMENT

    def get_schedule(self) -> str:
        """Run every 4 hours."""
        return "0 */4 * * *"

    def get_max_items_per_run(self) -> int:
        return 15

    async def collect(self) -> List[KnowledgeItem]:
        """Collect broker sentiment data."""
        if not self.client:
            logger.error("HTTP client not available")
            return []

        items = []

        # Myfxbook (primary source)
        try:
            myfxbook_items = await self._collect_myfxbook()
            items.extend(myfxbook_items)
        except Exception as e:
            logger.error(f"Myfxbook error: {e}")

        # DailyFX (backup)
        try:
            dailyfx_items = await self._collect_dailyfx()
            items.extend(dailyfx_items)
        except Exception as e:
            logger.debug(f"DailyFX error: {e}")

        logger.info(f"Collected {len(items)} broker sentiment items")
        return items

    async def _collect_myfxbook(self) -> List[KnowledgeItem]:
        """Scrape Myfxbook community outlook."""
        items = []

        try:
            response = await self.client.get(MYFXBOOK_URL)

            if response.status_code != 200:
                logger.warning(f"Myfxbook returned {response.status_code}")
                return items

            soup = BeautifulSoup(response.text, 'html.parser')

            # Find sentiment data in tables or divs
            # Myfxbook shows % long vs % short for each pair
            sentiment_data = {}

            # Look for table rows with symbol data
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 3:
                        text = row.get_text()

                        # Check if this row contains our symbols
                        for symbol, aliases in TARGET_SYMBOLS.items():
                            for alias in aliases:
                                if alias.lower() in text.lower():
                                    # Try to extract percentages
                                    numbers = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
                                    if len(numbers) >= 2:
                                        # Usually: long%, short%
                                        long_pct = float(numbers[0])
                                        short_pct = float(numbers[1])

                                        sentiment_data[symbol] = {
                                            "long_pct": long_pct,
                                            "short_pct": short_pct,
                                            "source": "myfxbook"
                                        }
                                    break

            # Also try to find divs with sentiment info
            divs = soup.find_all('div', class_=re.compile(r'sentiment|outlook|position', re.I))
            for div in divs:
                text = div.get_text()
                for symbol, aliases in TARGET_SYMBOLS.items():
                    if symbol in sentiment_data:
                        continue
                    for alias in aliases:
                        if alias.lower() in text.lower():
                            numbers = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
                            if len(numbers) >= 2:
                                sentiment_data[symbol] = {
                                    "long_pct": float(numbers[0]),
                                    "short_pct": float(numbers[1]),
                                    "source": "myfxbook"
                                }
                            break

            # Create items from collected data
            for symbol, data in sentiment_data.items():
                item = self._create_sentiment_item(symbol, data)
                if item:
                    items.append(item)

        except Exception as e:
            logger.error(f"Myfxbook scraping error: {e}")

        return items

    async def _collect_dailyfx(self) -> List[KnowledgeItem]:
        """Scrape DailyFX sentiment page."""
        items = []

        try:
            response = await self.client.get(DAILYFX_URL)

            if response.status_code != 200:
                return items

            soup = BeautifulSoup(response.text, 'html.parser')

            # DailyFX has specific sentiment cards
            sentiment_cards = soup.find_all('div', class_=re.compile(r'sentiment|client', re.I))

            for card in sentiment_cards:
                text = card.get_text()

                for symbol, aliases in TARGET_SYMBOLS.items():
                    for alias in aliases:
                        if alias.lower() in text.lower():
                            # Extract long/short percentages
                            numbers = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
                            if len(numbers) >= 2:
                                item = self._create_sentiment_item(symbol, {
                                    "long_pct": float(numbers[0]),
                                    "short_pct": float(numbers[1]),
                                    "source": "dailyfx"
                                })
                                if item:
                                    items.append(item)
                            break

        except Exception as e:
            logger.debug(f"DailyFX error: {e}")

        return items

    def _create_sentiment_item(
        self,
        symbol: str,
        data: Dict[str, Any]
    ) -> Optional[KnowledgeItem]:
        """Create KnowledgeItem from sentiment data."""
        try:
            long_pct = data.get("long_pct", 50)
            short_pct = data.get("short_pct", 50)
            source = data.get("source", "unknown")

            # Normalize to 100%
            total = long_pct + short_pct
            if total > 0:
                long_pct = (long_pct / total) * 100
                short_pct = (short_pct / total) * 100

            # Determine contrarian signal
            # Key insight: When retail is extremely long (>70%), go SHORT
            # When retail is extremely short (>70%), go LONG
            if long_pct >= 70:
                signal = "STRONG SELL (contrarian)"
                bias = "bearish"
                confidence_mod = 1.2
                reasoning = f"Retail is {long_pct:.0f}% long - fade the crowd"
            elif long_pct >= 60:
                signal = "SELL bias (contrarian)"
                bias = "bearish"
                confidence_mod = 1.1
                reasoning = f"Retail majority long ({long_pct:.0f}%) - slight bearish edge"
            elif short_pct >= 70:
                signal = "STRONG BUY (contrarian)"
                bias = "bullish"
                confidence_mod = 1.2
                reasoning = f"Retail is {short_pct:.0f}% short - fade the crowd"
            elif short_pct >= 60:
                signal = "BUY bias (contrarian)"
                bias = "bullish"
                confidence_mod = 1.1
                reasoning = f"Retail majority short ({short_pct:.0f}%) - slight bullish edge"
            else:
                signal = "NEUTRAL"
                bias = "neutral"
                confidence_mod = 1.0
                reasoning = "No extreme positioning"

            summary = f"""Broker Sentiment for {symbol}:

Retail Positioning: {long_pct:.0f}% Long / {short_pct:.0f}% Short
Contrarian Signal: {signal}
{reasoning}

Source: {source}
"""

            full_content = f"""Retail Broker Sentiment Analysis

Symbol: {symbol}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
Source: {source}

=== Positioning Data ===

Retail Traders:
- Long: {long_pct:.1f}%
- Short: {short_pct:.1f}%

=== Contrarian Analysis ===

Key Principle: Retail traders lose ~70% of the time.
When retail is extremely positioned one way, the market often moves against them.

Current Signal: {signal}
Bias: {bias.upper()}
Confidence Modifier: {confidence_mod}

=== Historical Context ===

Extreme readings (>70% one way) historically precede reversals:
- >70% long → price often drops
- >70% short → price often rises

This is because:
1. Smart money sees retail order flow
2. Brokers profit when retail loses
3. Liquidity pools form where retail places stops

=== HYDRA Integration ===

For {symbol} trades:

if direction == "BUY":
    confidence *= {confidence_mod if bias == 'bullish' else 1/confidence_mod:.2f}
elif direction == "SELL":
    confidence *= {confidence_mod if bias == 'bearish' else 1/confidence_mod:.2f}

# Current recommendation
bias = "{bias}"
signal = "{signal}"
"""

            item = KnowledgeItem(
                source=KnowledgeSource.SENTIMENT,
                source_url=f"https://www.myfxbook.com/community/outlook",
                title=f"Retail Sentiment: {symbol} - {long_pct:.0f}% Long ({signal})",
                content_type=ContentType.MARKET_DATA,
                summary=summary,
                full_content=full_content,
                quality_score=0.8,
            )

            item.symbols = [symbol]
            item.tags = [
                "sentiment", "retail", "contrarian",
                bias, source,
                "extreme" if max(long_pct, short_pct) >= 70 else "normal"
            ]

            return item

        except Exception as e:
            logger.debug(f"Error creating sentiment item: {e}")
            return None

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()


async def run_broker_sentiment_collector() -> int:
    """Run the broker sentiment collector."""
    collector = BrokerSentimentCollector()
    storage = get_storage()

    log_id = storage.start_scrape_log(KnowledgeSource.SENTIMENT, "broker_sentiment")

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
        logger.info(f"Broker sentiment collection: {saved}/{len(items)} saved")
        return saved

    except Exception as e:
        storage.complete_scrape_log(log_id, "failed", 0, 0, str(e))
        logger.error(f"Broker sentiment collection failed: {e}")
        raise

    finally:
        await collector.close()


if __name__ == "__main__":
    asyncio.run(run_broker_sentiment_collector())
