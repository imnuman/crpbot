"""
Fear & Greed / VIX data collector.

Collects market sentiment indicators:
- CNN Fear & Greed Index (stocks)
- Crypto Fear & Greed Index
- VIX levels and interpretation

These help gauge when to be aggressive vs defensive.
"""

import os
import asyncio
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from loguru import logger

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not installed - Fear/Greed collector disabled")

from ..base import (
    BaseCollector,
    KnowledgeItem,
    KnowledgeSource,
    ContentType,
    auto_tag_content,
)
from ..storage import get_storage


# API endpoints (free, no auth required)
CRYPTO_FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=30"
CNN_FEAR_GREED_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"

# VIX interpretation levels
VIX_LEVELS = {
    "extreme_fear": {"min": 30, "max": 100, "signal": "BUY opportunity - market panic"},
    "fear": {"min": 20, "max": 30, "signal": "Consider buying - elevated fear"},
    "neutral": {"min": 15, "max": 20, "signal": "Normal conditions"},
    "complacency": {"min": 12, "max": 15, "signal": "Be cautious - low volatility"},
    "extreme_complacency": {"min": 0, "max": 12, "signal": "DANGER - correction likely"},
}


class FearGreedCollector(BaseCollector):
    """Collector for Fear & Greed indices."""

    def __init__(self):
        self._client = None

    @property
    def client(self):
        """Lazy initialization of HTTP client."""
        if self._client is None and HTTPX_AVAILABLE:
            self._client = httpx.AsyncClient(
                headers={"User-Agent": "HYDRA-Knowledge/1.0"},
                timeout=30.0,
                follow_redirects=True,
            )
        return self._client

    def get_source_name(self) -> KnowledgeSource:
        return KnowledgeSource.SENTIMENT

    def get_schedule(self) -> str:
        """Run every 6 hours."""
        return "0 */6 * * *"

    def get_max_items_per_run(self) -> int:
        return 5

    async def collect(self) -> List[KnowledgeItem]:
        """Collect fear/greed data."""
        if not self.client:
            logger.error("HTTP client not available")
            return []

        items = []

        # Crypto Fear & Greed
        try:
            crypto_item = await self._collect_crypto_fear_greed()
            if crypto_item:
                items.append(crypto_item)
        except Exception as e:
            logger.error(f"Error collecting crypto fear/greed: {e}")

        # CNN Fear & Greed (stocks)
        try:
            cnn_item = await self._collect_cnn_fear_greed()
            if cnn_item:
                items.append(cnn_item)
        except Exception as e:
            logger.debug(f"CNN Fear/Greed not available: {e}")

        logger.info(f"Collected {len(items)} fear/greed items")
        return items

    async def _collect_crypto_fear_greed(self) -> Optional[KnowledgeItem]:
        """Collect Crypto Fear & Greed Index."""
        try:
            response = await self.client.get(CRYPTO_FEAR_GREED_URL)

            if response.status_code != 200:
                return None

            data = response.json()
            fng_data = data.get("data", [])

            if not fng_data:
                return None

            current = fng_data[0]
            value = int(current.get("value", 50))
            classification = current.get("value_classification", "Neutral")

            # Calculate trend (7-day average vs current)
            if len(fng_data) >= 7:
                avg_7d = sum(int(d.get("value", 50)) for d in fng_data[:7]) / 7
                trend = "rising" if value > avg_7d else "falling"
                trend_diff = value - avg_7d
            else:
                trend = "neutral"
                trend_diff = 0

            # Determine trading signal
            if value <= 25:
                signal = "STRONG BUY - Extreme Fear (contrarian)"
                action = "aggressive_buy"
            elif value <= 40:
                signal = "BUY - Fear present, good entry zone"
                action = "buy"
            elif value <= 60:
                signal = "NEUTRAL - Market balanced"
                action = "hold"
            elif value <= 75:
                signal = "CAUTION - Greed building, tighten stops"
                action = "reduce"
            else:
                signal = "SELL/REDUCE - Extreme Greed (correction likely)"
                action = "sell"

            summary = f"""Crypto Fear & Greed Index: {value} ({classification})

7-Day Trend: {trend} ({trend_diff:+.1f} points)
Signal: {signal}

Extreme Fear (<25): Historically best buying opportunity
Greed (>75): Market euphoria, correction often follows
"""

            full_content = f"""Crypto Fear & Greed Index Analysis

Current Reading: {value}/100 ({classification})
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

=== Index Components ===
- Volatility (25%): Market volatility vs 30/90 day average
- Market Momentum/Volume (25%): Current volume vs averages
- Social Media (15%): Crypto mentions and engagement
- Surveys (15%): Weekly crypto polls
- Bitcoin Dominance (10%): BTC market cap share
- Google Trends (10%): Search volume for crypto terms

=== Historical Data (Last 7 Days) ===
"""
            for i, d in enumerate(fng_data[:7]):
                full_content += f"Day -{i}: {d.get('value')} ({d.get('value_classification')})\n"

            full_content += f"""

=== Trading Implications ===

Current Signal: {signal}
Recommended Action: {action.upper()}

For BTC/ETH trades:
- Extreme Fear (0-25): Increase position size, DCA aggressively
- Fear (26-40): Normal entries, slightly larger size
- Neutral (41-60): Standard position sizing
- Greed (61-75): Smaller positions, tighter stops
- Extreme Greed (76-100): Take profits, avoid new longs

=== HYDRA Integration ===

confidence_modifier = {{
    'extreme_fear': 1.3,  # Increase confidence for longs
    'fear': 1.1,
    'neutral': 1.0,
    'greed': 0.9,
    'extreme_greed': 0.7  # Reduce confidence for longs
}}

Current modifier for LONG trades: {1.3 if value <= 25 else 1.1 if value <= 40 else 1.0 if value <= 60 else 0.9 if value <= 75 else 0.7}
Current modifier for SHORT trades: {0.7 if value <= 25 else 0.9 if value <= 40 else 1.0 if value <= 60 else 1.1 if value <= 75 else 1.3}
"""

            item = KnowledgeItem(
                source=KnowledgeSource.SENTIMENT,
                source_url="https://alternative.me/crypto/fear-and-greed-index/",
                title=f"Crypto Fear & Greed: {value} ({classification})",
                content_type=ContentType.MARKET_DATA,
                summary=summary,
                full_content=full_content,
                quality_score=0.85,
            )

            item.symbols = ["BTCUSD", "ETHUSD"]
            item.tags = [
                "fear_greed", "sentiment", "crypto",
                classification.lower().replace(" ", "_"),
                action
            ]

            return item

        except Exception as e:
            logger.error(f"Error parsing crypto fear/greed: {e}")
            return None

    async def _collect_cnn_fear_greed(self) -> Optional[KnowledgeItem]:
        """Collect CNN Fear & Greed Index (stocks)."""
        try:
            # CNN's API may require different headers
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
            }

            response = await self.client.get(CNN_FEAR_GREED_URL, headers=headers)

            if response.status_code != 200:
                logger.debug(f"CNN API returned {response.status_code}")
                return None

            data = response.json()
            score = data.get("fear_and_greed", {}).get("score", 50)
            rating = data.get("fear_and_greed", {}).get("rating", "Neutral")

            # Determine signal
            if score <= 25:
                signal = "STRONG BUY stocks/indices - Extreme Fear"
            elif score <= 40:
                signal = "BUY - Fear zone"
            elif score <= 60:
                signal = "NEUTRAL"
            elif score <= 75:
                signal = "CAUTION - Greed"
            else:
                signal = "SELL/REDUCE - Extreme Greed"

            summary = f"""CNN Fear & Greed Index (Stocks): {score:.0f} ({rating})

Signal: {signal}

This affects US30, NAS100, US500 trades.
"""

            item = KnowledgeItem(
                source=KnowledgeSource.SENTIMENT,
                source_url="https://edition.cnn.com/markets/fear-and-greed",
                title=f"CNN Fear & Greed: {score:.0f} ({rating})",
                content_type=ContentType.MARKET_DATA,
                summary=summary,
                full_content=f"{summary}\n\nRaw data: {data}",
                quality_score=0.8,
            )

            item.symbols = ["US30", "NAS100", "US500"]
            item.tags = ["fear_greed", "sentiment", "stocks", rating.lower()]

            return item

        except Exception as e:
            logger.debug(f"CNN Fear/Greed error: {e}")
            return None

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()


async def run_fear_greed_collector() -> int:
    """Run the Fear/Greed collector and save results."""
    collector = FearGreedCollector()
    storage = get_storage()

    log_id = storage.start_scrape_log(KnowledgeSource.SENTIMENT, "fear_greed_collect")

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
        logger.info(f"Fear/Greed collection complete: {saved}/{len(items)} saved")
        return saved

    except Exception as e:
        storage.complete_scrape_log(log_id, "failed", 0, 0, str(e))
        logger.error(f"Fear/Greed collection failed: {e}")
        raise

    finally:
        await collector.close()


if __name__ == "__main__":
    asyncio.run(run_fear_greed_collector())
