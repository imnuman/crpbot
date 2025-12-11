"""
Options Flow Collector - Unusual Options Activity.

Tracks unusual options activity from free sources.
Large options bets often precede big moves in the underlying.

Sources:
- Barchart unusual options (free tier)
- Yahoo Finance options chain (free)
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
    logger.warning("httpx/beautifulsoup4 not installed - Options flow disabled")

from ..base import (
    BaseCollector,
    KnowledgeItem,
    KnowledgeSource,
    ContentType,
)
from ..storage import get_storage


# Barchart unusual options page (free, no auth)
BARCHART_URL = "https://www.barchart.com/options/unusual-activity/stocks"

# Symbols we care about (map to our trading symbols)
SYMBOL_MAP = {
    "SPY": "US500",
    "QQQ": "NAS100",
    "DIA": "US30",
    "GLD": "XAUUSD",
    "SLV": "XAGUSD",
    "USO": "USOIL",
    "FXE": "EURUSD",
    "FXB": "GBPUSD",
    "IBIT": "BTCUSD",
    "ETHE": "ETHUSD",
}


class OptionsFlowCollector(BaseCollector):
    """Collector for unusual options activity."""

    def __init__(self):
        self._client = None

    @property
    def client(self):
        """Lazy initialization of HTTP client."""
        if self._client is None and DEPS_AVAILABLE:
            self._client = httpx.AsyncClient(
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "text/html,application/xhtml+xml",
                },
                timeout=30.0,
                follow_redirects=True,
            )
        return self._client

    def get_source_name(self) -> KnowledgeSource:
        return KnowledgeSource.SENTIMENT

    def get_schedule(self) -> str:
        """Run every 4 hours during market hours."""
        return "0 */4 * * *"

    def get_max_items_per_run(self) -> int:
        return 20

    async def collect(self) -> List[KnowledgeItem]:
        """Collect unusual options activity."""
        if not self.client:
            logger.error("HTTP client not available")
            return []

        items = []

        try:
            response = await self.client.get(BARCHART_URL)

            if response.status_code != 200:
                logger.warning(f"Barchart returned {response.status_code}")
                return items

            soup = BeautifulSoup(response.text, 'html.parser')

            # Find the options table
            tables = soup.find_all('table')

            for table in tables:
                rows = table.find_all('tr')
                for row in rows[1:]:  # Skip header
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 6:
                        try:
                            # Extract data from cells
                            symbol = cells[0].get_text(strip=True).upper()
                            strike = cells[1].get_text(strip=True) if len(cells) > 1 else ""
                            exp_date = cells[2].get_text(strip=True) if len(cells) > 2 else ""
                            opt_type = cells[3].get_text(strip=True).upper() if len(cells) > 3 else ""
                            volume = cells[4].get_text(strip=True) if len(cells) > 4 else "0"
                            oi = cells[5].get_text(strip=True) if len(cells) > 5 else "0"

                            # Check if this symbol maps to our trading symbols
                            trading_symbol = SYMBOL_MAP.get(symbol)
                            if not trading_symbol:
                                continue

                            # Parse volume and OI
                            volume_num = self._parse_number(volume)
                            oi_num = self._parse_number(oi)

                            # Calculate volume/OI ratio (unusual if > 2x)
                            if oi_num > 0:
                                vol_oi_ratio = volume_num / oi_num
                            else:
                                vol_oi_ratio = 0

                            # Only collect if volume is significant
                            if volume_num < 1000:
                                continue

                            item = self._create_options_item(
                                symbol=symbol,
                                trading_symbol=trading_symbol,
                                strike=strike,
                                exp_date=exp_date,
                                opt_type=opt_type,
                                volume=volume_num,
                                oi=oi_num,
                                vol_oi_ratio=vol_oi_ratio
                            )
                            if item:
                                items.append(item)

                        except Exception as e:
                            logger.debug(f"Error parsing options row: {e}")

            # Dedupe by symbol/type combo
            seen = set()
            unique_items = []
            for item in items:
                key = f"{item.title}"
                if key not in seen:
                    seen.add(key)
                    unique_items.append(item)
            items = unique_items[:self.get_max_items_per_run()]

        except Exception as e:
            logger.error(f"Options flow collection error: {e}")

        logger.info(f"Collected {len(items)} unusual options items")
        return items

    def _parse_number(self, text: str) -> float:
        """Parse number with K/M suffix."""
        text = text.replace(",", "").strip()
        if not text:
            return 0

        multiplier = 1
        if text.endswith("K"):
            multiplier = 1000
            text = text[:-1]
        elif text.endswith("M"):
            multiplier = 1000000
            text = text[:-1]

        try:
            return float(text) * multiplier
        except ValueError:
            return 0

    def _create_options_item(
        self,
        symbol: str,
        trading_symbol: str,
        strike: str,
        exp_date: str,
        opt_type: str,
        volume: float,
        oi: float,
        vol_oi_ratio: float
    ) -> Optional[KnowledgeItem]:
        """Create KnowledgeItem from options data."""
        try:
            # Determine signal
            is_call = "CALL" in opt_type.upper() or "C" in opt_type.upper()
            is_unusual = vol_oi_ratio > 2.0

            if is_call:
                signal = "BULLISH" if is_unusual else "Slightly bullish"
                bias = "buy"
            else:
                signal = "BEARISH" if is_unusual else "Slightly bearish"
                bias = "sell"

            if is_unusual:
                signal = f"UNUSUAL {signal}"

            summary = f"""Unusual Options: {symbol} ({trading_symbol})

Type: {opt_type} @ {strike} (exp: {exp_date})
Volume: {volume:,.0f} | Open Interest: {oi:,.0f}
Vol/OI Ratio: {vol_oi_ratio:.1f}x {'(UNUSUAL!)' if is_unusual else ''}

Signal: {signal}
Implication: Large {'call' if is_call else 'put'} buying suggests smart money expects {trading_symbol} to {'rise' if is_call else 'fall'}
"""

            full_content = f"""Unusual Options Activity Alert

Underlying: {symbol} → Maps to: {trading_symbol}
Option Type: {opt_type}
Strike: {strike}
Expiration: {exp_date}

=== Volume Analysis ===

Volume: {volume:,.0f}
Open Interest: {oi:,.0f}
Volume/OI Ratio: {vol_oi_ratio:.1f}x

Interpretation:
- Vol/OI > 2x indicates unusual activity (current: {vol_oi_ratio:.1f}x)
- High volume on calls = bullish bet
- High volume on puts = bearish bet

=== Trading Signal ===

Bias: {signal}
For {trading_symbol}: Consider {'BUY' if is_call else 'SELL'} bias

=== How to Use ===

1. Unusual call activity often precedes upward moves
2. Unusual put activity often precedes downward moves
3. Works best when combined with technical analysis
4. Check expiration date - closer expirations = more urgent signal

=== HYDRA Integration ===

if option_signal == "BULLISH":
    confidence_modifier *= 1.1 for BUY trades
elif option_signal == "BEARISH":
    confidence_modifier *= 1.1 for SELL trades
"""

            item = KnowledgeItem(
                source=KnowledgeSource.SENTIMENT,
                source_url=BARCHART_URL,
                title=f"Options Flow: {symbol} {opt_type} - {signal}",
                content_type=ContentType.MARKET_DATA,
                summary=summary,
                full_content=full_content,
                quality_score=0.85 if is_unusual else 0.7,
            )

            item.symbols = [trading_symbol]
            item.tags = [
                "options", "flow", bias,
                "unusual" if is_unusual else "normal",
                "call" if is_call else "put"
            ]

            return item

        except Exception as e:
            logger.debug(f"Error creating options item: {e}")
            return None

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()


async def run_options_flow_collector() -> int:
    """Run the options flow collector."""
    collector = OptionsFlowCollector()
    storage = get_storage()

    log_id = storage.start_scrape_log(KnowledgeSource.SENTIMENT, "options_flow")

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
        logger.info(f"Options flow collection: {saved}/{len(items)} saved")
        return saved

    except Exception as e:
        storage.complete_scrape_log(log_id, "failed", 0, 0, str(e))
        logger.error(f"Options flow collection failed: {e}")
        raise

    finally:
        await collector.close()


if __name__ == "__main__":
    asyncio.run(run_options_flow_collector())
