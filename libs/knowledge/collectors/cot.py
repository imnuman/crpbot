"""
CFTC Commitment of Traders (COT) data collector.

Weekly institutional positioning data - shows how large speculators,
commercials, and small traders are positioned in futures markets.

Source: CFTC.gov (free, public data)
"""

import os
import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from loguru import logger

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not installed - COT collector disabled")

from ..base import (
    BaseCollector,
    KnowledgeItem,
    KnowledgeSource,
    ContentType,
    auto_tag_content,
)
from ..storage import get_storage


# CFTC COT Data URLs - New API (as of 2025)
# TFF (Traders in Financial Futures) for forex, metals, indices
COT_TFF_API = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"
# Legacy report for commodities
COT_LEGACY_API = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
# Disaggregated for more detail
COT_DISAGG_API = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"

# Relevant contracts - mapped from API commodity_name to our symbols
CONTRACT_MAP = {
    # Forex (TFF report)
    "EURO FX": "EURUSD",
    "BRITISH POUND": "GBPUSD",
    "JAPANESE YEN": "USDJPY",
    "SWISS FRANC": "USDCHF",
    "CANADIAN DOLLAR": "USDCAD",
    "AUSTRALIAN DOLLAR": "AUDUSD",
    "NEW ZEALAND DOLLAR": "NZDUSD",

    # Metals (from disaggregated)
    "GOLD": "XAUUSD",
    "SILVER": "XAGUSD",

    # Indices (TFF report)
    "E-MINI S&P 500": "US500",
    "NASDAQ-100": "NAS100",
    "E-MINI NASDAQ": "NAS100",
    "DJIA": "US30",
    "DOW JONES": "US30",
    "UST BOND": "T-BOND",
    "UST 10Y NOTE": "T-NOTE",

    # Crypto
    "BITCOIN": "BTCUSD",
    "MICRO BITCOIN": "BTCUSD",
}

# Keywords to search for in contract names
CONTRACT_KEYWORDS = [
    "EURO FX", "BRITISH POUND", "JAPANESE YEN", "SWISS FRANC",
    "CANADIAN DOLLAR", "AUSTRALIAN DOLLAR", "GOLD", "SILVER",
    "S&P 500", "NASDAQ", "DJIA", "DOW", "BITCOIN"
]


class COTCollector(BaseCollector):
    """Collector for CFTC Commitment of Traders data."""

    def __init__(self):
        self._client = None

    @property
    def client(self):
        """Lazy initialization of HTTP client."""
        if self._client is None and HTTPX_AVAILABLE:
            self._client = httpx.AsyncClient(
                headers={"User-Agent": "HYDRA-Knowledge/1.0"},
                timeout=60.0,
                follow_redirects=True,
            )
        return self._client

    def get_source_name(self) -> KnowledgeSource:
        return KnowledgeSource.CFTC

    def get_schedule(self) -> str:
        """Run every Saturday at 01:00 UTC (after Friday release)."""
        return "0 1 * * 6"

    def get_max_items_per_run(self) -> int:
        return 20

    async def collect(self) -> List[KnowledgeItem]:
        """Collect COT data from CFTC Public Reporting API."""
        if not self.client:
            logger.error("HTTP client not available")
            return []

        items = []
        seen_symbols = set()

        try:
            # Fetch TFF (Traders in Financial Futures) data - forex, indices
            # Get latest 50 records, ordered by date descending
            params = {
                "$limit": 100,
                "$order": "report_date_as_yyyy_mm_dd DESC"
            }
            response = await self.client.get(COT_TFF_API, params=params)

            if response.status_code != 200:
                logger.error(f"COT TFF fetch failed: {response.status_code}")
                return items

            data = response.json()
            logger.info(f"Fetched {len(data)} TFF records from CFTC API")

            # Parse JSON response
            for record in data:
                contract_name = record.get("market_and_exchange_names", "")
                commodity_name = record.get("commodity_name", "").upper()

                # Check if this contract matches our interests
                symbol = self._match_contract(contract_name, commodity_name)
                if not symbol or symbol in seen_symbols:
                    continue

                seen_symbols.add(symbol)

                # Extract positioning data
                cot_data = {
                    "report_date": record.get("report_date_as_yyyy_mm_dd", ""),
                    "open_interest": self._safe_int(record.get("open_interest_all", 0)),
                    # TFF uses different categories
                    "dealer_long": self._safe_int(record.get("dealer_positions_long_all", 0)),
                    "dealer_short": self._safe_int(record.get("dealer_positions_short_all", 0)),
                    "asset_mgr_long": self._safe_int(record.get("asset_mgr_positions_long", 0)),
                    "asset_mgr_short": self._safe_int(record.get("asset_mgr_positions_short", 0)),
                    "lev_money_long": self._safe_int(record.get("lev_money_positions_long", 0)),
                    "lev_money_short": self._safe_int(record.get("lev_money_positions_short", 0)),
                    "nonrept_long": self._safe_int(record.get("nonrept_positions_long_all", 0)),
                    "nonrept_short": self._safe_int(record.get("nonrept_positions_short_all", 0)),
                }

                item = self._create_cot_item(contract_name, symbol, cot_data)
                if item:
                    items.append(item)
                    logger.debug(f"Created COT item for {symbol}")

        except Exception as e:
            logger.error(f"Error collecting COT data: {e}")
            import traceback
            traceback.print_exc()

        logger.info(f"Collected {len(items)} COT reports")
        return items

    def _safe_int(self, value) -> int:
        """Safely convert to int."""
        try:
            if value is None:
                return 0
            return int(float(str(value).replace(",", "")))
        except (ValueError, TypeError):
            return 0

    def _match_contract(self, market_name: str, commodity_name: str) -> Optional[str]:
        """Match contract name to our trading symbol."""
        combined = f"{market_name} {commodity_name}".upper()

        for keyword, symbol in CONTRACT_MAP.items():
            if keyword.upper() in combined:
                return symbol
        return None

    def _create_cot_item(
        self,
        contract_name: str,
        symbol: str,
        data: Dict[str, Any]
    ) -> Optional[KnowledgeItem]:
        """Create KnowledgeItem from COT TFF data."""
        try:
            # TFF categories:
            # - Dealer (banks) - usually hedging, fade them
            # - Asset Manager (pension funds) - trend followers
            # - Leveraged Money (hedge funds) - speculators, key signal
            # - Other Reportable + Non-reportable (retail)

            # Calculate net positions for each category
            dealer_net = data.get("dealer_long", 0) - data.get("dealer_short", 0)
            asset_mgr_net = data.get("asset_mgr_long", 0) - data.get("asset_mgr_short", 0)
            lev_money_net = data.get("lev_money_long", 0) - data.get("lev_money_short", 0)

            # Leveraged Money (hedge funds) is the key speculator signal
            if lev_money_net > 0:
                spec_bias = "LONG"
                spec_strength = "strong" if abs(lev_money_net) > 50000 else "moderate"
            else:
                spec_bias = "SHORT"
                spec_strength = "strong" if abs(lev_money_net) > 50000 else "moderate"

            report_date = data.get("report_date", datetime.now().strftime('%Y-%m-%d'))
            if "T" in str(report_date):
                report_date = str(report_date).split("T")[0]

            # Create summary
            summary = f"""COT Report for {contract_name} ({symbol}):

Leveraged Money (Hedge Funds): Net {spec_bias} ({lev_money_net:,}) - {spec_strength}
Asset Managers: Net {'LONG' if asset_mgr_net > 0 else 'SHORT'} ({asset_mgr_net:,})
Dealers (Banks): Net {'LONG' if dealer_net > 0 else 'SHORT'} ({dealer_net:,})
Open Interest: {data.get('open_interest', 0):,} contracts

Trading Implication: Hedge funds are {spec_strength}ly {spec_bias.lower()} on {symbol}.
{'⚠️ Extreme positioning - consider contrarian SHORT.' if spec_bias == 'LONG' and spec_strength == 'strong' else ''}
{'⚠️ Extreme positioning - consider contrarian LONG.' if spec_bias == 'SHORT' and spec_strength == 'strong' else ''}
"""

            full_content = f"""CFTC Commitment of Traders (TFF) Analysis

Contract: {contract_name}
Symbol: {symbol}
Report Date: {report_date}

=== Position Breakdown (TFF Categories) ===

Dealers (Banks, Market Makers):
- Long: {data.get('dealer_long', 0):,}
- Short: {data.get('dealer_short', 0):,}
- Net: {dealer_net:,}
- Note: Usually hedging client flow, fade large positions

Asset Managers (Pension Funds, Institutions):
- Long: {data.get('asset_mgr_long', 0):,}
- Short: {data.get('asset_mgr_short', 0):,}
- Net: {asset_mgr_net:,}
- Note: Trend followers, slow to change positions

Leveraged Money (Hedge Funds, CTAs):
- Long: {data.get('lev_money_long', 0):,}
- Short: {data.get('lev_money_short', 0):,}
- Net: {lev_money_net:,}
- Bias: {spec_bias} ({spec_strength})
- Note: KEY SIGNAL - active speculators

Non-Reportable (Retail):
- Long: {data.get('nonrept_long', 0):,}
- Short: {data.get('nonrept_short', 0):,}

Open Interest: {data.get('open_interest', 0):,}

=== Trading Signals ===

1. Leveraged Money Positioning (Primary Signal):
   - Extreme long = potential top, consider shorts
   - Extreme short = potential bottom, consider longs
   - Current: {spec_strength} {spec_bias.lower()}

2. Dealer vs Hedge Fund Divergence:
   - When dealers opposite to hedge funds = reversal coming
   - Dealers net: {dealer_net:,} vs Hedge funds: {lev_money_net:,}

3. Asset Manager Confirmation:
   - Asset managers confirm trend when aligned with hedge funds
   - Asset managers: {asset_mgr_net:,}

=== HYDRA Integration ===

For {symbol} trades:
- Speculator (Hedge Fund) bias: {spec_bias}
- Confidence modifier: {0.8 if spec_strength == 'strong' else 1.0}x for counter-bias trades
- Contrarian signal: {'ACTIVE' if spec_strength == 'strong' else 'INACTIVE'}
"""

            item = KnowledgeItem(
                source=KnowledgeSource.CFTC,
                source_url=f"https://publicreporting.cftc.gov/stories/s/TFF-Futures-Only/98ig-3k9y/",
                title=f"COT Report: {symbol} - Hedge Funds {spec_bias}",
                content_type=ContentType.MARKET_DATA,
                summary=summary,
                full_content=full_content,
                quality_score=0.9,  # Official government data
            )

            item.symbols = [symbol]
            item.tags = [
                "cot", "institutional", "positioning", "tff",
                spec_bias.lower(), "hedge_funds"
            ]

            return item

        except Exception as e:
            logger.debug(f"Error creating COT item: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()


async def run_cot_collector() -> int:
    """Run the COT collector and save results."""
    collector = COTCollector()
    storage = get_storage()

    log_id = storage.start_scrape_log(KnowledgeSource.CFTC, "cot_collect")

    try:
        items = await collector.collect()

        saved = 0
        for item in items:
            try:
                storage.save_item(item)
                saved += 1
            except Exception as e:
                logger.error(f"Error saving COT item: {e}")

        storage.complete_scrape_log(log_id, "success", len(items), saved)
        logger.info(f"COT collection complete: {saved}/{len(items)} saved")
        return saved

    except Exception as e:
        storage.complete_scrape_log(log_id, "failed", 0, 0, str(e))
        logger.error(f"COT collection failed: {e}")
        raise

    finally:
        await collector.close()


if __name__ == "__main__":
    asyncio.run(run_cot_collector())
