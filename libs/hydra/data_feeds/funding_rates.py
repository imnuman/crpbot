"""
HYDRA 3.0 - Funding Rates Data Feed

Provides perpetual futures funding rate data:
- Current funding rates across exchanges
- Funding rate history
- Sentiment analysis from funding
- Extreme funding alerts

Funding rates indicate market sentiment:
- Positive = Longs pay shorts = Bullish crowd (contrarian bearish)
- Negative = Shorts pay longs = Bearish crowd (contrarian bullish)
- Extreme rates often precede reversals

Uses Bybit API (works in Canada, Binance blocked).
"""

import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import httpx

logger = logging.getLogger(__name__)


# API Configuration - Using Bybit instead of Binance (blocked in Canada)
BYBIT_URL = "https://api.bybit.com"
CACHE_TTL_SECONDS = 60  # Cache for 1 minute

# Funding rate thresholds (annualized)
EXTREME_HIGH_ANNUAL = 100.0    # >100% APR = extremely bullish crowd
EXTREME_LOW_ANNUAL = -50.0     # <-50% APR = extremely bearish crowd
HIGH_ANNUAL = 50.0             # >50% APR = bullish
LOW_ANNUAL = -25.0             # <-25% APR = bearish

# Symbol mapping (Coinbase spot -> Bybit futures)
SYMBOL_MAP = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
    "SOL-USD": "SOLUSDT",
    "XRP-USD": "XRPUSDT",
    "DOGE-USD": "DOGEUSDT",
    "ADA-USD": "ADAUSDT",
    "AVAX-USD": "AVAXUSDT",
    "LINK-USD": "LINKUSDT",
    "DOT-USD": "DOTUSDT",
    "MATIC-USD": "MATICUSDT",
    "LTC-USD": "LTCUSDT",
}


@dataclass
class FundingRate:
    """Funding rate data for a symbol."""
    symbol: str
    funding_rate: float           # Current rate (per 8 hours)
    funding_rate_annual: float    # Annualized rate
    next_funding_time: datetime
    mark_price: float
    index_price: float
    timestamp: datetime

    # Sentiment derived from funding
    sentiment: str                # BULLISH_CROWD, BEARISH_CROWD, NEUTRAL
    sentiment_strength: str       # EXTREME, HIGH, MODERATE, LOW
    contrarian_signal: str        # LONG, SHORT, NEUTRAL

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "next_funding_time": self.next_funding_time.isoformat(),
            "timestamp": self.timestamp.isoformat(),
        }

    def format_for_llm(self) -> str:
        """Format for LLM consumption."""
        time_to_funding = self.next_funding_time - datetime.now()
        hours_to_funding = time_to_funding.total_seconds() / 3600

        return (
            f"{self.symbol}: {self.funding_rate*100:.4f}% (8h) / {self.funding_rate_annual:.1f}% APR\n"
            f"  Sentiment: {self.sentiment} ({self.sentiment_strength})\n"
            f"  Contrarian: {self.contrarian_signal}\n"
            f"  Next funding in: {hours_to_funding:.1f}h"
        )


@dataclass
class FundingSnapshot:
    """Snapshot of funding rates across multiple symbols."""
    timestamp: datetime
    rates: dict[str, FundingRate]

    # Aggregate metrics
    avg_funding_annual: float
    bullish_count: int
    bearish_count: int
    extreme_count: int
    market_sentiment: str

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "rates": {k: v.to_dict() for k, v in self.rates.items()},
            "avg_funding_annual": self.avg_funding_annual,
            "bullish_count": self.bullish_count,
            "bearish_count": self.bearish_count,
            "extreme_count": self.extreme_count,
            "market_sentiment": self.market_sentiment,
        }

    def format_for_llm(self) -> str:
        """Format snapshot for LLM."""
        lines = [
            "=== Funding Rates Analysis ===",
            f"Market Sentiment: {self.market_sentiment}",
            f"Avg Funding (APR): {self.avg_funding_annual:+.1f}%",
            f"Bullish crowd: {self.bullish_count} | Bearish crowd: {self.bearish_count}",
            f"Extreme readings: {self.extreme_count}",
            "",
            "Individual Rates:",
        ]

        # Sort by absolute funding rate
        sorted_rates = sorted(
            self.rates.values(),
            key=lambda x: abs(x.funding_rate_annual),
            reverse=True
        )

        for rate in sorted_rates[:10]:  # Top 10
            emoji = "🔴" if rate.funding_rate > 0 else "🟢" if rate.funding_rate < 0 else "⚪"
            lines.append(f"  {emoji} {rate.symbol}: {rate.funding_rate_annual:+.1f}% APR ({rate.contrarian_signal})")

        return "\n".join(lines)


class FundingRatesFeed:
    """
    Funding rates data feed using Binance Futures API.

    Features:
    - Real-time funding rates
    - Historical funding data
    - Sentiment analysis
    - Contrarian signals
    """

    _instance: Optional["FundingRatesFeed"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, data_dir: Optional[Path] = None):
        if self._initialized:
            return

        # Use central config for path detection
        if data_dir is None:
            from ..config import HYDRA_DATA_DIR
            data_dir = HYDRA_DATA_DIR

        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Cache
        self.cache: dict[str, tuple[datetime, FundingRate]] = {}
        self.snapshot_cache: Optional[tuple[datetime, FundingSnapshot]] = None

        self._initialized = True
        logger.info("FundingRatesFeed initialized")

    def _convert_symbol(self, symbol: str) -> str:
        """Convert Coinbase symbol to Bybit futures symbol."""
        return SYMBOL_MAP.get(symbol, symbol.replace("-USD", "USDT"))

    def _analyze_sentiment(self, funding_rate: float) -> tuple[str, str, str]:
        """
        Analyze sentiment from funding rate.

        Returns:
            (sentiment, strength, contrarian_signal)
        """
        annual = funding_rate * 3 * 365 * 100  # Convert to annualized %

        # Determine crowd sentiment
        if annual > EXTREME_HIGH_ANNUAL:
            sentiment = "BULLISH_CROWD"
            strength = "EXTREME"
            contrarian = "SHORT"  # Fade the crowd
        elif annual > HIGH_ANNUAL:
            sentiment = "BULLISH_CROWD"
            strength = "HIGH"
            contrarian = "SHORT"
        elif annual > 10:
            sentiment = "BULLISH_CROWD"
            strength = "MODERATE"
            contrarian = "NEUTRAL"
        elif annual < EXTREME_LOW_ANNUAL:
            sentiment = "BEARISH_CROWD"
            strength = "EXTREME"
            contrarian = "LONG"  # Fade the crowd
        elif annual < LOW_ANNUAL:
            sentiment = "BEARISH_CROWD"
            strength = "HIGH"
            contrarian = "LONG"
        elif annual < -10:
            sentiment = "BEARISH_CROWD"
            strength = "MODERATE"
            contrarian = "NEUTRAL"
        else:
            sentiment = "NEUTRAL"
            strength = "LOW"
            contrarian = "NEUTRAL"

        return sentiment, strength, contrarian

    async def fetch_funding_rate(self, symbol: str = "BTC-USD") -> Optional[FundingRate]:
        """
        Fetch current funding rate for a symbol using Bybit API.

        Args:
            symbol: Trading pair (Coinbase format like BTC-USD)

        Returns:
            FundingRate or None on error
        """
        bybit_symbol = self._convert_symbol(symbol)

        # Check cache
        cache_key = bybit_symbol
        if cache_key in self.cache:
            cached_time, cached_rate = self.cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < CACHE_TTL_SECONDS:
                return cached_rate

        url = f"{BYBIT_URL}/v5/market/tickers"
        params = {"category": "linear", "symbol": bybit_symbol}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

            if data.get("retCode") != 0 or not data.get("result", {}).get("list"):
                logger.warning(f"Bybit returned no data for {symbol}")
                return None

            ticker = data["result"]["list"][0]
            funding_rate = float(ticker.get("fundingRate", 0))
            annual_rate = funding_rate * 3 * 365 * 100  # 3 funding periods/day * 365 days

            sentiment, strength, contrarian = self._analyze_sentiment(funding_rate)

            # Bybit provides mark/index prices
            mark_price = float(ticker.get("markPrice", 0))
            index_price = float(ticker.get("indexPrice", 0))
            next_funding_ts = int(ticker.get("nextFundingTime", 0))

            rate = FundingRate(
                symbol=symbol,
                funding_rate=funding_rate,
                funding_rate_annual=annual_rate,
                next_funding_time=datetime.fromtimestamp(next_funding_ts / 1000) if next_funding_ts else datetime.now(),
                mark_price=mark_price,
                index_price=index_price,
                timestamp=datetime.now(),
                sentiment=sentiment,
                sentiment_strength=strength,
                contrarian_signal=contrarian,
            )

            # Cache
            self.cache[cache_key] = (datetime.now(), rate)

            return rate

        except Exception as e:
            logger.error(f"Failed to fetch funding rate for {symbol}: {e}")
            return None

    async def fetch_all_funding_rates(self) -> Optional[FundingSnapshot]:
        """
        Fetch funding rates for all major symbols using Bybit API.

        Returns:
            FundingSnapshot with all rates
        """
        # Check cache
        if self.snapshot_cache:
            cached_time, cached_snapshot = self.snapshot_cache
            if (datetime.now() - cached_time).total_seconds() < CACHE_TTL_SECONDS:
                return cached_snapshot

        # Bybit V5 API - get all linear tickers
        url = f"{BYBIT_URL}/v5/market/tickers"
        params = {"category": "linear"}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

            if data.get("retCode") != 0 or not data.get("result", {}).get("list"):
                logger.warning("Bybit returned no data for all tickers")
                return None

            all_data = data["result"]["list"]

            rates = {}
            bullish_count = 0
            bearish_count = 0
            extreme_count = 0
            total_annual = 0.0

            # Filter to our symbols
            target_symbols = set(SYMBOL_MAP.values())

            for ticker in all_data:
                bybit_symbol = ticker["symbol"]
                if bybit_symbol not in target_symbols:
                    continue

                # Convert back to Coinbase format
                coinbase_symbol = next(
                    (k for k, v in SYMBOL_MAP.items() if v == bybit_symbol),
                    bybit_symbol
                )

                funding_rate = float(ticker.get("fundingRate", 0))
                annual_rate = funding_rate * 3 * 365 * 100

                sentiment, strength, contrarian = self._analyze_sentiment(funding_rate)

                mark_price = float(ticker.get("markPrice", 0))
                index_price = float(ticker.get("indexPrice", 0))
                next_funding_ts = int(ticker.get("nextFundingTime", 0))

                rate = FundingRate(
                    symbol=coinbase_symbol,
                    funding_rate=funding_rate,
                    funding_rate_annual=annual_rate,
                    next_funding_time=datetime.fromtimestamp(next_funding_ts / 1000) if next_funding_ts else datetime.now(),
                    mark_price=mark_price,
                    index_price=index_price,
                    timestamp=datetime.now(),
                    sentiment=sentiment,
                    sentiment_strength=strength,
                    contrarian_signal=contrarian,
                )

                rates[coinbase_symbol] = rate
                total_annual += annual_rate

                if "BULLISH" in sentiment:
                    bullish_count += 1
                elif "BEARISH" in sentiment:
                    bearish_count += 1

                if strength == "EXTREME":
                    extreme_count += 1

            # Determine overall market sentiment
            avg_annual = total_annual / len(rates) if rates else 0

            if avg_annual > 30:
                market_sentiment = "CROWDED_LONG (contrarian bearish)"
            elif avg_annual > 15:
                market_sentiment = "LEANING_LONG"
            elif avg_annual < -20:
                market_sentiment = "CROWDED_SHORT (contrarian bullish)"
            elif avg_annual < -10:
                market_sentiment = "LEANING_SHORT"
            else:
                market_sentiment = "NEUTRAL"

            snapshot = FundingSnapshot(
                timestamp=datetime.now(),
                rates=rates,
                avg_funding_annual=avg_annual,
                bullish_count=bullish_count,
                bearish_count=bearish_count,
                extreme_count=extreme_count,
                market_sentiment=market_sentiment,
            )

            # Cache
            self.snapshot_cache = (datetime.now(), snapshot)

            return snapshot

        except Exception as e:
            logger.error(f"Failed to fetch all funding rates: {e}")
            return None

    async def fetch_funding_history(
        self,
        symbol: str = "BTC-USD",
        limit: int = 100
    ) -> list[dict]:
        """
        Fetch historical funding rates using Bybit API.

        Args:
            symbol: Trading pair
            limit: Number of historical records

        Returns:
            List of historical funding records
        """
        bybit_symbol = self._convert_symbol(symbol)
        url = f"{BYBIT_URL}/v5/market/funding/history"
        params = {"category": "linear", "symbol": bybit_symbol, "limit": min(limit, 200)}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

            if data.get("retCode") != 0 or not data.get("result", {}).get("list"):
                logger.warning(f"Bybit returned no history for {symbol}")
                return []

            history = []
            for record in data["result"]["list"]:
                funding_rate = float(record.get("fundingRate", 0))
                annual_rate = funding_rate * 3 * 365 * 100
                funding_ts = int(record.get("fundingRateTimestamp", 0))

                history.append({
                    "symbol": symbol,
                    "funding_rate": funding_rate,
                    "funding_rate_annual": annual_rate,
                    "funding_time": datetime.fromtimestamp(funding_ts / 1000).isoformat() if funding_ts else None,
                })

            return history

        except Exception as e:
            logger.error(f"Failed to fetch funding history for {symbol}: {e}")
            return []

    def get_extreme_funding_alert(self, snapshot: FundingSnapshot) -> Optional[str]:
        """
        Generate alert if funding rates are extreme.

        Returns:
            Alert message or None
        """
        if snapshot.extreme_count == 0:
            return None

        extreme_rates = [
            r for r in snapshot.rates.values()
            if r.sentiment_strength == "EXTREME"
        ]

        if not extreme_rates:
            return None

        lines = ["⚠️ EXTREME FUNDING ALERT ⚠️"]
        for rate in extreme_rates:
            direction = "LONG" if rate.funding_rate > 0 else "SHORT"
            lines.append(
                f"  {rate.symbol}: {rate.funding_rate_annual:+.1f}% APR - "
                f"Crowd is {direction}, consider {rate.contrarian_signal}"
            )

        return "\n".join(lines)


# Singleton accessor
_funding_instance: Optional[FundingRatesFeed] = None


def get_funding_rates_feed() -> FundingRatesFeed:
    """Get or create the funding rates feed singleton."""
    global _funding_instance
    if _funding_instance is None:
        _funding_instance = FundingRatesFeed()
    return _funding_instance
