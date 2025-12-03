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

Uses Binance Futures API (free, no auth needed).
"""

import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import httpx

logger = logging.getLogger(__name__)


# API Configuration
BINANCE_FUTURES_URL = "https://fapi.binance.com"
CACHE_TTL_SECONDS = 60  # Cache for 1 minute

# Funding rate thresholds (annualized)
EXTREME_HIGH_ANNUAL = 100.0    # >100% APR = extremely bullish crowd
EXTREME_LOW_ANNUAL = -50.0     # <-50% APR = extremely bearish crowd
HIGH_ANNUAL = 50.0             # >50% APR = bullish
LOW_ANNUAL = -25.0             # <-25% APR = bearish

# Symbol mapping (Coinbase spot -> Binance futures)
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
    "BNB-USD": "BNBUSDT",
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

        # Auto-detect data directory
        if data_dir is None:
            if os.path.exists("/root/crpbot"):
                data_dir = Path("/root/crpbot/data/hydra")
            else:
                data_dir = Path.home() / "crpbot" / "data" / "hydra"

        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Cache
        self.cache: dict[str, tuple[datetime, FundingRate]] = {}
        self.snapshot_cache: Optional[tuple[datetime, FundingSnapshot]] = None

        self._initialized = True
        logger.info("FundingRatesFeed initialized")

    def _convert_symbol(self, symbol: str) -> str:
        """Convert Coinbase symbol to Binance futures symbol."""
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
        Fetch current funding rate for a symbol.

        Args:
            symbol: Trading pair (Coinbase format like BTC-USD)

        Returns:
            FundingRate or None on error
        """
        binance_symbol = self._convert_symbol(symbol)

        # Check cache
        cache_key = binance_symbol
        if cache_key in self.cache:
            cached_time, cached_rate = self.cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < CACHE_TTL_SECONDS:
                return cached_rate

        url = f"{BINANCE_FUTURES_URL}/fapi/v1/premiumIndex"
        params = {"symbol": binance_symbol}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

            funding_rate = float(data["lastFundingRate"])
            annual_rate = funding_rate * 3 * 365 * 100  # 3 funding periods/day * 365 days

            sentiment, strength, contrarian = self._analyze_sentiment(funding_rate)

            rate = FundingRate(
                symbol=symbol,
                funding_rate=funding_rate,
                funding_rate_annual=annual_rate,
                next_funding_time=datetime.fromtimestamp(int(data["nextFundingTime"]) / 1000),
                mark_price=float(data["markPrice"]),
                index_price=float(data["indexPrice"]),
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
        Fetch funding rates for all major symbols.

        Returns:
            FundingSnapshot with all rates
        """
        # Check cache
        if self.snapshot_cache:
            cached_time, cached_snapshot = self.snapshot_cache
            if (datetime.now() - cached_time).total_seconds() < CACHE_TTL_SECONDS:
                return cached_snapshot

        url = f"{BINANCE_FUTURES_URL}/fapi/v1/premiumIndex"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                all_data = resp.json()

            rates = {}
            bullish_count = 0
            bearish_count = 0
            extreme_count = 0
            total_annual = 0.0

            # Filter to our symbols
            target_symbols = set(SYMBOL_MAP.values())

            for data in all_data:
                binance_symbol = data["symbol"]
                if binance_symbol not in target_symbols:
                    continue

                # Convert back to Coinbase format
                coinbase_symbol = next(
                    (k for k, v in SYMBOL_MAP.items() if v == binance_symbol),
                    binance_symbol
                )

                funding_rate = float(data["lastFundingRate"])
                annual_rate = funding_rate * 3 * 365 * 100

                sentiment, strength, contrarian = self._analyze_sentiment(funding_rate)

                rate = FundingRate(
                    symbol=coinbase_symbol,
                    funding_rate=funding_rate,
                    funding_rate_annual=annual_rate,
                    next_funding_time=datetime.fromtimestamp(int(data["nextFundingTime"]) / 1000),
                    mark_price=float(data["markPrice"]),
                    index_price=float(data["indexPrice"]),
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
        Fetch historical funding rates.

        Args:
            symbol: Trading pair
            limit: Number of historical records

        Returns:
            List of historical funding records
        """
        binance_symbol = self._convert_symbol(symbol)
        url = f"{BINANCE_FUTURES_URL}/fapi/v1/fundingRate"
        params = {"symbol": binance_symbol, "limit": limit}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

            history = []
            for record in data:
                funding_rate = float(record["fundingRate"])
                annual_rate = funding_rate * 3 * 365 * 100

                history.append({
                    "symbol": symbol,
                    "funding_rate": funding_rate,
                    "funding_rate_annual": annual_rate,
                    "funding_time": datetime.fromtimestamp(int(record["fundingTime"]) / 1000).isoformat(),
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
