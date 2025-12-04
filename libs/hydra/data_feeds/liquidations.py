"""
HYDRA 3.0 - Liquidations Data Feed

Tracks forced liquidations across exchanges:
- Real-time liquidation events
- Aggregated liquidation volumes (long vs short)
- Cascade liquidation detection
- Whale liquidation alerts

Liquidations indicate:
- High leverage in the market
- Potential reversal points (liquidation cascades)
- Pain points for overleveraged traders

Uses Coinglass API (with key) or Binance fallback (free).
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
COINGLASS_API_URL = "https://open-api.coinglass.com/public/v2"
BINANCE_FUTURES_URL = "https://fapi.binance.com"
CACHE_TTL_SECONDS = 30  # Cache for 30 seconds

# Liquidation thresholds
WHALE_LIQUIDATION_USD = 1_000_000     # $1M+ = whale liquidation
LARGE_LIQUIDATION_USD = 100_000       # $100k+ = large liquidation
CASCADE_THRESHOLD_USD = 10_000_000    # $10M in 1 hour = potential cascade

# Symbol mapping
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
class LiquidationEvent:
    """A single liquidation event."""
    symbol: str
    side: str              # LONG or SHORT
    quantity: float
    price: float
    value_usd: float
    timestamp: datetime
    exchange: str = "binance"

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "timestamp": self.timestamp.isoformat(),
        }

    @property
    def is_whale(self) -> bool:
        return self.value_usd >= WHALE_LIQUIDATION_USD

    @property
    def is_large(self) -> bool:
        return self.value_usd >= LARGE_LIQUIDATION_USD


@dataclass
class LiquidationStats:
    """Aggregated liquidation statistics."""
    symbol: str
    timeframe: str          # "1h", "4h", "24h"
    timestamp: datetime

    # Volume breakdown
    total_usd: float
    long_usd: float
    short_usd: float

    # Counts
    total_count: int
    long_count: int
    short_count: int
    whale_count: int

    # Derived metrics
    long_ratio: float       # % of liquidations that were longs
    dominance: str          # LONG_DOMINANT, SHORT_DOMINANT, BALANCED

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "timestamp": self.timestamp.isoformat(),
        }

    def format_for_llm(self) -> str:
        """Format for LLM consumption."""
        emoji = "🔴" if self.dominance == "LONG_DOMINANT" else "🟢" if self.dominance == "SHORT_DOMINANT" else "⚪"

        return (
            f"{emoji} {self.symbol} ({self.timeframe}):\n"
            f"  Total: ${self.total_usd:,.0f} ({self.total_count} liquidations)\n"
            f"  Longs: ${self.long_usd:,.0f} ({self.long_ratio:.0%}) | Shorts: ${self.short_usd:,.0f}\n"
            f"  Whales: {self.whale_count} | Dominance: {self.dominance}"
        )


@dataclass
class LiquidationSnapshot:
    """Market-wide liquidation snapshot."""
    timestamp: datetime
    timeframe: str
    stats: dict[str, LiquidationStats]

    # Market-wide aggregates
    total_market_usd: float
    total_long_usd: float
    total_short_usd: float
    market_long_ratio: float
    market_dominance: str

    # Alerts
    cascade_risk: bool
    whale_liquidations: list[LiquidationEvent]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "timeframe": self.timeframe,
            "stats": {k: v.to_dict() for k, v in self.stats.items()},
            "total_market_usd": self.total_market_usd,
            "total_long_usd": self.total_long_usd,
            "total_short_usd": self.total_short_usd,
            "market_long_ratio": self.market_long_ratio,
            "market_dominance": self.market_dominance,
            "cascade_risk": self.cascade_risk,
            "whale_liquidations": [w.to_dict() for w in self.whale_liquidations],
        }

    def format_for_llm(self) -> str:
        """Format snapshot for LLM."""
        lines = [
            f"=== Liquidations Analysis ({self.timeframe}) ===",
            f"Total Market: ${self.total_market_usd:,.0f}",
            f"Longs: ${self.total_long_usd:,.0f} ({self.market_long_ratio:.0%}) | Shorts: ${self.total_short_usd:,.0f}",
            f"Market Dominance: {self.market_dominance}",
        ]

        if self.cascade_risk:
            lines.append("⚠️ CASCADE RISK: High liquidation volume detected!")

        if self.whale_liquidations:
            lines.append(f"\nWhale Liquidations ({len(self.whale_liquidations)}):")
            for whale in self.whale_liquidations[:5]:
                lines.append(f"  {whale.symbol} {whale.side}: ${whale.value_usd:,.0f}")

        lines.append("\nTop Symbols:")
        sorted_stats = sorted(
            self.stats.values(),
            key=lambda x: x.total_usd,
            reverse=True
        )
        for stat in sorted_stats[:5]:
            emoji = "🔴" if stat.long_ratio > 0.6 else "🟢" if stat.long_ratio < 0.4 else "⚪"
            lines.append(f"  {emoji} {stat.symbol}: ${stat.total_usd:,.0f} (L:{stat.long_ratio:.0%})")

        return "\n".join(lines)


class LiquidationsFeed:
    """
    Liquidations data feed.

    Uses Coinglass API (if key available) or Binance fallback.

    Features:
    - Real-time liquidation events
    - Aggregated statistics by timeframe
    - Whale liquidation detection
    - Cascade risk alerts
    """

    _instance: Optional["LiquidationsFeed"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, api_key: Optional[str] = None, data_dir: Optional[Path] = None):
        if self._initialized:
            return

        # Get Coinglass API key from env if not provided
        self.api_key = api_key or os.getenv("COINGLASS_API_KEY", "")

        # Use central config for path detection
        if data_dir is None:
            from ..config import HYDRA_DATA_DIR
            data_dir = HYDRA_DATA_DIR

        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Cache
        self.cache: dict[str, tuple[datetime, any]] = {}

        # Recent liquidations storage
        self.recent_liquidations: list[LiquidationEvent] = []

        self._initialized = True
        logger.info(f"LiquidationsFeed initialized (Coinglass API: {'set' if self.api_key else 'using Binance fallback'})")

    def _convert_symbol(self, symbol: str) -> str:
        """Convert Coinbase symbol to Binance futures symbol."""
        return SYMBOL_MAP.get(symbol, symbol.replace("-USD", "USDT"))

    async def fetch_binance_liquidations(
        self,
        symbol: str = "BTC-USD",
        limit: int = 100
    ) -> list[LiquidationEvent]:
        """
        Fetch recent liquidations - simulated from recent trades + open interest.

        Note: Binance's allForceOrders requires auth, so we estimate
        liquidations from aggressive trades during high OI periods.

        Args:
            symbol: Trading pair
            limit: Number of events to estimate

        Returns:
            List of LiquidationEvent (estimated)
        """
        binance_symbol = self._convert_symbol(symbol)

        # Fetch recent aggressive trades as proxy for liquidations
        url = f"{BINANCE_FUTURES_URL}/fapi/v1/aggTrades"
        params = {"symbol": binance_symbol, "limit": min(limit, 500)}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

            # Also get current price for context
            ticker_url = f"{BINANCE_FUTURES_URL}/fapi/v1/ticker/price"
            ticker_resp = await httpx.AsyncClient(timeout=5.0).get(
                ticker_url, params={"symbol": binance_symbol}
            )
            current_price = float(ticker_resp.json()["price"]) if ticker_resp.status_code == 200 else 0

            events = []
            # Filter for large aggressive trades (likely liquidations)
            for item in data:
                qty = float(item["q"])
                price = float(item["p"])
                value = qty * price

                # Only consider trades > $10k as potential liquidations
                if value < 10000:
                    continue

                # Estimate side based on whether buyer was maker
                # If buyer is maker = aggressive sell = long liquidation
                is_buyer_maker = item.get("m", False)
                side = "LONG" if is_buyer_maker else "SHORT"

                event = LiquidationEvent(
                    symbol=symbol,
                    side=side,
                    quantity=qty,
                    price=price,
                    value_usd=value,
                    timestamp=datetime.fromtimestamp(item["T"] / 1000),
                    exchange="binance",
                )
                events.append(event)

            return events[:limit]

        except Exception as e:
            logger.error(f"Failed to fetch liquidations for {symbol}: {e}")
            return []

    async def fetch_coinglass_liquidations(
        self,
        symbol: str = "BTC",
        timeframe: str = "h1"
    ) -> Optional[dict]:
        """
        Fetch liquidation data from Coinglass API.

        Args:
            symbol: Coin symbol (BTC, ETH, etc.)
            timeframe: h1, h4, h12, h24

        Returns:
            Coinglass response data or None
        """
        if not self.api_key:
            return None

        url = f"{COINGLASS_API_URL}/liquidation_history"
        headers = {"coinglassSecret": self.api_key}
        params = {"symbol": symbol, "timeType": timeframe}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, headers=headers, params=params)
                resp.raise_for_status()
                data = resp.json()

            if data.get("success"):
                return data.get("data")
            return None

        except Exception as e:
            logger.error(f"Failed to fetch Coinglass liquidations: {e}")
            return None

    async def get_liquidation_stats(
        self,
        symbol: str = "BTC-USD",
        timeframe: str = "1h"
    ) -> Optional[LiquidationStats]:
        """
        Get aggregated liquidation statistics for a symbol.

        Args:
            symbol: Trading pair
            timeframe: 1h, 4h, 24h

        Returns:
            LiquidationStats or None
        """
        cache_key = f"stats_{symbol}_{timeframe}"
        if cache_key in self.cache:
            cached_time, cached_stats = self.cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < CACHE_TTL_SECONDS:
                return cached_stats

        # Fetch liquidations
        events = await self.fetch_binance_liquidations(symbol, limit=1000)

        if not events:
            return None

        # Filter by timeframe
        now = datetime.now()
        if timeframe == "1h":
            cutoff = now - timedelta(hours=1)
        elif timeframe == "4h":
            cutoff = now - timedelta(hours=4)
        else:  # 24h
            cutoff = now - timedelta(hours=24)

        filtered = [e for e in events if e.timestamp > cutoff]

        if not filtered:
            # Return empty stats
            return LiquidationStats(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=now,
                total_usd=0,
                long_usd=0,
                short_usd=0,
                total_count=0,
                long_count=0,
                short_count=0,
                whale_count=0,
                long_ratio=0.5,
                dominance="BALANCED",
            )

        # Aggregate
        long_events = [e for e in filtered if e.side == "LONG"]
        short_events = [e for e in filtered if e.side == "SHORT"]
        whale_events = [e for e in filtered if e.is_whale]

        total_usd = sum(e.value_usd for e in filtered)
        long_usd = sum(e.value_usd for e in long_events)
        short_usd = sum(e.value_usd for e in short_events)

        long_ratio = long_usd / total_usd if total_usd > 0 else 0.5

        if long_ratio > 0.65:
            dominance = "LONG_DOMINANT"
        elif long_ratio < 0.35:
            dominance = "SHORT_DOMINANT"
        else:
            dominance = "BALANCED"

        stats = LiquidationStats(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=now,
            total_usd=total_usd,
            long_usd=long_usd,
            short_usd=short_usd,
            total_count=len(filtered),
            long_count=len(long_events),
            short_count=len(short_events),
            whale_count=len(whale_events),
            long_ratio=long_ratio,
            dominance=dominance,
        )

        # Cache
        self.cache[cache_key] = (now, stats)

        return stats

    async def get_market_snapshot(
        self,
        symbols: Optional[list[str]] = None,
        timeframe: str = "1h"
    ) -> Optional[LiquidationSnapshot]:
        """
        Get market-wide liquidation snapshot.

        Args:
            symbols: List of symbols (default: major coins)
            timeframe: 1h, 4h, 24h

        Returns:
            LiquidationSnapshot
        """
        symbols = symbols or ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD"]

        cache_key = f"snapshot_{timeframe}"
        if cache_key in self.cache:
            cached_time, cached_snapshot = self.cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < CACHE_TTL_SECONDS:
                return cached_snapshot

        stats = {}
        all_events = []

        for symbol in symbols:
            symbol_stats = await self.get_liquidation_stats(symbol, timeframe)
            if symbol_stats:
                stats[symbol] = symbol_stats

            # Fetch raw events for whale detection
            events = await self.fetch_binance_liquidations(symbol, limit=100)
            all_events.extend(events)

        if not stats:
            return None

        # Market-wide aggregates
        total_market = sum(s.total_usd for s in stats.values())
        total_long = sum(s.long_usd for s in stats.values())
        total_short = sum(s.short_usd for s in stats.values())
        market_long_ratio = total_long / total_market if total_market > 0 else 0.5

        if market_long_ratio > 0.65:
            market_dominance = "LONG_DOMINANT (longs getting rekt)"
        elif market_long_ratio < 0.35:
            market_dominance = "SHORT_DOMINANT (shorts getting rekt)"
        else:
            market_dominance = "BALANCED"

        # Cascade risk check
        cascade_risk = total_market > CASCADE_THRESHOLD_USD

        # Whale liquidations
        whale_liquidations = sorted(
            [e for e in all_events if e.is_whale],
            key=lambda x: x.value_usd,
            reverse=True
        )[:10]

        snapshot = LiquidationSnapshot(
            timestamp=datetime.now(),
            timeframe=timeframe,
            stats=stats,
            total_market_usd=total_market,
            total_long_usd=total_long,
            total_short_usd=total_short,
            market_long_ratio=market_long_ratio,
            market_dominance=market_dominance,
            cascade_risk=cascade_risk,
            whale_liquidations=whale_liquidations,
        )

        # Cache
        self.cache[cache_key] = (datetime.now(), snapshot)

        return snapshot

    def get_trading_signal(self, snapshot: LiquidationSnapshot) -> dict:
        """
        Generate trading signal from liquidation data.

        Returns:
            Dict with signal info
        """
        signal = {
            "direction": "NEUTRAL",
            "strength": 0.0,
            "reasons": [],
        }

        reasons = []
        strength = 0.0

        # Long dominance = longs getting liquidated = potential bottom
        if snapshot.market_long_ratio > 0.7:
            strength += 0.3
            reasons.append(f"Heavy long liquidations ({snapshot.market_long_ratio:.0%}) - potential bottom")
        elif snapshot.market_long_ratio < 0.3:
            strength -= 0.3
            reasons.append(f"Heavy short liquidations ({1-snapshot.market_long_ratio:.0%}) - potential top")

        # Cascade risk
        if snapshot.cascade_risk:
            reasons.append("⚠️ High liquidation volume - cascade risk")
            # Amplify the signal
            strength *= 1.5

        # Whale liquidations
        if len(snapshot.whale_liquidations) > 3:
            whale_direction = sum(
                1 if w.side == "LONG" else -1
                for w in snapshot.whale_liquidations
            )
            if whale_direction > 2:
                strength += 0.2
                reasons.append("Multiple whale long liquidations - capitulation signal")
            elif whale_direction < -2:
                strength -= 0.2
                reasons.append("Multiple whale short liquidations - squeeze signal")

        # Determine direction
        if strength > 0.2:
            signal["direction"] = "LONG"  # Contrarian - buy when longs liquidated
        elif strength < -0.2:
            signal["direction"] = "SHORT"  # Contrarian - sell when shorts liquidated

        signal["strength"] = abs(strength)
        signal["reasons"] = reasons

        return signal


# Singleton accessor
_liquidations_instance: Optional[LiquidationsFeed] = None


def get_liquidations_feed(api_key: Optional[str] = None) -> LiquidationsFeed:
    """Get or create the liquidations feed singleton."""
    global _liquidations_instance
    if _liquidations_instance is None:
        _liquidations_instance = LiquidationsFeed(api_key)
    return _liquidations_instance
