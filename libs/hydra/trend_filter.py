"""
HYDRA Trend Filter - Prevents counter-trend trades

Analyzes recent price action to determine if a trade aligns with the trend.
Helps avoid SELL trades in uptrends and BUY trades in downtrends.

Usage:
    filter = TrendFilter()
    can_trade, reason = filter.check_trend_alignment(symbol, direction, market_data)
"""

import os
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from loguru import logger


@dataclass
class TrendFilterConfig:
    """Configuration for trend filter."""
    # Enable/disable trend filtering
    enabled: bool = True

    # Lookback periods for trend detection (in candles)
    short_period: int = 10   # ~10 minutes for 1m candles
    medium_period: int = 30  # ~30 minutes
    long_period: int = 60    # ~1 hour

    # Minimum trend strength to block counter-trend (percentage)
    min_trend_strength_pct: float = 1.5  # 1.5% move to be considered "strong trend"

    # Allow counter-trend in ranging markets
    allow_in_range: bool = True
    range_threshold_pct: float = 0.5  # Less than 0.5% = ranging

    # Require majority of timeframes to agree
    require_majority: bool = True


class TrendFilter:
    """
    Filters trades based on trend alignment.

    Prevents:
    - SELL signals when price is in a strong uptrend
    - BUY signals when price is in a strong downtrend
    """

    def __init__(self, config: Optional[TrendFilterConfig] = None):
        self.config = config or TrendFilterConfig()

        # Enable/disable via environment variable
        if os.getenv("TREND_FILTER_ENABLED", "true").lower() == "false":
            self.config.enabled = False

        logger.info(f"TrendFilter initialized (enabled: {self.config.enabled})")

    def _calculate_trend(self, prices: list) -> Tuple[str, float]:
        """
        Calculate trend direction and strength from price list.

        Args:
            prices: List of prices (oldest to newest)

        Returns:
            (direction, strength_pct): "UP", "DOWN", or "RANGE" with strength percentage
        """
        if not prices or len(prices) < 2:
            return "RANGE", 0.0

        start_price = prices[0]
        end_price = prices[-1]

        if start_price == 0:
            return "RANGE", 0.0

        change_pct = ((end_price - start_price) / start_price) * 100

        if abs(change_pct) < self.config.range_threshold_pct:
            return "RANGE", abs(change_pct)
        elif change_pct > 0:
            return "UP", change_pct
        else:
            return "DOWN", abs(change_pct)

    def check_trend_alignment(
        self,
        symbol: str,
        direction: str,
        market_data: Dict
    ) -> Tuple[bool, str]:
        """
        Check if trade direction aligns with current trend.

        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            direction: Trade direction ("BUY" or "SELL")
            market_data: Dict with price data {
                "prices": [list of recent prices],
                "current_price": float,
                "candles": [optional list of OHLCV candles]
            }

        Returns:
            (allowed, reason): Whether trade is allowed and why
        """
        if not self.config.enabled:
            return True, "Trend filter disabled"

        prices = market_data.get("prices", [])
        candles = market_data.get("candles", [])

        # Extract close prices from candles if available
        if candles and not prices:
            prices = [c.get("close", c.get("c", 0)) for c in candles]

        if not prices or len(prices) < self.config.short_period:
            logger.debug(f"[TrendFilter] Insufficient price data for {symbol}, allowing trade")
            return True, "Insufficient price data"

        # Calculate trends at different timeframes
        short_prices = prices[-self.config.short_period:]
        medium_prices = prices[-self.config.medium_period:] if len(prices) >= self.config.medium_period else prices
        long_prices = prices[-self.config.long_period:] if len(prices) >= self.config.long_period else prices

        short_trend, short_strength = self._calculate_trend(short_prices)
        medium_trend, medium_strength = self._calculate_trend(medium_prices)
        long_trend, long_strength = self._calculate_trend(long_prices)

        trends = [
            ("short", short_trend, short_strength),
            ("medium", medium_trend, medium_strength),
            ("long", long_trend, long_strength)
        ]

        # Count trend directions
        up_count = sum(1 for _, t, _ in trends if t == "UP")
        down_count = sum(1 for _, t, _ in trends if t == "DOWN")
        range_count = sum(1 for _, t, _ in trends if t == "RANGE")

        # Get strongest trend
        max_strength = max(short_strength, medium_strength, long_strength)

        # Determine dominant trend
        if range_count >= 2 or max_strength < self.config.min_trend_strength_pct:
            dominant_trend = "RANGE"
        elif up_count >= 2:
            dominant_trend = "UP"
        elif down_count >= 2:
            dominant_trend = "DOWN"
        else:
            dominant_trend = "MIXED"

        logger.debug(
            f"[TrendFilter] {symbol}: Short={short_trend}({short_strength:.1f}%), "
            f"Medium={medium_trend}({medium_strength:.1f}%), "
            f"Long={long_trend}({long_strength:.1f}%) -> Dominant={dominant_trend}"
        )

        # Check for counter-trend trades
        if direction == "BUY":
            if dominant_trend == "DOWN" and max_strength >= self.config.min_trend_strength_pct:
                return False, f"BUY blocked: Strong downtrend ({max_strength:.1f}%)"
        elif direction == "SELL":
            if dominant_trend == "UP" and max_strength >= self.config.min_trend_strength_pct:
                return False, f"SELL blocked: Strong uptrend ({max_strength:.1f}%)"

        # Allow ranging markets
        if dominant_trend == "RANGE" and self.config.allow_in_range:
            return True, f"Ranging market, trade allowed"

        # Allow trend-aligned trades
        if (direction == "BUY" and dominant_trend == "UP") or \
           (direction == "SELL" and dominant_trend == "DOWN"):
            return True, f"Trade aligned with {dominant_trend} trend"

        return True, "No strong trend detected"

    def get_recommended_direction(
        self,
        symbol: str,
        market_data: Dict
    ) -> str:
        """
        Get recommended trade direction based on trend.

        Returns:
            "BUY", "SELL", or "HOLD"
        """
        prices = market_data.get("prices", [])
        candles = market_data.get("candles", [])

        if candles and not prices:
            prices = [c.get("close", c.get("c", 0)) for c in candles]

        if not prices or len(prices) < self.config.medium_period:
            return "HOLD"

        medium_prices = prices[-self.config.medium_period:]
        trend, strength = self._calculate_trend(medium_prices)

        if trend == "UP" and strength >= self.config.min_trend_strength_pct:
            return "BUY"
        elif trend == "DOWN" and strength >= self.config.min_trend_strength_pct:
            return "SELL"
        else:
            return "HOLD"


# Global singleton
_trend_filter: Optional[TrendFilter] = None


def get_trend_filter() -> TrendFilter:
    """Get or create global trend filter singleton."""
    global _trend_filter
    if _trend_filter is None:
        _trend_filter = TrendFilter()
    return _trend_filter
