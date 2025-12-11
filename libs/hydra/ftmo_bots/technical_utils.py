"""
Technical Analysis Utilities for FTMO Bots

Provides RSI and Z-score calculations for filtered shorts feature.
These indicators help identify overbought/oversold conditions.

Usage:
    from libs.hydra.ftmo_bots.technical_utils import calculate_rsi, calculate_zscore, TechnicalIndicators

    # From candle data
    indicators = TechnicalIndicators.from_candles(candles)
    print(f"RSI: {indicators.rsi}, Z-score: {indicators.zscore}")

    # For filtered shorts check
    if indicators.is_overbought():
        # Allow short trade
        pass
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import numpy as np
from loguru import logger


@dataclass
class TechnicalIndicators:
    """Container for technical indicator values."""
    rsi: Optional[float] = None  # RSI (0-100)
    zscore: Optional[float] = None  # Price z-score (std devs from mean)
    sma_20: Optional[float] = None  # 20-period SMA
    sma_50: Optional[float] = None  # 50-period SMA
    atr: Optional[float] = None  # Average True Range
    momentum: Optional[float] = None  # Price momentum (% change)

    # Thresholds for filtered shorts
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    ZSCORE_OVERBOUGHT = 2.0
    ZSCORE_OVERSOLD = -2.0

    def is_overbought(self) -> bool:
        """Check if conditions indicate overbought (good for short)."""
        rsi_ob = self.rsi is not None and self.rsi >= self.RSI_OVERBOUGHT
        zscore_ob = self.zscore is not None and self.zscore >= self.ZSCORE_OVERBOUGHT
        return rsi_ob and zscore_ob

    def is_oversold(self) -> bool:
        """Check if conditions indicate oversold (good for long)."""
        rsi_os = self.rsi is not None and self.rsi <= self.RSI_OVERSOLD
        zscore_os = self.zscore is not None and self.zscore <= self.ZSCORE_OVERSOLD
        return rsi_os and zscore_os

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for signal inclusion."""
        return {
            "rsi": self.rsi,
            "zscore": self.zscore,
            "sma_20": self.sma_20,
            "sma_50": self.sma_50,
            "atr": self.atr,
            "momentum": self.momentum,
            "is_overbought": self.is_overbought(),
            "is_oversold": self.is_oversold(),
        }

    @classmethod
    def from_candles(cls, candles: List[Dict[str, float]], period: int = 14) -> "TechnicalIndicators":
        """
        Calculate all technical indicators from candle data.

        Args:
            candles: List of dicts with 'open', 'high', 'low', 'close' keys
            period: Lookback period for RSI and ATR (default 14)

        Returns:
            TechnicalIndicators with calculated values
        """
        if not candles or len(candles) < period + 1:
            return cls()

        try:
            closes = [c.get("close", 0) for c in candles if c.get("close")]
            highs = [c.get("high", 0) for c in candles if c.get("high")]
            lows = [c.get("low", 0) for c in candles if c.get("low")]

            if len(closes) < period + 1:
                return cls()

            # Calculate RSI
            rsi = calculate_rsi(closes, period)

            # Calculate Z-score
            zscore = calculate_zscore(closes)

            # Calculate SMAs
            sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else None
            sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else None

            # Calculate ATR
            atr = calculate_atr(candles, period)

            # Calculate momentum (% change over period)
            if len(closes) >= period:
                momentum = ((closes[-1] - closes[-period]) / closes[-period]) * 100
            else:
                momentum = None

            return cls(
                rsi=rsi,
                zscore=zscore,
                sma_20=sma_20,
                sma_50=sma_50,
                atr=atr,
                momentum=momentum,
            )

        except Exception as e:
            logger.warning(f"Error calculating technical indicators: {e}")
            return cls()


def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    """
    Calculate Relative Strength Index (RSI).

    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss

    Args:
        prices: List of closing prices (most recent last)
        period: RSI period (default 14)

    Returns:
        RSI value (0-100) or None if insufficient data
    """
    if len(prices) < period + 1:
        return None

    try:
        # Calculate price changes
        changes = np.diff(prices[-period - 1:])

        # Separate gains and losses
        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)

        # Calculate average gain and loss (simple moving average)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0  # All gains, no losses

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return round(rsi, 2)

    except Exception as e:
        logger.debug(f"RSI calculation error: {e}")
        return None


def calculate_zscore(prices: List[float], period: int = 20) -> Optional[float]:
    """
    Calculate Z-score of current price relative to recent mean.

    Z-score = (Current Price - Mean) / Standard Deviation

    Interpretation:
    - z > 2: Price is 2+ std devs above mean (overbought)
    - z < -2: Price is 2+ std devs below mean (oversold)
    - z near 0: Price is near the mean

    Args:
        prices: List of closing prices (most recent last)
        period: Lookback period for mean/std calculation (default 20)

    Returns:
        Z-score or None if insufficient data
    """
    if len(prices) < period:
        return None

    try:
        recent_prices = prices[-period:]
        current_price = prices[-1]

        mean = np.mean(recent_prices)
        std = np.std(recent_prices)

        if std == 0:
            return 0.0  # No variation

        zscore = (current_price - mean) / std
        return round(zscore, 3)

    except Exception as e:
        logger.debug(f"Z-score calculation error: {e}")
        return None


def calculate_atr(candles: List[Dict[str, float]], period: int = 14) -> Optional[float]:
    """
    Calculate Average True Range (ATR).

    True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    ATR = SMA of True Range over period

    Args:
        candles: List of candle dicts with 'high', 'low', 'close' keys
        period: ATR period (default 14)

    Returns:
        ATR value or None if insufficient data
    """
    if len(candles) < period + 1:
        return None

    try:
        true_ranges = []
        for i in range(1, len(candles)):
            high = candles[i].get("high", 0)
            low = candles[i].get("low", 0)
            prev_close = candles[i - 1].get("close", 0)

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)

        # Calculate ATR as simple moving average of last 'period' true ranges
        atr = np.mean(true_ranges[-period:])
        return round(atr, 4)

    except Exception as e:
        logger.debug(f"ATR calculation error: {e}")
        return None


def calculate_bollinger_bands(
    prices: List[float],
    period: int = 20,
    std_dev: float = 2.0
) -> Optional[Dict[str, float]]:
    """
    Calculate Bollinger Bands.

    Args:
        prices: List of closing prices
        period: SMA period (default 20)
        std_dev: Standard deviation multiplier (default 2)

    Returns:
        Dict with 'upper', 'middle', 'lower' bands or None
    """
    if len(prices) < period:
        return None

    try:
        recent = prices[-period:]
        middle = np.mean(recent)
        std = np.std(recent)

        return {
            "upper": middle + (std * std_dev),
            "middle": middle,
            "lower": middle - (std * std_dev),
        }

    except Exception as e:
        logger.debug(f"Bollinger Bands calculation error: {e}")
        return None


def get_trend_direction(prices: List[float], short_period: int = 10, long_period: int = 30) -> str:
    """
    Determine trend direction using dual moving averages.

    Args:
        prices: List of closing prices
        short_period: Short MA period
        long_period: Long MA period

    Returns:
        "BULLISH", "BEARISH", or "NEUTRAL"
    """
    if len(prices) < long_period:
        return "NEUTRAL"

    try:
        short_ma = np.mean(prices[-short_period:])
        long_ma = np.mean(prices[-long_period:])
        current = prices[-1]

        # Strong trend: price and short MA above/below long MA
        if current > short_ma > long_ma:
            return "BULLISH"
        elif current < short_ma < long_ma:
            return "BEARISH"
        else:
            return "NEUTRAL"

    except Exception:
        return "NEUTRAL"


# Convenience function for FTMO bots
def get_market_indicators(candles: List[Dict[str, float]]) -> TechnicalIndicators:
    """
    Get technical indicators for market analysis.

    This is the main entry point for FTMO bots to get RSI/z-score
    for filtered shorts functionality.

    Args:
        candles: List of candle dicts from market_data

    Returns:
        TechnicalIndicators with all calculated values
    """
    return TechnicalIndicators.from_candles(candles)
