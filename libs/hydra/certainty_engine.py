"""
Certainty Engine - Multi-Factor Confidence Scoring

Calculates trade certainty based on 4 weighted factors:
- Technical Confluence (40%): MTF alignment, RSI, MACD, EMA stack
- Market Structure (30%): S/R levels, trend direction
- Sentiment/Order Flow (20%): Volume, momentum
- Time Factors (10%): Session quality, day of week

Only trades with >= 80% certainty score should be taken.

Usage:
    from libs.hydra.certainty_engine import get_certainty_engine

    engine = get_certainty_engine()
    result = engine.calculate_certainty(
        symbol="BTC-USD",
        direction="BUY",
        candles=market_data["candles"],
        current_price=97500.0,
    )

    if result.should_trade:
        print(f"Take trade: {result.total_score:.0%} certainty")
    else:
        print(f"Skip trade: {result.total_score:.0%} < 80% threshold")
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

from loguru import logger


# ==================== DATA CLASSES ====================

@dataclass
class CertaintyFactors:
    """Individual factor scores (0.0 - 1.0)."""
    technical_confluence: float = 0.0  # 40% weight
    market_structure: float = 0.0      # 30% weight
    sentiment_order_flow: float = 0.0  # 20% weight
    time_factors: float = 0.0          # 10% weight

    # Sub-component details for debugging
    technical_details: Dict[str, float] = field(default_factory=dict)
    structure_details: Dict[str, float] = field(default_factory=dict)
    sentiment_details: Dict[str, float] = field(default_factory=dict)
    time_details: Dict[str, float] = field(default_factory=dict)


@dataclass
class CertaintyResult:
    """Final certainty calculation result."""
    total_score: float               # 0.0 - 1.0 weighted score
    should_trade: bool               # >= CERTAINTY_THRESHOLD
    factors: CertaintyFactors
    reason: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ==================== MAIN CLASS ====================

class CertaintyEngine:
    """
    Multi-factor certainty scoring for trade signals.

    Calculates confidence based on:
    1. Technical Confluence (40%): Multi-timeframe alignment, indicator convergence
    2. Market Structure (30%): S/R levels, trend direction, HH/HL patterns
    3. Sentiment/Order Flow (20%): Volume profile, momentum
    4. Time Factors (10%): Session quality, day of week

    A trade should only be taken if certainty >= 80%.
    """

    # Configuration
    CERTAINTY_THRESHOLD = 0.80  # 80% required to trade

    WEIGHTS = {
        "technical": 0.40,   # 40%
        "structure": 0.30,   # 30%
        "sentiment": 0.20,   # 20%
        "time": 0.10,        # 10%
    }

    def __init__(self):
        """Initialize Certainty Engine."""
        self._lock = threading.Lock()
        self._history: deque = deque(maxlen=1000)  # Track recent calculations

        logger.info("[CertaintyEngine] Initialized")

    # ==================== PUBLIC METHODS ====================

    def calculate_certainty(
        self,
        symbol: str,
        direction: str,
        candles: List[Dict],
        current_price: float,
        session_data: Optional[Dict] = None,
        order_flow: Optional[Dict] = None,
    ) -> CertaintyResult:
        """
        Calculate multi-factor certainty score for a trade signal.

        Args:
            symbol: Trading symbol (BTC-USD, XAUUSD, etc.)
            direction: Trade direction (BUY, SELL)
            candles: List of OHLCV candles
            current_price: Current market price
            session_data: Optional session info
            order_flow: Optional order flow data

        Returns:
            CertaintyResult with score and recommendation
        """
        direction = direction.upper()

        # Calculate each factor
        tech_score, tech_details = self._calc_technical_confluence(
            symbol, direction, candles, current_price
        )
        struct_score, struct_details = self._calc_market_structure(
            symbol, direction, candles, current_price
        )
        sent_score, sent_details = self._calc_sentiment_orderflow(
            symbol, direction, candles, order_flow
        )
        time_score, time_details = self._calc_time_factors(
            symbol, session_data
        )

        # Create factors object
        factors = CertaintyFactors(
            technical_confluence=tech_score,
            market_structure=struct_score,
            sentiment_order_flow=sent_score,
            time_factors=time_score,
            technical_details=tech_details,
            structure_details=struct_details,
            sentiment_details=sent_details,
            time_details=time_details,
        )

        # Calculate weighted total
        total_score = (
            self.WEIGHTS["technical"] * tech_score +
            self.WEIGHTS["structure"] * struct_score +
            self.WEIGHTS["sentiment"] * sent_score +
            self.WEIGHTS["time"] * time_score
        )

        # Determine if should trade
        should_trade = total_score >= self.CERTAINTY_THRESHOLD

        # Build reason string
        if should_trade:
            reason = f"Certainty {total_score:.0%} >= {self.CERTAINTY_THRESHOLD:.0%} threshold"
        else:
            # Find lowest factor
            factor_scores = [
                ("technical", tech_score),
                ("structure", struct_score),
                ("sentiment", sent_score),
                ("time", time_score),
            ]
            lowest = min(factor_scores, key=lambda x: x[1])
            reason = f"Certainty {total_score:.0%} < {self.CERTAINTY_THRESHOLD:.0%} (lowest: {lowest[0]}={lowest[1]:.0%})"

        result = CertaintyResult(
            total_score=total_score,
            should_trade=should_trade,
            factors=factors,
            reason=reason,
        )

        # Store in history
        with self._lock:
            self._history.append({
                "symbol": symbol,
                "direction": direction,
                "score": total_score,
                "should_trade": should_trade,
                "timestamp": result.timestamp.isoformat(),
            })

        logger.debug(
            f"[CertaintyEngine] {symbol} {direction}: {total_score:.0%} "
            f"(tech={tech_score:.0%}, struct={struct_score:.0%}, "
            f"sent={sent_score:.0%}, time={time_score:.0%}) → {should_trade}"
        )

        return result

    # ==================== FACTOR CALCULATIONS ====================

    def _calc_technical_confluence(
        self,
        symbol: str,
        direction: str,
        candles: List[Dict],
        current_price: float,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Technical Confluence (40% weight):
        - Multi-timeframe alignment (M1, M5, M15, H1 trends)
        - Indicator convergence (RSI, MACD, EMA alignment)
        - Volume confirmation
        """
        details = {}
        scores = []

        if not candles or len(candles) < 20:
            return 0.5, {"error": "insufficient_data"}

        try:
            closes = [c.get("close", 0) for c in candles if c.get("close")]
            highs = [c.get("high", 0) for c in candles if c.get("high")]
            lows = [c.get("low", 0) for c in candles if c.get("low")]
            volumes = [c.get("volume", 0) for c in candles if c.get("volume")]

            if len(closes) < 20:
                return 0.5, {"error": "insufficient_closes"}

            # 1. RSI Check (weight: 25%)
            rsi = self._calculate_rsi(closes, 14)
            if direction == "BUY":
                # For BUY: RSI should not be overbought, ideally 30-60
                if rsi < 30:
                    rsi_score = 0.9  # Oversold - good for buy
                elif rsi < 50:
                    rsi_score = 0.8  # Below mid - decent
                elif rsi < 70:
                    rsi_score = 0.6  # Above mid - caution
                else:
                    rsi_score = 0.3  # Overbought - bad for buy
            else:  # SELL
                # For SELL: RSI should not be oversold, ideally 40-70
                if rsi > 70:
                    rsi_score = 0.9  # Overbought - good for sell
                elif rsi > 50:
                    rsi_score = 0.8  # Above mid - decent
                elif rsi > 30:
                    rsi_score = 0.6  # Below mid - caution
                else:
                    rsi_score = 0.3  # Oversold - bad for sell

            details["rsi"] = rsi_score
            details["rsi_value"] = rsi
            scores.append(rsi_score)

            # 2. EMA Alignment (weight: 25%)
            ema_20 = self._calculate_ema(closes, 20)
            ema_50 = self._calculate_ema(closes, 50) if len(closes) >= 50 else ema_20

            if direction == "BUY":
                # Price above EMAs and EMAs aligned bullish
                if current_price > ema_20 > ema_50:
                    ema_score = 0.9  # Perfect bullish alignment
                elif current_price > ema_20:
                    ema_score = 0.7  # Price above short EMA
                elif current_price > ema_50:
                    ema_score = 0.5  # Price above long EMA only
                else:
                    ema_score = 0.3  # Price below both
            else:  # SELL
                if current_price < ema_20 < ema_50:
                    ema_score = 0.9  # Perfect bearish alignment
                elif current_price < ema_20:
                    ema_score = 0.7
                elif current_price < ema_50:
                    ema_score = 0.5
                else:
                    ema_score = 0.3

            details["ema_alignment"] = ema_score
            scores.append(ema_score)

            # 3. MACD Momentum (weight: 25%)
            macd_line, signal_line, histogram = self._calculate_macd(closes)
            if direction == "BUY":
                if histogram > 0 and macd_line > signal_line:
                    macd_score = 0.9  # Bullish crossover confirmed
                elif macd_line > signal_line:
                    macd_score = 0.7  # Above signal
                elif histogram > 0:
                    macd_score = 0.5  # Histogram positive
                else:
                    macd_score = 0.3  # Bearish
            else:  # SELL
                if histogram < 0 and macd_line < signal_line:
                    macd_score = 0.9
                elif macd_line < signal_line:
                    macd_score = 0.7
                elif histogram < 0:
                    macd_score = 0.5
                else:
                    macd_score = 0.3

            details["macd"] = macd_score
            scores.append(macd_score)

            # 4. Short-term Momentum (weight: 25%)
            # Price change over last 5 candles
            if len(closes) >= 5:
                momentum = (closes[-1] - closes[-5]) / closes[-5] if closes[-5] > 0 else 0

                if direction == "BUY":
                    if momentum > 0.01:  # +1%
                        mom_score = 0.9
                    elif momentum > 0:
                        mom_score = 0.7
                    elif momentum > -0.01:
                        mom_score = 0.5
                    else:
                        mom_score = 0.3
                else:  # SELL
                    if momentum < -0.01:
                        mom_score = 0.9
                    elif momentum < 0:
                        mom_score = 0.7
                    elif momentum < 0.01:
                        mom_score = 0.5
                    else:
                        mom_score = 0.3

                details["momentum"] = mom_score
                scores.append(mom_score)

            # Average all scores
            final_score = sum(scores) / len(scores) if scores else 0.5
            return final_score, details

        except Exception as e:
            logger.warning(f"[CertaintyEngine] Technical calc error: {e}")
            return 0.5, {"error": str(e)}

    def _calc_market_structure(
        self,
        symbol: str,
        direction: str,
        candles: List[Dict],
        current_price: float,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Market Structure (30% weight):
        - S/R level proximity and quality
        - Trend direction and strength
        - Higher-high/higher-low pattern (for longs)
        - Lower-high/lower-low pattern (for shorts)
        """
        details = {}
        scores = []

        if not candles or len(candles) < 20:
            return 0.5, {"error": "insufficient_data"}

        try:
            closes = [c.get("close", 0) for c in candles if c.get("close")]
            highs = [c.get("high", 0) for c in candles if c.get("high")]
            lows = [c.get("low", 0) for c in candles if c.get("low")]

            if len(closes) < 20:
                return 0.5, {"error": "insufficient_data"}

            # 1. Trend Direction (weight: 40%)
            sma_10 = sum(closes[-10:]) / 10
            sma_20 = sum(closes[-20:]) / 20

            trend_slope = (sma_10 - sma_20) / sma_20 if sma_20 > 0 else 0

            if direction == "BUY":
                if trend_slope > 0.02:  # Strong uptrend (>2%)
                    trend_score = 0.95
                elif trend_slope > 0.01:  # Moderate uptrend
                    trend_score = 0.8
                elif trend_slope > 0:  # Weak uptrend
                    trend_score = 0.6
                elif trend_slope > -0.01:  # Slight downtrend (range)
                    trend_score = 0.5  # Neutral
                else:
                    trend_score = 0.3  # Against trend
            else:  # SELL
                if trend_slope < -0.02:
                    trend_score = 0.95
                elif trend_slope < -0.01:
                    trend_score = 0.8
                elif trend_slope < 0:
                    trend_score = 0.6
                elif trend_slope < 0.01:
                    trend_score = 0.5
                else:
                    trend_score = 0.3

            details["trend"] = trend_score
            details["trend_slope"] = trend_slope
            scores.append(trend_score)

            # 2. Higher Highs / Lower Lows Pattern (weight: 30%)
            # Look at last 4 swing points
            recent_highs = highs[-20:]
            recent_lows = lows[-20:]

            # Simple check: compare recent halves
            first_half_high = max(recent_highs[:10])
            second_half_high = max(recent_highs[10:])
            first_half_low = min(recent_lows[:10])
            second_half_low = min(recent_lows[10:])

            if direction == "BUY":
                hh = second_half_high > first_half_high  # Higher high
                hl = second_half_low > first_half_low    # Higher low

                if hh and hl:
                    pattern_score = 0.9  # Perfect bullish structure
                elif hl:
                    pattern_score = 0.7  # Higher lows (accumulation)
                elif hh:
                    pattern_score = 0.6  # New highs but weak lows
                else:
                    pattern_score = 0.4  # Bearish structure
            else:  # SELL
                lh = second_half_high < first_half_high  # Lower high
                ll = second_half_low < first_half_low    # Lower low

                if lh and ll:
                    pattern_score = 0.9
                elif lh:
                    pattern_score = 0.7
                elif ll:
                    pattern_score = 0.6
                else:
                    pattern_score = 0.4

            details["structure_pattern"] = pattern_score
            scores.append(pattern_score)

            # 3. S/R Level Proximity (weight: 30%)
            # Check if near recent S/R levels
            recent_high = max(highs[-10:])
            recent_low = min(lows[-10:])
            range_size = recent_high - recent_low

            if range_size > 0:
                position_in_range = (current_price - recent_low) / range_size

                if direction == "BUY":
                    # For buys: better near support (low in range)
                    if position_in_range < 0.3:
                        sr_score = 0.9  # Near support
                    elif position_in_range < 0.5:
                        sr_score = 0.7  # Lower half
                    elif position_in_range < 0.7:
                        sr_score = 0.5  # Mid range
                    else:
                        sr_score = 0.3  # Near resistance
                else:  # SELL
                    if position_in_range > 0.7:
                        sr_score = 0.9  # Near resistance
                    elif position_in_range > 0.5:
                        sr_score = 0.7
                    elif position_in_range > 0.3:
                        sr_score = 0.5
                    else:
                        sr_score = 0.3
            else:
                sr_score = 0.5

            details["sr_proximity"] = sr_score
            scores.append(sr_score)

            final_score = sum(scores) / len(scores) if scores else 0.5
            return final_score, details

        except Exception as e:
            logger.warning(f"[CertaintyEngine] Structure calc error: {e}")
            return 0.5, {"error": str(e)}

    def _calc_sentiment_orderflow(
        self,
        symbol: str,
        direction: str,
        candles: List[Dict],
        order_flow: Optional[Dict],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Sentiment/Order Flow (20% weight):
        - Volume profile analysis
        - Momentum divergence check
        - Large order detection (if available)
        """
        details = {}
        scores = []

        if not candles or len(candles) < 10:
            return 0.5, {"error": "insufficient_data"}

        try:
            closes = [c.get("close", 0) for c in candles if c.get("close")]
            volumes = [c.get("volume", 0) for c in candles if c.get("volume")]

            # 1. Volume Trend (weight: 50%)
            if volumes and len(volumes) >= 10:
                avg_vol_early = sum(volumes[:5]) / 5
                avg_vol_recent = sum(volumes[-5:]) / 5

                if avg_vol_early > 0:
                    vol_change = (avg_vol_recent - avg_vol_early) / avg_vol_early
                else:
                    vol_change = 0

                # Increasing volume is generally positive
                if vol_change > 0.5:  # 50% increase
                    vol_score = 0.9
                elif vol_change > 0.2:
                    vol_score = 0.7
                elif vol_change > -0.2:
                    vol_score = 0.5
                else:
                    vol_score = 0.3  # Decreasing volume

                details["volume_trend"] = vol_score
                scores.append(vol_score)

            # 2. Price-Volume Alignment (weight: 50%)
            # Check if volume supports price direction
            if len(closes) >= 5 and len(volumes) >= 5:
                price_change = closes[-1] - closes[-5]
                recent_vol = sum(volumes[-5:])

                # Calculate average volume per candle
                avg_candle_vol = recent_vol / 5 if recent_vol > 0 else 1

                # Volume-weighted price change
                if direction == "BUY":
                    if price_change > 0 and recent_vol > avg_candle_vol:
                        pv_score = 0.9  # Price up with volume
                    elif price_change > 0:
                        pv_score = 0.6  # Price up, weak volume
                    elif recent_vol < avg_candle_vol * 0.5:
                        pv_score = 0.5  # Low volume pullback (bullish)
                    else:
                        pv_score = 0.3  # Price down with volume (bearish)
                else:  # SELL
                    if price_change < 0 and recent_vol > avg_candle_vol:
                        pv_score = 0.9
                    elif price_change < 0:
                        pv_score = 0.6
                    elif recent_vol < avg_candle_vol * 0.5:
                        pv_score = 0.5
                    else:
                        pv_score = 0.3

                details["price_volume"] = pv_score
                scores.append(pv_score)

            final_score = sum(scores) / len(scores) if scores else 0.5
            return final_score, details

        except Exception as e:
            logger.warning(f"[CertaintyEngine] Sentiment calc error: {e}")
            return 0.5, {"error": str(e)}

    def _calc_time_factors(
        self,
        symbol: str,
        session_data: Optional[Dict],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Time Factors (10% weight):
        - Session quality (London, NY, overlap)
        - Day of week edge
        - News avoidance
        """
        details = {}
        scores = []

        try:
            now = datetime.now(timezone.utc)
            hour = now.hour
            day = now.weekday()  # 0=Monday, 6=Sunday

            # 1. Session Quality (weight: 60%)
            # Best sessions: London (08-16 UTC), NY (13-21 UTC), Overlap (13-17 UTC)
            if 13 <= hour < 17:
                session_score = 0.95  # London/NY overlap - best (13:00-17:00 UTC)
            elif 8 <= hour < 13:
                session_score = 0.8   # London morning (08:00-13:00 UTC)
            elif 17 <= hour < 21:
                session_score = 0.75  # NY session post-overlap (17:00-21:00 UTC)
            elif 0 <= hour < 8:
                session_score = 0.5   # Asian session (00:00-08:00 UTC)
            else:
                session_score = 0.4   # Off-hours (21:00-00:00 UTC)

            details["session"] = session_score
            scores.append(session_score)

            # 2. Day of Week (weight: 40%)
            if day == 4:  # Friday
                day_score = 0.5   # Lower due to weekend risk
            elif day in [5, 6]:  # Weekend
                day_score = 0.3   # Avoid if possible
            elif day == 0:  # Monday
                day_score = 0.7   # Monday gaps
            else:  # Tue-Thu
                day_score = 0.85  # Best days

            details["day_of_week"] = day_score
            scores.append(day_score)

            final_score = sum(scores) / len(scores) if scores else 0.5
            return final_score, details

        except Exception as e:
            logger.warning(f"[CertaintyEngine] Time calc error: {e}")
            return 0.5, {"error": str(e)}

    # ==================== TECHNICAL INDICATOR HELPERS ====================

    def _calculate_rsi(self, closes: List[float], period: int = 14) -> float:
        """Calculate RSI."""
        if len(closes) < period + 1:
            return 50.0  # Default neutral

        gains = []
        losses = []

        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        if len(gains) < period:
            return 50.0

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_ema(self, closes: List[float], period: int) -> float:
        """Calculate EMA."""
        if len(closes) < period:
            return sum(closes) / len(closes) if closes else 0

        multiplier = 2 / (period + 1)
        ema = sum(closes[:period]) / period

        for price in closes[period:]:
            ema = (price - ema) * multiplier + ema

        return ema

    def _calculate_macd(
        self,
        closes: List[float],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[float, float, float]:
        """Calculate MACD line, signal line, and histogram."""
        if len(closes) < slow:
            return 0, 0, 0

        ema_fast = self._calculate_ema(closes, fast)
        ema_slow = self._calculate_ema(closes, slow)

        macd_line = ema_fast - ema_slow

        # Signal line (EMA of MACD)
        # Simplified: use current MACD as signal approximation
        signal_line = macd_line * 0.9  # Lag approximation

        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    # ==================== UTILITY METHODS ====================

    def get_recent_calculations(self, limit: int = 50) -> List[Dict]:
        """Get recent certainty calculations."""
        with self._lock:
            return list(self._history)[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about certainty calculations."""
        with self._lock:
            if not self._history:
                return {"total_calculations": 0}

            passed = sum(1 for h in self._history if h["should_trade"])
            failed = len(self._history) - passed

            return {
                "total_calculations": len(self._history),
                "passed": passed,
                "failed": failed,
                "pass_rate": passed / len(self._history) if self._history else 0,
                "threshold": self.CERTAINTY_THRESHOLD,
            }


# ==================== SINGLETON ACCESSOR ====================

_instance: Optional[CertaintyEngine] = None
_instance_lock = threading.Lock()


def get_certainty_engine() -> CertaintyEngine:
    """
    Get the singleton instance of CertaintyEngine.

    Returns:
        CertaintyEngine instance
    """
    global _instance

    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = CertaintyEngine()

    return _instance


def reset_certainty_engine() -> None:
    """Reset the singleton instance (for testing)."""
    global _instance
    with _instance_lock:
        _instance = None


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Example usage
    engine = get_certainty_engine()

    # Simulate some candle data
    import random
    base_price = 97000
    candles = []
    for i in range(50):
        o = base_price + random.uniform(-500, 500)
        h = o + random.uniform(0, 300)
        l = o - random.uniform(0, 300)
        c = o + random.uniform(-200, 200)
        v = random.uniform(100, 1000)
        candles.append({
            "open": o, "high": h, "low": l, "close": c, "volume": v
        })
        base_price = c  # Trend up slightly

    result = engine.calculate_certainty(
        symbol="BTC-USD",
        direction="BUY",
        candles=candles,
        current_price=base_price,
    )

    print(f"Certainty Score: {result.total_score:.0%}")
    print(f"Should Trade: {result.should_trade}")
    print(f"Reason: {result.reason}")
    print(f"\nFactors:")
    print(f"  Technical: {result.factors.technical_confluence:.0%}")
    print(f"  Structure: {result.factors.market_structure:.0%}")
    print(f"  Sentiment: {result.factors.sentiment_order_flow:.0%}")
    print(f"  Time: {result.factors.time_factors:.0%}")
