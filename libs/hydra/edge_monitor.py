"""
Edge Monitor - Real-time edge tracking with auto-exit.

This module monitors active trades and checks if the original edge
(the reason for entering) is still valid. If too many factors
invalidate, we exit early to protect profits.

5 Edge Factors:
1. Price Action - HH/HL for longs, LH/LL for shorts
2. Momentum - RSI, MACD direction supporting trade
3. Order Flow - Volume confirming direction
4. Correlation - Correlated assets confirming move
5. Time - Still in favorable session

Rules:
- MIN_FACTORS_TO_HOLD = 2 (exit if <2/5 valid)
- MIN_FACTORS_TO_ADD = 4 (only pyramid if 4+/5 valid)
"""

import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class EdgeAction(Enum):
    """Action to take based on edge state."""
    HOLD = "hold"           # Edge intact, continue holding
    EXIT = "exit"           # Edge lost, exit position
    ADD_OK = "add_ok"       # Strong edge, safe to pyramid


@dataclass
class EdgeFactor:
    """Individual edge factor result."""
    name: str
    valid: bool
    score: float  # 0.0 to 1.0
    reason: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeState:
    """Complete edge evaluation state."""
    trade_id: str
    timestamp: datetime
    factors: List[EdgeFactor]
    valid_count: int
    total_count: int
    action: EdgeAction
    reason: str

    @property
    def edge_score(self) -> float:
        """Overall edge score as percentage."""
        if self.total_count == 0:
            return 0.0
        return self.valid_count / self.total_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "timestamp": self.timestamp.isoformat(),
            "factors": [
                {
                    "name": f.name,
                    "valid": f.valid,
                    "score": f.score,
                    "reason": f.reason,
                    "data": f.data,
                }
                for f in self.factors
            ],
            "valid_count": self.valid_count,
            "total_count": self.total_count,
            "edge_score": self.edge_score,
            "action": self.action.value,
            "reason": self.reason,
        }


@dataclass
class TradeContext:
    """Context needed to evaluate edge for a trade."""
    trade_id: str
    symbol: str
    direction: str  # "BUY" or "SELL"
    entry_price: float
    current_price: float
    entry_time: datetime
    # Technical data
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    ema_short: Optional[float] = None
    ema_long: Optional[float] = None
    # Structure
    recent_highs: List[float] = field(default_factory=list)
    recent_lows: List[float] = field(default_factory=list)
    # Volume
    current_volume: Optional[float] = None
    avg_volume: Optional[float] = None
    # Time
    session: Optional[str] = None


class EdgeMonitor:
    """
    Real-time edge tracking with auto-exit recommendations.

    Monitors 5 edge factors and recommends exit when edge is lost.
    """

    # Configuration
    MIN_FACTORS_TO_HOLD = 2   # Exit if <2/5 factors valid
    MIN_FACTORS_TO_ADD = 4    # Only pyramid if 4+/5 valid

    # Consecutive checks required before action
    CONSECUTIVE_CHECKS_FOR_EXIT = 2  # Prevent whipsaw

    def __init__(self):
        self._lock = threading.Lock()
        # Track consecutive invalid checks per trade
        self._consecutive_invalid: Dict[str, int] = {}
        # Track last edge state per trade
        self._last_states: Dict[str, EdgeState] = {}

        # Load config from environment
        self.MIN_FACTORS_TO_HOLD = int(
            os.environ.get("EDGE_MIN_FACTORS_HOLD", "2")
        )
        self.MIN_FACTORS_TO_ADD = int(
            os.environ.get("EDGE_MIN_FACTORS_ADD", "4")
        )
        self.CONSECUTIVE_CHECKS_FOR_EXIT = int(
            os.environ.get("EDGE_CONSECUTIVE_CHECKS", "2")
        )

        logger.info(
            f"[EdgeMonitor] Initialized: hold={self.MIN_FACTORS_TO_HOLD}/5, "
            f"add={self.MIN_FACTORS_TO_ADD}/5, checks_for_exit={self.CONSECUTIVE_CHECKS_FOR_EXIT}"
        )

    def check_edge(
        self,
        context: TradeContext,
        candles: Optional[List[Dict]] = None,
    ) -> EdgeState:
        """
        Check if edge is still valid for a trade.

        Args:
            context: Trade context with current market data
            candles: Optional list of recent candles for analysis

        Returns:
            EdgeState with recommended action
        """
        factors = []

        # 1. Price Action
        pa_factor = self._check_price_action(context, candles)
        factors.append(pa_factor)

        # 2. Momentum
        mom_factor = self._check_momentum(context)
        factors.append(mom_factor)

        # 3. Order Flow (Volume)
        vol_factor = self._check_order_flow(context, candles)
        factors.append(vol_factor)

        # 4. Correlation (simplified - check direction consistency)
        corr_factor = self._check_correlation(context)
        factors.append(corr_factor)

        # 5. Time
        time_factor = self._check_time(context)
        factors.append(time_factor)

        # Count valid factors
        valid_count = sum(1 for f in factors if f.valid)
        total_count = len(factors)

        # Determine action
        action, reason = self._determine_action(
            context.trade_id, valid_count, total_count
        )

        state = EdgeState(
            trade_id=context.trade_id,
            timestamp=datetime.now(timezone.utc),
            factors=factors,
            valid_count=valid_count,
            total_count=total_count,
            action=action,
            reason=reason,
        )

        # Store state
        with self._lock:
            self._last_states[context.trade_id] = state

        # Log significant events
        if action == EdgeAction.EXIT:
            logger.warning(
                f"[EdgeMonitor] EXIT recommended for {context.trade_id}: "
                f"{valid_count}/{total_count} factors valid - {reason}"
            )
        elif action == EdgeAction.ADD_OK:
            logger.info(
                f"[EdgeMonitor] ADD_OK for {context.trade_id}: "
                f"{valid_count}/{total_count} factors valid"
            )

        return state

    def _check_price_action(
        self,
        context: TradeContext,
        candles: Optional[List[Dict]] = None,
    ) -> EdgeFactor:
        """
        Check if price action supports the trade direction.

        For LONG: Want HH (Higher Highs) and HL (Higher Lows)
        For SHORT: Want LH (Lower Highs) and LL (Lower Lows)
        """
        name = "price_action"

        # Need candles or recent highs/lows
        if not candles and not (context.recent_highs and context.recent_lows):
            return EdgeFactor(
                name=name,
                valid=True,  # Assume valid if no data
                score=0.5,
                reason="Insufficient data",
                data={"has_data": False},
            )

        # Extract highs and lows from candles if not provided
        highs = context.recent_highs
        lows = context.recent_lows

        if candles and len(candles) >= 3:
            highs = [c.get("high", c.get("h", 0)) for c in candles[-5:]]
            lows = [c.get("low", c.get("l", 0)) for c in candles[-5:]]

        if len(highs) < 2 or len(lows) < 2:
            return EdgeFactor(
                name=name,
                valid=True,
                score=0.5,
                reason="Not enough data points",
            )

        is_long = context.direction.upper() == "BUY"

        # Check structure
        if is_long:
            # Want HH and HL
            hh = highs[-1] >= highs[-2]  # Higher high
            hl = lows[-1] >= lows[-2]    # Higher low

            valid = hh or hl  # At least one bullish structure
            score = (0.5 if hh else 0.0) + (0.5 if hl else 0.0)

            if valid:
                reason = f"Bullish structure: HH={hh}, HL={hl}"
            else:
                reason = "Bearish structure detected (LH+LL)"
        else:
            # Want LH and LL for shorts
            lh = highs[-1] <= highs[-2]  # Lower high
            ll = lows[-1] <= lows[-2]    # Lower low

            valid = lh or ll  # At least one bearish structure
            score = (0.5 if lh else 0.0) + (0.5 if ll else 0.0)

            if valid:
                reason = f"Bearish structure: LH={lh}, LL={ll}"
            else:
                reason = "Bullish structure detected (HH+HL)"

        return EdgeFactor(
            name=name,
            valid=valid,
            score=score,
            reason=reason,
            data={"highs": highs[-3:], "lows": lows[-3:]},
        )

    def _check_momentum(self, context: TradeContext) -> EdgeFactor:
        """
        Check if momentum indicators support the trade.

        Uses RSI and MACD direction.
        """
        name = "momentum"

        # Need at least RSI
        if context.rsi is None:
            return EdgeFactor(
                name=name,
                valid=True,
                score=0.5,
                reason="No RSI data",
            )

        is_long = context.direction.upper() == "BUY"
        score = 0.0
        reasons = []

        # RSI check
        if is_long:
            # For longs, RSI should not be overbought (>70)
            # Ideal: 30-65 range
            if context.rsi < 30:
                score += 0.3
                reasons.append(f"RSI oversold ({context.rsi:.1f})")
            elif context.rsi < 70:
                score += 0.5
                reasons.append(f"RSI healthy ({context.rsi:.1f})")
            else:
                reasons.append(f"RSI overbought ({context.rsi:.1f})")
        else:
            # For shorts, RSI should not be oversold (<30)
            if context.rsi > 70:
                score += 0.3
                reasons.append(f"RSI overbought ({context.rsi:.1f})")
            elif context.rsi > 30:
                score += 0.5
                reasons.append(f"RSI healthy ({context.rsi:.1f})")
            else:
                reasons.append(f"RSI oversold ({context.rsi:.1f})")

        # MACD check
        if context.macd is not None and context.macd_signal is not None:
            macd_bullish = context.macd > context.macd_signal

            if (is_long and macd_bullish) or (not is_long and not macd_bullish):
                score += 0.5
                reasons.append("MACD confirming")
            else:
                reasons.append("MACD diverging")

        # Normalize score
        score = min(score, 1.0)
        valid = score >= 0.4  # Need at least 40% momentum alignment

        return EdgeFactor(
            name=name,
            valid=valid,
            score=score,
            reason=", ".join(reasons) if reasons else "Unknown",
            data={"rsi": context.rsi, "macd": context.macd},
        )

    def _check_order_flow(
        self,
        context: TradeContext,
        candles: Optional[List[Dict]] = None,
    ) -> EdgeFactor:
        """
        Check if volume/order flow supports the trade.

        For breakouts/continuations, want above-average volume.
        """
        name = "order_flow"

        # Check provided volume first
        if context.current_volume and context.avg_volume:
            vol_ratio = context.current_volume / context.avg_volume
            valid = vol_ratio > 0.8  # At least 80% of average
            score = min(vol_ratio / 1.5, 1.0)  # Cap at 1.5x for max score

            return EdgeFactor(
                name=name,
                valid=valid,
                score=score,
                reason=f"Volume ratio: {vol_ratio:.2f}x avg",
                data={"ratio": vol_ratio},
            )

        # Try to calculate from candles
        if candles and len(candles) >= 5:
            volumes = [c.get("volume", c.get("v", 0)) for c in candles if c.get("volume", c.get("v", 0)) > 0]

            if len(volumes) >= 3:
                current_vol = volumes[-1]
                avg_vol = sum(volumes[:-1]) / len(volumes[:-1])

                if avg_vol > 0:
                    vol_ratio = current_vol / avg_vol
                    valid = vol_ratio > 0.8
                    score = min(vol_ratio / 1.5, 1.0)

                    return EdgeFactor(
                        name=name,
                        valid=valid,
                        score=score,
                        reason=f"Volume ratio: {vol_ratio:.2f}x avg",
                        data={"current": current_vol, "avg": avg_vol, "ratio": vol_ratio},
                    )

        # No volume data - assume neutral
        return EdgeFactor(
            name=name,
            valid=True,
            score=0.5,
            reason="No volume data",
        )

    def _check_correlation(self, context: TradeContext) -> EdgeFactor:
        """
        Check if correlated assets confirm the move.

        Simplified: Check if price is moving in expected direction.
        Full implementation would check BTC for alts, DXY for forex, etc.
        """
        name = "correlation"

        is_long = context.direction.upper() == "BUY"

        # Simple check: Is price moving in our direction?
        if context.entry_price and context.entry_price > 0 and context.current_price:
            pnl_pct = (context.current_price - context.entry_price) / context.entry_price * 100

            if not is_long:
                pnl_pct = -pnl_pct  # Invert for shorts

            # Price moving in our direction?
            if pnl_pct > 0:
                score = min(0.5 + pnl_pct / 2, 1.0)  # Up to 1.0 for +1%
                valid = True
                reason = f"Price confirming (+{pnl_pct:.2f}%)"
            elif pnl_pct > -0.5:
                score = 0.5
                valid = True
                reason = f"Price neutral ({pnl_pct:.2f}%)"
            else:
                score = max(0.5 + pnl_pct / 2, 0.0)  # Down to 0 for -1%
                valid = pnl_pct > -1.0  # Invalid if down more than 1%
                reason = f"Price against ({pnl_pct:.2f}%)"

            return EdgeFactor(
                name=name,
                valid=valid,
                score=score,
                reason=reason,
                data={"pnl_pct": pnl_pct},
            )

        # No price data
        return EdgeFactor(
            name=name,
            valid=True,
            score=0.5,
            reason="No price data",
        )

    def _check_time(self, context: TradeContext) -> EdgeFactor:
        """
        Check if we're in a favorable trading session.

        Sessions:
        - new_york (13:30-21:00 UTC): Best for most
        - london (08:00-16:30 UTC): Good for forex
        - asia (00:00-09:00 UTC): Lower volume
        - off_hours: Weekend, holidays
        """
        name = "time"

        now = datetime.now(timezone.utc)
        hour = now.hour
        weekday = now.weekday()

        # Weekend - generally bad
        if weekday >= 5:  # Saturday or Sunday
            return EdgeFactor(
                name=name,
                valid=False,
                score=0.2,
                reason="Weekend - low liquidity",
                data={"weekday": weekday, "hour": hour},
            )

        # Determine session
        session = context.session
        if not session:
            if 13 <= hour < 21:
                session = "new_york"
            elif 8 <= hour < 17:
                session = "london"
            elif hour < 9 or hour >= 22:
                session = "asia"
            else:
                session = "transition"

        # Score by session quality
        session_scores = {
            "new_york": (True, 1.0, "NY session - high liquidity"),
            "london": (True, 0.9, "London session - good liquidity"),
            "asia": (True, 0.6, "Asia session - lower liquidity"),
            "transition": (True, 0.7, "Session transition"),
            "off_hours": (False, 0.3, "Off hours - poor liquidity"),
        }

        valid, score, reason = session_scores.get(
            session.lower(),
            (True, 0.5, f"Unknown session: {session}")
        )

        # Friday late - reduce score (weekend gap risk)
        if weekday == 4 and hour >= 20:
            score *= 0.7
            reason = "Friday close - weekend gap risk"
            valid = False

        return EdgeFactor(
            name=name,
            valid=valid,
            score=score,
            reason=reason,
            data={"session": session, "hour": hour, "weekday": weekday},
        )

    def _determine_action(
        self,
        trade_id: str,
        valid_count: int,
        total_count: int,
    ) -> Tuple[EdgeAction, str]:
        """
        Determine action based on valid factor count.

        Uses consecutive check tracking to prevent whipsaw.
        """
        with self._lock:
            if valid_count < self.MIN_FACTORS_TO_HOLD:
                # Edge potentially lost
                self._consecutive_invalid[trade_id] = (
                    self._consecutive_invalid.get(trade_id, 0) + 1
                )

                if self._consecutive_invalid[trade_id] >= self.CONSECUTIVE_CHECKS_FOR_EXIT:
                    return (
                        EdgeAction.EXIT,
                        f"Edge lost: {valid_count}/{total_count} valid for "
                        f"{self._consecutive_invalid[trade_id]} checks"
                    )
                else:
                    return (
                        EdgeAction.HOLD,
                        f"Edge weakening: {valid_count}/{total_count} valid "
                        f"(check {self._consecutive_invalid[trade_id]}/{self.CONSECUTIVE_CHECKS_FOR_EXIT})"
                    )
            else:
                # Edge intact - reset counter
                self._consecutive_invalid[trade_id] = 0

                if valid_count >= self.MIN_FACTORS_TO_ADD:
                    return (
                        EdgeAction.ADD_OK,
                        f"Strong edge: {valid_count}/{total_count} valid - safe to pyramid"
                    )
                else:
                    return (
                        EdgeAction.HOLD,
                        f"Edge intact: {valid_count}/{total_count} valid"
                    )

    def get_last_state(self, trade_id: str) -> Optional[EdgeState]:
        """Get the last edge state for a trade."""
        with self._lock:
            return self._last_states.get(trade_id)

    def clear_trade(self, trade_id: str):
        """Clear tracking data for a closed trade."""
        with self._lock:
            self._consecutive_invalid.pop(trade_id, None)
            self._last_states.pop(trade_id, None)

    def get_all_states(self) -> Dict[str, EdgeState]:
        """Get all current edge states."""
        with self._lock:
            return dict(self._last_states)


# Singleton instance
_edge_monitor: Optional[EdgeMonitor] = None
_init_lock = threading.Lock()


def get_edge_monitor() -> EdgeMonitor:
    """Get or create singleton EdgeMonitor instance."""
    global _edge_monitor

    if _edge_monitor is None:
        with _init_lock:
            if _edge_monitor is None:
                _edge_monitor = EdgeMonitor()

    return _edge_monitor


# Convenience function for quick edge check
def check_trade_edge(
    trade_id: str,
    symbol: str,
    direction: str,
    entry_price: float,
    current_price: float,
    entry_time: datetime,
    rsi: Optional[float] = None,
    macd: Optional[float] = None,
    macd_signal: Optional[float] = None,
    session: Optional[str] = None,
    candles: Optional[List[Dict]] = None,
) -> EdgeState:
    """
    Convenience function to check edge for a trade.

    Returns:
        EdgeState with action recommendation
    """
    context = TradeContext(
        trade_id=trade_id,
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
        current_price=current_price,
        entry_time=entry_time,
        rsi=rsi,
        macd=macd,
        macd_signal=macd_signal,
        session=session,
    )

    return get_edge_monitor().check_edge(context, candles)


if __name__ == "__main__":
    # Test the edge monitor
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("EDGE MONITOR TEST")
    print("=" * 60)

    monitor = get_edge_monitor()

    # Simulate a trade with good edge
    context1 = TradeContext(
        trade_id="test_001",
        symbol="BTC-USD",
        direction="BUY",
        entry_price=100000.0,
        current_price=100500.0,  # +0.5%
        entry_time=datetime.now(timezone.utc),
        rsi=55.0,  # Healthy
        macd=100.0,
        macd_signal=80.0,  # MACD bullish
        session="new_york",
    )

    # Sample candles with bullish structure
    candles1 = [
        {"high": 99500, "low": 99000, "volume": 1000},
        {"high": 100000, "low": 99500, "volume": 1100},
        {"high": 100500, "low": 100000, "volume": 1200},  # HH + HL
    ]

    print("\n1. Trade with GOOD edge (BUY at +0.5%):")
    state1 = monitor.check_edge(context1, candles1)
    print(f"   Valid factors: {state1.valid_count}/{state1.total_count}")
    print(f"   Action: {state1.action.value}")
    print(f"   Reason: {state1.reason}")
    for f in state1.factors:
        status = "OK" if f.valid else "X "
        print(f"     [{status}] {f.name}: {f.reason} (score: {f.score:.2f})")

    # Simulate a trade losing edge
    context2 = TradeContext(
        trade_id="test_002",
        symbol="ETH-USD",
        direction="BUY",
        entry_price=4000.0,
        current_price=3950.0,  # -1.25% (against us)
        entry_time=datetime.now(timezone.utc),
        rsi=75.0,  # Overbought
        macd=-10.0,
        macd_signal=5.0,  # MACD bearish
        session="asia",  # Lower liquidity
    )

    # Candles with bearish structure
    candles2 = [
        {"high": 4100, "low": 4000, "volume": 1000},
        {"high": 4050, "low": 3980, "volume": 900},  # LH + LL
        {"high": 4000, "low": 3950, "volume": 800},  # LH + LL
    ]

    print("\n2. Trade with WEAK edge (BUY at -1.25%):")
    state2 = monitor.check_edge(context2, candles2)
    print(f"   Valid factors: {state2.valid_count}/{state2.total_count}")
    print(f"   Action: {state2.action.value}")
    print(f"   Reason: {state2.reason}")
    for f in state2.factors:
        status = "OK" if f.valid else "X "
        print(f"     [{status}] {f.name}: {f.reason} (score: {f.score:.2f})")

    # Second check should trigger exit (consecutive)
    print("\n3. Second check (should trigger EXIT):")
    state3 = monitor.check_edge(context2, candles2)
    print(f"   Valid factors: {state3.valid_count}/{state3.total_count}")
    print(f"   Action: {state3.action.value}")
    print(f"   Reason: {state3.reason}")

    # Strong edge trade (safe to pyramid)
    context4 = TradeContext(
        trade_id="test_003",
        symbol="SOL-USD",
        direction="BUY",
        entry_price=200.0,
        current_price=204.0,  # +2%
        entry_time=datetime.now(timezone.utc),
        rsi=45.0,  # Healthy
        macd=5.0,
        macd_signal=2.0,  # MACD bullish
        session="new_york",
        current_volume=1500,
        avg_volume=1000,  # 1.5x volume
    )

    candles4 = [
        {"high": 198, "low": 196, "volume": 1000},
        {"high": 201, "low": 199, "volume": 1200},
        {"high": 205, "low": 202, "volume": 1500},  # HH + HL + volume
    ]

    print("\n4. Trade with STRONG edge (safe to pyramid):")
    state4 = monitor.check_edge(context4, candles4)
    print(f"   Valid factors: {state4.valid_count}/{state4.total_count}")
    print(f"   Action: {state4.action.value}")
    print(f"   Reason: {state4.reason}")
    for f in state4.factors:
        status = "OK" if f.valid else "X "
        print(f"     [{status}] {f.name}: {f.reason} (score: {f.score:.2f})")

    print("\n" + "=" * 60)
    print("EDGE MONITOR TEST COMPLETE")
    print("=" * 60)
