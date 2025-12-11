"""
London Breakout Strategy Bot (v3 - Critical Bug Fixes)

Strategy:
- Track Asian session high/low (22:00-06:00 UTC)
- Wait for London open (08:00 UTC)
- CONFIRM breakout with RETEST pattern (break → pullback → continuation)
- Only trade in daily trend direction (200 SMA filter)
- Stop: OPPOSITE side of Asian range
- Target: 1.0x range size
- Trade window: 08:00-12:00 UTC

SESSION TIMES (UTC):
- Asian: 22:00 - 06:00 UTC (marks the range)
- London: 08:00 - 16:00 UTC (trade the breakout)

v3 BUG FIXES (2025-12-11):
1. CRITICAL: Stop loss was placed on WRONG SIDE of range
   - Old (broken): BUY SL below range HIGH (immediate stop on retest)
   - New (fixed): BUY SL below range LOW (proper breakout protection)
2. SMA period was too short (20 M1 candles = 20 minutes)
   - New: 200 M1 candles (~3.3 hours) for meaningful trend

FALSE BREAKOUT PREVENTION:
1. RETEST confirmation - price must break range, pull back, then continue
2. CLOSE confirmation - candle must CLOSE beyond range, not just wick
3. Trend filter - only trade in direction of 200-period SMA

Expected Performance (v3):
- Win rate: 50-55% (with proper SL placement)
- Risk/Reward: ~1:1 (range as target, range as stop)
- Best pairs: EURUSD, GBPUSD
"""

import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass
from loguru import logger

from .base_ftmo_bot import BaseFTMOBot, BotConfig, TradeSignal


@dataclass
class AsianRange:
    """Asian session range data."""
    high: float
    low: float
    range_pips: float
    timestamp: datetime


@dataclass
class BreakoutState:
    """Track breakout confirmation state."""
    direction: str  # "BUY" or "SELL"
    initial_break_time: datetime
    initial_break_price: float
    retest_detected: bool = False
    retest_time: Optional[datetime] = None
    confirmed: bool = False


class LondonBreakoutBot(BaseFTMOBot):
    """
    London Breakout Strategy

    Trades breakout of Asian session range at London open.
    """

    # Asian session (for range calculation)
    ASIAN_START_HOUR = 22  # 22:00 UTC (previous day)
    ASIAN_END_HOUR = 6     # 06:00 UTC

    # London trading window
    TRADE_START_HOUR = 8   # 08:00 UTC (London open)
    TRADE_END_HOUR = 12    # 12:00 UTC (end breakout window)

    # Strategy parameters (v2 - FALSE BREAKOUT PREVENTION)
    MIN_RANGE_PIPS = 20.0   # Back to 20 - EURUSD typically has 15-25 pip Asian ranges
    MAX_RANGE_PIPS = 80.0   # Skip if Asian range too wide
    TP_MULTIPLIER = 1.0     # Reduced from 1.5 - faster exits
    BREAKOUT_BUFFER_PIPS = 3.0  # Pips above/below range for initial break detection

    # RETEST confirmation parameters
    RETEST_THRESHOLD_PIPS = 10.0  # How close price must get to range for "retest"
    RETEST_TIMEOUT_MINUTES = 60   # Max time to wait for retest after initial break
    CONTINUATION_BUFFER_PIPS = 5.0  # Price must move X pips beyond range after retest

    # Trend filter parameters
    # v3: Use 200 M1 candles (~3.3 hours) for trend direction
    # Previous bug: 20 candles = 20 minutes, not enough data
    SMA_PERIOD = 200  # 200 M1 candles for trend direction

    # Symbol pip values
    PIP_VALUES = {
        "EURUSD": 0.0001,
        "GBPUSD": 0.0001,
        "USDJPY": 0.01,
    }

    def __init__(self, symbol: str = "EURUSD", paper_mode: bool = True):
        pip_value = self.PIP_VALUES.get(symbol, 0.0001)

        config = BotConfig(
            bot_name="LondonBreakout",
            symbol=symbol,
            risk_percent=0.015,  # 1.5% risk
            max_daily_trades=1,  # One trade per day
            stop_loss_pips=50.0,  # Will be adjusted based on range
            take_profit_pips=75.0,  # Will be adjusted based on range
            max_hold_hours=4.0,
            enabled=True,
            paper_mode=paper_mode,
        )
        super().__init__(config)

        self._asian_range: Optional[AsianRange] = None
        self._range_date: Optional[datetime] = None
        self._breakout_traded = False
        self._pip_value = pip_value

        # v2: Retest confirmation state
        self._breakout_state: Optional[BreakoutState] = None
        self._daily_sma: Optional[float] = None

        logger.info(
            f"[{self.config.bot_name}] Strategy: London Breakout v2 on {symbol} "
            f"(RETEST confirmation enabled, SMA trend filter)"
        )

    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze for London breakout setup with RETEST confirmation.

        v2 Pattern:
        1. Initial break: candle CLOSES beyond Asian range
        2. Retest: price pulls back within RETEST_THRESHOLD_PIPS of range
        3. Confirmation: price moves CONTINUATION_BUFFER_PIPS beyond range again
        4. Trend filter: only trade in direction of 20 SMA
        """
        now = datetime.now(timezone.utc)

        # Update Asian range if needed
        self._update_asian_range(market_data)

        if self._asian_range is None:
            return {"tradeable": False, "reason": "Asian range not set"}

        # Check if in trading window
        if not self._is_trading_window(now):
            # Reset breakout state outside trading window
            self._breakout_state = None
            return {"tradeable": False, "reason": f"Outside trading window ({now.strftime('%H:%M')} UTC)"}

        # Check if already traded
        if self._breakout_traded:
            return {"tradeable": False, "reason": "Already traded today"}

        # Check range validity
        if self._asian_range.range_pips < self.MIN_RANGE_PIPS:
            return {"tradeable": False, "reason": f"Range too small ({self._asian_range.range_pips:.0f} pips < {self.MIN_RANGE_PIPS})"}

        if self._asian_range.range_pips > self.MAX_RANGE_PIPS:
            return {"tradeable": False, "reason": f"Range too wide ({self._asian_range.range_pips:.0f} pips)"}

        # Get candles
        candles = market_data.get("candles", [])
        if not candles or len(candles) < 2:
            return {"tradeable": False, "reason": "Insufficient candle data"}

        current_candle = candles[-1]
        current_close = current_candle.get("close", 0)
        current_high = current_candle.get("high", 0)
        current_low = current_candle.get("low", 0)

        # Calculate 20 SMA for trend filter
        self._update_daily_sma(candles)

        # Check for initial breakout (if not already tracking one)
        if self._breakout_state is None:
            breakout = self._check_initial_breakout(current_close, now)
            if breakout:
                self._breakout_state = breakout
                logger.info(
                    f"[{self.config.bot_name}] Initial {breakout.direction} breakout detected at {current_close:.5f}. "
                    f"Waiting for RETEST confirmation..."
                )

        # Process breakout state machine
        confirmed_direction = None
        if self._breakout_state:
            confirmed_direction = self._process_breakout_state(current_close, current_high, current_low, now)

            # Check for retest timeout
            if not self._breakout_state.confirmed:
                elapsed_minutes = (now - self._breakout_state.initial_break_time).total_seconds() / 60
                if elapsed_minutes > self.RETEST_TIMEOUT_MINUTES:
                    logger.info(f"[{self.config.bot_name}] Breakout retest timeout - resetting state")
                    self._breakout_state = None

        # Check trend filter
        trend_aligned = self._check_trend_alignment(confirmed_direction) if confirmed_direction else False

        return {
            "tradeable": True,
            "range_high": self._asian_range.high,
            "range_low": self._asian_range.low,
            "range_pips": self._asian_range.range_pips,
            "current_price": current_close,
            "breakout_direction": confirmed_direction if trend_aligned else None,
            "breakout_state": self._breakout_state,
            "daily_sma": self._daily_sma,
            "trend_aligned": trend_aligned,
        }

    def should_trade(self, analysis: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if breakout conditions are met with RETEST confirmation."""
        if not analysis.get("tradeable"):
            return False, analysis.get("reason", "Not tradeable")

        breakout_state = analysis.get("breakout_state")
        direction = analysis.get("breakout_direction")

        # No breakout at all
        if breakout_state is None and direction is None:
            return False, "No breakout - price within Asian range"

        # Breakout detected but not confirmed yet
        if breakout_state and not breakout_state.confirmed:
            if not breakout_state.retest_detected:
                return False, f"Initial {breakout_state.direction} break - waiting for RETEST"
            else:
                return False, f"{breakout_state.direction} retest detected - waiting for CONTINUATION"

        # Breakout confirmed but trend not aligned
        if direction is None and breakout_state and breakout_state.confirmed:
            return False, f"Breakout confirmed but BLOCKED by trend filter"

        # All conditions met
        if direction is not None:
            range_pips = analysis.get("range_pips", 0)
            return True, f"London breakout {direction} CONFIRMED (range: {range_pips:.0f} pips, trend aligned)"

        return False, "Breakout conditions not met"

    def generate_signal(self, analysis: Dict[str, Any], current_price: float) -> Optional[TradeSignal]:
        """Generate breakout signal after RETEST confirmation."""
        direction = analysis.get("breakout_direction")
        if direction is None:
            return None

        range_pips = analysis.get("range_pips", 40)
        range_high = analysis.get("range_high", current_price)
        range_low = analysis.get("range_low", current_price)

        # v3 FIX: Stop loss on OPPOSITE side of range (not near entry!)
        # Previous bug: SL was placed near entry, causing immediate stops on normal retests
        # Correct placement: SL beyond the opposite side of the range
        if direction == "BUY":
            # For BUY breakout: stop below the Asian LOW (opposite side)
            stop_loss = range_low - (10 * self._pip_value)  # 10 pip buffer below range low
            tp_distance = range_pips * self.TP_MULTIPLIER * self._pip_value
            take_profit = current_price + tp_distance
        else:  # SELL
            # For SELL breakout: stop above the Asian HIGH (opposite side)
            stop_loss = range_high + (10 * self._pip_value)  # 10 pip buffer above range high
            tp_distance = range_pips * self.TP_MULTIPLIER * self._pip_value
            take_profit = current_price - tp_distance

        # Calculate actual SL pips for position sizing
        sl_pips = abs(current_price - stop_loss) / self._pip_value

        # Safety check: ensure reasonable SL (min 15 pips, max 50 pips)
        sl_pips = max(15, min(50, sl_pips))

        # Get account info
        account_info = self.get_account_info()
        if not account_info:
            logger.warning(f"[{self.config.bot_name}] Skipping trade - account info unavailable")
            return None
        balance = account_info.get("balance", 15000)

        lot_size = self.calculate_lot_size(balance, sl_pips)

        self._breakout_traded = True
        self._breakout_state = None  # Reset state after trade

        return TradeSignal(
            bot_name=self.config.bot_name,
            symbol=self.config.symbol,
            direction=direction,
            entry_price=current_price,
            stop_loss=round(stop_loss, 5),
            take_profit=round(take_profit, 5),
            lot_size=lot_size,
            reason=f"London breakout v2: {direction} (RETEST confirmed, range: {range_pips:.0f} pips)",
            confidence=0.55,  # Lower confidence due to more filters, but higher quality trades
        )

    def _is_trading_window(self, now: datetime) -> bool:
        """Check if in London trading window (08:00-12:00 UTC)."""
        return self.TRADE_START_HOUR <= now.hour < self.TRADE_END_HOUR

    def _is_asian_session(self, hour: int) -> bool:
        """Check if hour is in Asian session (22:00-06:00 UTC)."""
        return hour >= self.ASIAN_START_HOUR or hour < self.ASIAN_END_HOUR

    def _update_asian_range(self, market_data: Dict[str, Any]) -> None:
        """Calculate Asian session high/low from 22:00-06:00 UTC candles."""
        today = datetime.now(timezone.utc).date()

        # Reset at new day
        if self._range_date != today:
            self._asian_range = None
            self._breakout_traded = False
            self._breakout_state = None  # v2: Reset retest state
            self._range_date = today

        # Only calculate once per day, after Asian session ends
        if self._asian_range is not None:
            return

        now = datetime.now(timezone.utc)
        # Wait until Asian session is complete (after 06:00 UTC)
        if now.hour < self.ASIAN_END_HOUR:
            return

        candles = market_data.get("candles", [])
        if not candles:
            return

        # Filter candles to Asian session (22:00-06:00 UTC)
        asian_candles = []
        yesterday = today - timedelta(days=1)

        for candle in candles:
            ts = candle.get("timestamp") or candle.get("time")
            if isinstance(ts, str):
                try:
                    candle_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except Exception:
                    continue
            elif isinstance(ts, (int, float)):
                candle_time = datetime.fromtimestamp(ts, tz=timezone.utc)
            else:
                continue

            # Check if in Asian session
            # Either yesterday after 22:00 or today before 06:00
            if candle_time.date() == yesterday and candle_time.hour >= self.ASIAN_START_HOUR:
                asian_candles.append(candle)
            elif candle_time.date() == today and candle_time.hour < self.ASIAN_END_HOUR:
                asian_candles.append(candle)

        if len(asian_candles) >= 12:  # At least 12 candles for 6-hour period
            high = max(c.get("high", 0) for c in asian_candles)
            low = min(c.get("low", float("inf")) for c in asian_candles)

            if high > 0 and low < float("inf"):
                range_pips = (high - low) / self._pip_value

                self._asian_range = AsianRange(
                    high=high,
                    low=low,
                    range_pips=range_pips,
                    timestamp=datetime.now(timezone.utc),
                )
                logger.info(
                    f"[{self.config.bot_name}] Asian range set: "
                    f"High: {high:.5f}, Low: {low:.5f}, Range: {range_pips:.0f} pips"
                )

    def _check_initial_breakout(self, current_close: float, now: datetime) -> Optional[BreakoutState]:
        """
        Check for initial breakout (candle CLOSE beyond Asian range).

        Returns BreakoutState if a new breakout is detected, None otherwise.
        """
        if self._asian_range is None:
            return None

        buffer = self.BREAKOUT_BUFFER_PIPS * self._pip_value

        # Check for bullish breakout (close above range high)
        if current_close > (self._asian_range.high + buffer):
            return BreakoutState(
                direction="BUY",
                initial_break_time=now,
                initial_break_price=current_close,
            )

        # Check for bearish breakout (close below range low)
        if current_close < (self._asian_range.low - buffer):
            return BreakoutState(
                direction="SELL",
                initial_break_time=now,
                initial_break_price=current_close,
            )

        return None

    def _process_breakout_state(
        self, current_close: float, current_high: float, current_low: float, now: datetime
    ) -> Optional[str]:
        """
        Process the breakout state machine for retest confirmation.

        States:
        1. Initial break detected (waiting for retest)
        2. Retest detected (price pulled back to range)
        3. Confirmation (price moved beyond range again)

        Returns direction string if breakout is confirmed, None otherwise.
        """
        if self._breakout_state is None or self._asian_range is None:
            return None

        retest_threshold = self.RETEST_THRESHOLD_PIPS * self._pip_value
        continuation_buffer = self.CONTINUATION_BUFFER_PIPS * self._pip_value

        if self._breakout_state.direction == "BUY":
            # For bullish breakout:
            # - Retest: price pulls back close to range high
            # - Confirmation: price closes above range high + continuation buffer

            if not self._breakout_state.retest_detected:
                # Check for retest (price came back near range high)
                distance_from_high = current_low - self._asian_range.high
                if distance_from_high <= retest_threshold:
                    self._breakout_state.retest_detected = True
                    self._breakout_state.retest_time = now
                    logger.info(
                        f"[{self.config.bot_name}] BUY breakout RETEST detected. "
                        f"Waiting for continuation above {self._asian_range.high + continuation_buffer:.5f}"
                    )

            elif not self._breakout_state.confirmed:
                # Check for continuation (close above range + buffer)
                if current_close > (self._asian_range.high + continuation_buffer):
                    self._breakout_state.confirmed = True
                    logger.info(
                        f"[{self.config.bot_name}] BUY breakout CONFIRMED after retest at {current_close:.5f}"
                    )
                    return "BUY"

        else:  # SELL
            # For bearish breakout:
            # - Retest: price pulls back close to range low
            # - Confirmation: price closes below range low - continuation buffer

            if not self._breakout_state.retest_detected:
                # Check for retest (price came back near range low)
                distance_from_low = self._asian_range.low - current_high
                if distance_from_low <= retest_threshold:
                    self._breakout_state.retest_detected = True
                    self._breakout_state.retest_time = now
                    logger.info(
                        f"[{self.config.bot_name}] SELL breakout RETEST detected. "
                        f"Waiting for continuation below {self._asian_range.low - continuation_buffer:.5f}"
                    )

            elif not self._breakout_state.confirmed:
                # Check for continuation (close below range - buffer)
                if current_close < (self._asian_range.low - continuation_buffer):
                    self._breakout_state.confirmed = True
                    logger.info(
                        f"[{self.config.bot_name}] SELL breakout CONFIRMED after retest at {current_close:.5f}"
                    )
                    return "SELL"

        return None

    def _update_daily_sma(self, candles: List[Dict]) -> None:
        """Calculate 20-period SMA for trend filter."""
        if len(candles) < self.SMA_PERIOD:
            self._daily_sma = None
            return

        # Use last N candles' close prices
        closes = [c.get("close", 0) for c in candles[-self.SMA_PERIOD:]]
        if all(c > 0 for c in closes):
            self._daily_sma = sum(closes) / len(closes)

    def _check_trend_alignment(self, direction: Optional[str]) -> bool:
        """
        Check if the breakout direction aligns with the overall trend.

        - BUY: only allowed if price is above 20 SMA (uptrend)
        - SELL: only allowed if price is below 20 SMA (downtrend)
        """
        if direction is None or self._daily_sma is None:
            return False

        if self._asian_range is None:
            return False

        # Use range midpoint as reference
        range_mid = (self._asian_range.high + self._asian_range.low) / 2

        if direction == "BUY":
            # Bullish breakout should be in uptrend (price > SMA)
            is_aligned = range_mid > self._daily_sma
            if not is_aligned:
                logger.info(
                    f"[{self.config.bot_name}] BUY breakout BLOCKED - against trend "
                    f"(price {range_mid:.5f} < SMA {self._daily_sma:.5f})"
                )
            return is_aligned

        else:  # SELL
            # Bearish breakout should be in downtrend (price < SMA)
            is_aligned = range_mid < self._daily_sma
            if not is_aligned:
                logger.info(
                    f"[{self.config.bot_name}] SELL breakout BLOCKED - against trend "
                    f"(price {range_mid:.5f} > SMA {self._daily_sma:.5f})"
                )
            return is_aligned

    def check_and_trade(self) -> Optional[TradeSignal]:
        """Main scheduler method."""
        now = datetime.now(timezone.utc)
        if not self._is_trading_window(now):
            return None
        return self.run_cycle()


# Singleton instances (one per symbol)
_london_bots: Dict[str, LondonBreakoutBot] = {}
_bot_lock = threading.Lock()


def get_london_breakout_bot(symbol: str = "EURUSD", paper_mode: bool = True) -> LondonBreakoutBot:
    """Get London Breakout bot singleton for symbol."""
    global _london_bots
    if symbol not in _london_bots:
        with _bot_lock:
            if symbol not in _london_bots:
                _london_bots[symbol] = LondonBreakoutBot(symbol=symbol, paper_mode=paper_mode)
    return _london_bots[symbol]
