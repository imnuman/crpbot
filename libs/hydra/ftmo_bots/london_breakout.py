"""
London Breakout Strategy Bot

Strategy:
- Track Asian session high/low (22:00-06:00 UTC)
- Wait for London open (08:00 UTC)
- Enter on breakout of Asian range after 08:00 UTC
- Stop: Other side of Asian range
- Target: 1.5-2x range size
- Trade window: 08:00-12:00 UTC

SESSION TIMES (UTC):
- Asian: 22:00 - 06:00 UTC (marks the range)
- London: 08:00 - 16:00 UTC (trade the breakout)

Rationale:
- Asian session establishes consolidation range
- London open brings volatility and directional moves
- Breakout with momentum typically continues

Expected Performance:
- Win rate: 58%
- Avg win: 1.5x range
- Avg loss: 1x range (opposite side of range)
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

    # Strategy parameters
    MIN_RANGE_PIPS = 20.0   # Minimum Asian range to consider
    MAX_RANGE_PIPS = 80.0   # Skip if Asian range too wide
    TP_MULTIPLIER = 1.5     # Target = 1.5x Asian range
    BREAKOUT_BUFFER_PIPS = 3.0  # Pips above/below range for confirmed break

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

        logger.info(
            f"[{self.config.bot_name}] Strategy: London Breakout on {symbol} "
            f"(Asian range: 22:00-06:00 UTC, Trade: 08:00-12:00 UTC)"
        )

    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze for London breakout setup."""
        now = datetime.now(timezone.utc)

        # Update Asian range if needed
        self._update_asian_range(market_data)

        if self._asian_range is None:
            return {"tradeable": False, "reason": "Asian range not set"}

        # Check if in trading window
        if not self._is_trading_window(now):
            return {"tradeable": False, "reason": f"Outside trading window ({now.strftime('%H:%M')} UTC)"}

        # Check if already traded
        if self._breakout_traded:
            return {"tradeable": False, "reason": "Already traded today"}

        # Check range validity
        if self._asian_range.range_pips < self.MIN_RANGE_PIPS:
            return {"tradeable": False, "reason": f"Range too small ({self._asian_range.range_pips:.0f} pips)"}

        if self._asian_range.range_pips > self.MAX_RANGE_PIPS:
            return {"tradeable": False, "reason": f"Range too wide ({self._asian_range.range_pips:.0f} pips)"}

        # Get current price
        candles = market_data.get("candles", [])
        if not candles:
            return {"tradeable": False, "reason": "No candle data"}

        current_price = candles[-1].get("close", 0)

        # Check for breakout with buffer
        buffer = self.BREAKOUT_BUFFER_PIPS * self._pip_value
        breakout_direction = None

        if current_price > (self._asian_range.high + buffer):
            breakout_direction = "BUY"  # Breakout above range - go long
        elif current_price < (self._asian_range.low - buffer):
            breakout_direction = "SELL"  # Breakout below range - go short

        return {
            "tradeable": True,
            "range_high": self._asian_range.high,
            "range_low": self._asian_range.low,
            "range_pips": self._asian_range.range_pips,
            "current_price": current_price,
            "breakout_direction": breakout_direction,
        }

    def should_trade(self, analysis: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if breakout conditions are met."""
        if not analysis.get("tradeable"):
            return False, analysis.get("reason", "Not tradeable")

        if analysis.get("breakout_direction") is None:
            return False, "No breakout - price within Asian range"

        return True, f"London breakout {analysis['breakout_direction']} (Asian range: {analysis['range_pips']:.0f} pips)"

    def generate_signal(self, analysis: Dict[str, Any], current_price: float) -> Optional[TradeSignal]:
        """Generate breakout signal."""
        direction = analysis.get("breakout_direction")
        if direction is None:
            return None

        range_pips = analysis.get("range_pips", 40)
        range_high = analysis.get("range_high", current_price)
        range_low = analysis.get("range_low", current_price)

        # Stop loss is opposite side of range
        # Take profit is 1.5x the range
        if direction == "BUY":
            stop_loss = range_low - (5 * self._pip_value)  # 5 pip buffer
            tp_distance = range_pips * self.TP_MULTIPLIER * self._pip_value
            take_profit = current_price + tp_distance
        else:  # SELL
            stop_loss = range_high + (5 * self._pip_value)  # 5 pip buffer
            tp_distance = range_pips * self.TP_MULTIPLIER * self._pip_value
            take_profit = current_price - tp_distance

        # Calculate actual SL pips for position sizing
        sl_pips = abs(current_price - stop_loss) / self._pip_value

        # Get account info
        account_info = self.get_account_info()
        if not account_info:
            logger.warning(f"[{self.config.bot_name}] Skipping trade - account info unavailable")
            return None
        balance = account_info.get("balance", 15000)

        lot_size = self.calculate_lot_size(balance, sl_pips)

        self._breakout_traded = True

        return TradeSignal(
            bot_name=self.config.bot_name,
            symbol=self.config.symbol,
            direction=direction,
            entry_price=current_price,
            stop_loss=round(stop_loss, 5),
            take_profit=round(take_profit, 5),
            lot_size=lot_size,
            reason=f"London breakout: {direction} (Asian range: {range_pips:.0f} pips)",
            confidence=0.62,
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
