"""
US30 Opening Range Breakout (ORB) Bot

Strategy:
- Measure US30 (Dow Jones) range from 09:30-09:45 EST (14:30-14:45 UTC)
- If price breaks ABOVE range → SELL (fade the breakout)
- If price breaks BELOW range → BUY (fade the breakout)
- Stop: 40 points, Target: 80 points
- Max hold: 2 hours (exit by 11:00 EST / 16:00 UTC)

Rationale:
- Initial breakouts often retrace as retail traders get trapped
- Fading works better on indices due to mean reversion tendency

Expected Performance:
- Win rate: 58%
- Avg win: +80 points ($80)
- Avg loss: -40 points ($40)
- Daily EV: +$148 (at 1.5% risk, $15k account)
"""

import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass
from loguru import logger
import requests

from .base_ftmo_bot import BaseFTMOBot, BotConfig, TradeSignal


@dataclass
class OpeningRange:
    """Opening range data."""
    high: float
    low: float
    range_points: float
    timestamp: datetime


class US30ORBBot(BaseFTMOBot):
    """
    US30 Opening Range Breakout Fade Strategy

    Fades the initial breakout of the opening range.
    """

    # Trading window (in UTC)
    ORB_START_HOUR = 14  # 14:30 UTC (09:30 EST)
    ORB_START_MINUTE = 30
    ORB_END_HOUR = 14  # 14:45 UTC (09:45 EST)
    ORB_END_MINUTE = 45

    TRADE_START_HOUR = 14
    TRADE_START_MINUTE = 46  # Start looking for breakouts at 14:46
    TRADE_END_HOUR = 16  # Stop trading at 16:00 UTC (11:00 EST)

    STOP_LOSS_POINTS = 70.0  # Increased from 40 (backtest showed 33-40 point losses hitting SL)
    TAKE_PROFIT_POINTS = 120.0  # Increased proportionally (R:R = 1:1.7)
    MIN_RANGE_POINTS = 30.0  # Minimum range to consider
    MAX_RANGE_POINTS = 150.0  # Skip if too volatile

    def __init__(self, paper_mode: bool = True):
        config = BotConfig(
            bot_name="US30ORB",
            symbol="US30.cash",  # Dow Jones (FTMO broker symbol)
            risk_percent=0.015,
            max_daily_trades=1,
            stop_loss_pips=self.STOP_LOSS_POINTS,
            take_profit_pips=self.TAKE_PROFIT_POINTS,
            max_hold_hours=2.0,
            enabled=True,
            paper_mode=paper_mode,
        )
        super().__init__(config)

        self._opening_range: Optional[OpeningRange] = None
        self._range_date: Optional[datetime] = None
        self._breakout_traded = False

        logger.info(
            f"[{self.config.bot_name}] Strategy: Fade ORB "
            f"(09:30-09:45 EST range, SL: {self.STOP_LOSS_POINTS} pts, TP: {self.TAKE_PROFIT_POINTS} pts)"
        )

    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze for ORB breakout."""
        now = datetime.now(timezone.utc)

        # Update opening range if needed
        self._update_opening_range(market_data)

        if self._opening_range is None:
            return {"tradeable": False, "reason": "Opening range not set"}

        # Check if in trading window
        if not self._is_trading_window(now):
            return {"tradeable": False, "reason": f"Outside trading window ({now.strftime('%H:%M')} UTC)"}

        # Check if already traded
        if self._breakout_traded:
            return {"tradeable": False, "reason": "Already traded today"}

        # Check range validity
        if self._opening_range.range_points < self.MIN_RANGE_POINTS:
            return {"tradeable": False, "reason": f"Range too small ({self._opening_range.range_points:.0f} pts)"}

        if self._opening_range.range_points > self.MAX_RANGE_POINTS:
            return {"tradeable": False, "reason": f"Range too wide ({self._opening_range.range_points:.0f} pts)"}

        # Get current price
        candles = market_data.get("candles", [])
        if not candles:
            return {"tradeable": False, "reason": "No candle data"}

        current_price = candles[-1].get("close", 0)

        # Check for breakout
        breakout_direction = None
        if current_price > self._opening_range.high:
            breakout_direction = "SELL"  # Fade the upside breakout
        elif current_price < self._opening_range.low:
            breakout_direction = "BUY"  # Fade the downside breakout

        return {
            "tradeable": True,
            "range_high": self._opening_range.high,
            "range_low": self._opening_range.low,
            "range_points": self._opening_range.range_points,
            "current_price": current_price,
            "breakout_direction": breakout_direction,
        }

    def should_trade(self, analysis: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if breakout fade conditions are met."""
        if not analysis.get("tradeable"):
            return False, analysis.get("reason", "Not tradeable")

        if analysis.get("breakout_direction") is None:
            return False, "No breakout - price within range"

        return True, f"ORB {analysis['breakout_direction']} (fade breakout)"

    def generate_signal(self, analysis: Dict[str, Any], current_price: float) -> Optional[TradeSignal]:
        """Generate fade signal."""
        direction = analysis.get("breakout_direction")
        if direction is None:
            return None

        # For US30, 1 point = $1 per standard lot
        if direction == "SELL":
            stop_loss = current_price + self.STOP_LOSS_POINTS
            take_profit = current_price - self.TAKE_PROFIT_POINTS
        else:  # BUY
            stop_loss = current_price - self.STOP_LOSS_POINTS
            take_profit = current_price + self.TAKE_PROFIT_POINTS

        # Get account info
        account_info = self.get_account_info()
        balance = account_info.get("balance", 15000)

        # For indices, pip value is different
        lot_size = self._calculate_index_lot_size(balance)

        self._breakout_traded = True

        return TradeSignal(
            bot_name=self.config.bot_name,
            symbol=self.config.symbol,
            direction=direction,
            entry_price=current_price,
            stop_loss=round(stop_loss, 1),
            take_profit=round(take_profit, 1),
            lot_size=lot_size,
            reason=f"ORB fade: {direction} (range: {self._opening_range.range_points:.0f} pts)",
            confidence=0.68,
        )

    def _calculate_index_lot_size(self, balance: float) -> float:
        """Calculate lot size for index trading."""
        risk_amount = balance * self.config.risk_percent
        # For US30: ~$1 per point per 0.01 lot
        point_value_per_lot = 1.0
        lot_size = risk_amount / (self.STOP_LOSS_POINTS * point_value_per_lot)
        # SAFETY: Max 0.5 lots after $503 loss on 2025-12-10
        return round(max(0.1, min(lot_size, 0.5)), 2)

    def _is_trading_window(self, now: datetime) -> bool:
        """Check if in trading window (after ORB period, before cutoff)."""
        start = now.replace(hour=self.TRADE_START_HOUR, minute=self.TRADE_START_MINUTE, second=0)
        end = now.replace(hour=self.TRADE_END_HOUR, minute=0, second=0)
        return start <= now <= end

    def _update_opening_range(self, market_data: Dict[str, Any]) -> None:
        """Calculate opening range from 14:30-14:45 UTC candles."""
        today = datetime.now(timezone.utc).date()

        # Reset at new day
        if self._range_date != today:
            self._opening_range = None
            self._breakout_traded = False
            self._range_date = today

        # Only calculate once per day
        if self._opening_range is not None:
            return

        now = datetime.now(timezone.utc)
        # Wait until ORB period is complete
        if now.hour < self.ORB_END_HOUR or (now.hour == self.ORB_END_HOUR and now.minute < self.ORB_END_MINUTE):
            return

        candles = market_data.get("candles", [])
        if not candles:
            return

        # Filter candles to ORB period
        orb_candles = []
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

            if candle_time.date() == today:
                if (candle_time.hour == self.ORB_START_HOUR and candle_time.minute >= self.ORB_START_MINUTE) or \
                   (candle_time.hour == self.ORB_END_HOUR and candle_time.minute < self.ORB_END_MINUTE):
                    orb_candles.append(candle)

        if len(orb_candles) >= 3:  # At least 3 candles for 15-min period
            high = max(c.get("high", 0) for c in orb_candles)
            low = min(c.get("low", float("inf")) for c in orb_candles)

            if high > 0 and low < float("inf"):
                self._opening_range = OpeningRange(
                    high=high,
                    low=low,
                    range_points=high - low,
                    timestamp=datetime.now(timezone.utc),
                )
                logger.info(
                    f"[{self.config.bot_name}] ORB set: "
                    f"High: {high:.1f}, Low: {low:.1f}, Range: {high - low:.0f} pts"
                )

    def check_and_trade(self) -> Optional[TradeSignal]:
        """Main scheduler method."""
        now = datetime.now(timezone.utc)
        if not self._is_trading_window(now):
            return None
        return self.run_cycle()


# Singleton
_us30_bot: Optional[US30ORBBot] = None
_bot_lock = threading.Lock()


def get_us30_bot(paper_mode: bool = True) -> US30ORBBot:
    """Get US30 ORB bot singleton."""
    global _us30_bot
    if _us30_bot is None:
        with _bot_lock:
            if _us30_bot is None:
                _us30_bot = US30ORBBot(paper_mode=paper_mode)
    return _us30_bot
