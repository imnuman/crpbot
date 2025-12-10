"""
NAS100 Gap Fill Bot

Strategy:
- Detect gap at NASDAQ open (14:30 UTC / 09:30 EST)
- If gap UP >0.4% from previous close → SELL (to fill gap)
- If gap DOWN >0.4% from previous close → BUY (to fill gap)
- Target: 70% of gap filled
- Stop: 25 points
- Exit by 16:30 UTC (11:30 EST) maximum

Rationale:
- Index gaps tend to fill within first few hours
- 70-80% of gaps fill on same day

Expected Performance:
- Win rate: 65%
- Avg win: Variable (depends on gap size)
- Avg loss: -25 points
- Daily EV: +$150 (at 1.5% risk, $15k account)
"""

import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
from loguru import logger

from .base_ftmo_bot import BaseFTMOBot, BotConfig, TradeSignal


@dataclass
class GapData:
    """Gap detection data."""
    previous_close: float
    open_price: float
    gap_percent: float
    gap_direction: str  # "UP" or "DOWN"
    gap_fill_target: float


class NAS100GapBot(BaseFTMOBot):
    """
    NAS100 Gap Fill Strategy

    Trades gap fills at market open.
    """

    # Trading parameters
    OPEN_HOUR = 14  # 14:30 UTC (09:30 EST)
    OPEN_MINUTE = 30

    MIN_GAP_PERCENT = 0.4  # Minimum gap to trade
    MAX_GAP_PERCENT = 2.0  # Skip huge gaps
    GAP_FILL_TARGET_PERCENT = 0.7  # Target 70% fill

    STOP_LOSS_POINTS = 80.0  # Increased from 25 (backtest showed 40-50 point losses hitting SL)
    EXIT_HOUR = 16  # 16:30 UTC (11:30 EST)
    EXIT_MINUTE = 30

    def __init__(self, paper_mode: bool = True):
        config = BotConfig(
            bot_name="NAS100Gap",
            symbol="US100.cash",  # NASDAQ 100 (FTMO broker symbol)
            risk_percent=0.015,
            max_daily_trades=1,
            stop_loss_pips=self.STOP_LOSS_POINTS,
            take_profit_pips=50.0,  # Variable based on gap
            max_hold_hours=2.0,
            enabled=True,
            paper_mode=paper_mode,
        )
        super().__init__(config)

        self._gap_data: Optional[GapData] = None
        self._gap_date: Optional[datetime] = None
        self._gap_traded = False

        logger.info(
            f"[{self.config.bot_name}] Strategy: Gap fill "
            f"(min gap: {self.MIN_GAP_PERCENT}%, target: {self.GAP_FILL_TARGET_PERCENT*100:.0f}% fill)"
        )

    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze for gap fill opportunity."""
        now = datetime.now(timezone.utc)

        # Update gap data
        self._update_gap_data(market_data)

        if self._gap_data is None:
            return {"tradeable": False, "reason": "No gap detected"}

        # Check trading window
        if not self._is_trading_window(now):
            return {"tradeable": False, "reason": f"Outside trading window ({now.strftime('%H:%M')} UTC)"}

        # Check if already traded
        if self._gap_traded:
            return {"tradeable": False, "reason": "Already traded gap today"}

        # Get current price
        candles = market_data.get("candles", [])
        if not candles:
            return {"tradeable": False, "reason": "No candle data"}

        current_price = candles[-1].get("close", 0)

        # Check if gap is worth trading
        if abs(self._gap_data.gap_percent) < self.MIN_GAP_PERCENT:
            return {"tradeable": False, "reason": f"Gap too small ({self._gap_data.gap_percent:.2f}%)"}

        if abs(self._gap_data.gap_percent) > self.MAX_GAP_PERCENT:
            return {"tradeable": False, "reason": f"Gap too large ({self._gap_data.gap_percent:.2f}%)"}

        # Check if gap already filled
        if self._gap_data.gap_direction == "UP":
            if current_price <= self._gap_data.gap_fill_target:
                return {"tradeable": False, "reason": "Gap already filled"}
        else:
            if current_price >= self._gap_data.gap_fill_target:
                return {"tradeable": False, "reason": "Gap already filled"}

        return {
            "tradeable": True,
            "gap_data": self._gap_data,
            "current_price": current_price,
            "direction": "SELL" if self._gap_data.gap_direction == "UP" else "BUY",
        }

    def should_trade(self, analysis: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if gap fill trade should be taken."""
        if not analysis.get("tradeable"):
            return False, analysis.get("reason", "Not tradeable")

        gap_data = analysis.get("gap_data")
        return True, f"Gap {gap_data.gap_direction} {gap_data.gap_percent:+.2f}%"

    def generate_signal(self, analysis: Dict[str, Any], current_price: float) -> Optional[TradeSignal]:
        """Generate gap fill signal."""
        gap_data = analysis.get("gap_data")
        if gap_data is None:
            return None

        direction = analysis.get("direction")

        # Calculate TP as gap fill target
        if direction == "SELL":  # Gap up - selling to fill
            stop_loss = current_price + self.STOP_LOSS_POINTS
            take_profit = gap_data.gap_fill_target
        else:  # Gap down - buying to fill
            stop_loss = current_price - self.STOP_LOSS_POINTS
            take_profit = gap_data.gap_fill_target

        account_info = self.get_account_info()
        balance = account_info.get("balance", 15000)

        lot_size = self._calculate_index_lot_size(balance)

        self._gap_traded = True

        return TradeSignal(
            bot_name=self.config.bot_name,
            symbol=self.config.symbol,
            direction=direction,
            entry_price=current_price,
            stop_loss=round(stop_loss, 1),
            take_profit=round(take_profit, 1),
            lot_size=lot_size,
            reason=f"Gap fill: {gap_data.gap_direction} {gap_data.gap_percent:+.2f}%",
            confidence=0.70,
        )

    def _calculate_index_lot_size(self, balance: float) -> float:
        """Calculate lot size for NAS100."""
        risk_amount = balance * self.config.risk_percent
        point_value_per_lot = 1.0
        lot_size = risk_amount / (self.STOP_LOSS_POINTS * point_value_per_lot)
        return round(max(0.1, min(lot_size, 10.0)), 2)

    def _is_trading_window(self, now: datetime) -> bool:
        """Check if in trading window (after open, before exit time)."""
        start = now.replace(hour=self.OPEN_HOUR, minute=self.OPEN_MINUTE + 5, second=0)
        end = now.replace(hour=self.EXIT_HOUR, minute=self.EXIT_MINUTE, second=0)
        return start <= now <= end

    def _update_gap_data(self, market_data: Dict[str, Any]) -> None:
        """Detect gap at market open."""
        today = datetime.now(timezone.utc).date()

        # Reset at new day
        if self._gap_date != today:
            self._gap_data = None
            self._gap_traded = False
            self._gap_date = today

        # Only calculate once per day
        if self._gap_data is not None:
            return

        now = datetime.now(timezone.utc)
        # Wait until market is open
        if now.hour < self.OPEN_HOUR or (now.hour == self.OPEN_HOUR and now.minute < self.OPEN_MINUTE + 2):
            return

        candles = market_data.get("candles", [])
        d1_candles = market_data.get("d1_candles", [])

        # Try to get previous close from D1 candles
        previous_close = None
        if d1_candles and len(d1_candles) >= 2:
            previous_close = d1_candles[-2].get("close")

        # Get today's open from recent candles
        open_price = None
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
                if candle_time.hour == self.OPEN_HOUR and candle_time.minute <= self.OPEN_MINUTE + 2:
                    open_price = candle.get("open", candle.get("close"))
                    break

        if previous_close is None or open_price is None:
            # Fallback: use first and last candle of yesterday
            if candles and len(candles) >= 500:
                previous_close = candles[-500].get("close")
                open_price = candles[-1].get("open")

        if previous_close and open_price and previous_close > 0:
            gap_percent = ((open_price - previous_close) / previous_close) * 100
            gap_direction = "UP" if gap_percent > 0 else "DOWN"

            # Target 70% fill
            fill_amount = (open_price - previous_close) * self.GAP_FILL_TARGET_PERCENT
            gap_fill_target = open_price - fill_amount

            self._gap_data = GapData(
                previous_close=previous_close,
                open_price=open_price,
                gap_percent=gap_percent,
                gap_direction=gap_direction,
                gap_fill_target=gap_fill_target,
            )

            logger.info(
                f"[{self.config.bot_name}] Gap detected: {gap_direction} {gap_percent:+.2f}% "
                f"(prev: {previous_close:.1f}, open: {open_price:.1f}, target: {gap_fill_target:.1f})"
            )

    def check_and_trade(self) -> Optional[TradeSignal]:
        """Main scheduler method."""
        now = datetime.now(timezone.utc)
        if not self._is_trading_window(now):
            return None
        return self.run_cycle()


# Singleton
_nas100_bot: Optional[NAS100GapBot] = None
_bot_lock = threading.Lock()


def get_nas100_bot(paper_mode: bool = True) -> NAS100GapBot:
    """Get NAS100 gap bot singleton."""
    global _nas100_bot
    if _nas100_bot is None:
        with _bot_lock:
            if _nas100_bot is None:
                _nas100_bot = NAS100GapBot(paper_mode=paper_mode)
    return _nas100_bot
