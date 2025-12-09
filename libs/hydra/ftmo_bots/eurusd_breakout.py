"""
EUR/USD Breakout Bot

Strategy:
- Monitor EUR/USD for daily high/low breakouts
- Enter when price breaks above yesterday's high (BUY)
- Enter when price breaks below yesterday's low (SELL)
- Confirm with 5-minute candle close beyond level
- Stop: 20 pips, Target: 40 pips
- Max trades per day: 2

Expected Performance:
- Win rate: 55%
- Avg win: +40 pips
- Avg loss: -20 pips
- Daily EV: +$172 (at 1.5% risk, $15k account)
"""

import os
import requests
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass
from loguru import logger

from .base_ftmo_bot import BaseFTMOBot, BotConfig, TradeSignal


@dataclass
class DailyLevels:
    """Yesterday's high/low levels."""
    high: float
    low: float
    date: datetime
    range_pips: float


class EURUSDBreakoutBot(BaseFTMOBot):
    """
    EUR/USD Daily S/R Breakout Strategy

    Trades breakouts of yesterday's high/low levels.
    Requires M5 candle close confirmation.
    """

    # Strategy parameters
    STOP_LOSS_PIPS = 20.0
    TAKE_PROFIT_PIPS = 40.0
    MIN_RANGE_PIPS = 30.0  # Minimum daily range to consider
    MAX_RANGE_PIPS = 150.0  # Skip if range too wide (volatile day)
    CONFIRMATION_PIPS = 3.0  # Must break by at least 3 pips

    # Trading hours (London + NY sessions)
    TRADE_START_HOUR = 7  # 07:00 UTC
    TRADE_END_HOUR = 20  # 20:00 UTC

    def __init__(self, paper_mode: bool = True):
        config = BotConfig(
            bot_name="EURUSDBreakout",
            symbol="EURUSD",
            risk_percent=0.015,  # 1.5% risk per trade
            max_daily_trades=2,
            stop_loss_pips=self.STOP_LOSS_PIPS,
            take_profit_pips=self.TAKE_PROFIT_PIPS,
            max_hold_hours=4.0,
            enabled=True,
            paper_mode=paper_mode,
        )
        super().__init__(config)

        self._daily_levels: Optional[DailyLevels] = None
        self._levels_date: Optional[datetime] = None
        self._breakout_triggered: Dict[str, bool] = {"high": False, "low": False}

        logger.info(
            f"[{self.config.bot_name}] Strategy: "
            f"Daily high/low breakouts (SL: {self.STOP_LOSS_PIPS} pips, TP: {self.TAKE_PROFIT_PIPS} pips)"
        )

    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze for breakout conditions.

        Args:
            market_data: Dict with "candles" (D1 and M5)

        Returns:
            Analysis result
        """
        now = datetime.now(timezone.utc)

        # Check trading hours
        if not self._is_trading_hours(now):
            return {
                "tradeable": False,
                "reason": f"Outside trading hours ({now.strftime('%H:%M')} UTC)",
            }

        # Get/update daily levels
        self._update_daily_levels(market_data)

        if self._daily_levels is None:
            return {
                "tradeable": False,
                "reason": "Could not determine daily levels",
            }

        # Check range is reasonable
        if self._daily_levels.range_pips < self.MIN_RANGE_PIPS:
            return {
                "tradeable": False,
                "reason": f"Daily range too small ({self._daily_levels.range_pips:.0f} pips)",
            }

        if self._daily_levels.range_pips > self.MAX_RANGE_PIPS:
            return {
                "tradeable": False,
                "reason": f"Daily range too wide ({self._daily_levels.range_pips:.0f} pips)",
            }

        # Get current price from M5 candles
        m5_candles = market_data.get("m5_candles", market_data.get("candles", []))
        if not m5_candles:
            return {
                "tradeable": False,
                "reason": "No M5 candle data",
            }

        current_candle = m5_candles[-1]
        current_price = current_candle.get("close", 0)
        candle_high = current_candle.get("high", 0)
        candle_low = current_candle.get("low", 0)

        # Check for breakout
        breakout_direction = None
        breakout_level = None

        # High breakout (BUY)
        if not self._breakout_triggered["high"]:
            if candle_low > self._daily_levels.high + (self.CONFIRMATION_PIPS * 0.0001):
                breakout_direction = "BUY"
                breakout_level = self._daily_levels.high

        # Low breakout (SELL)
        if not self._breakout_triggered["low"]:
            if candle_high < self._daily_levels.low - (self.CONFIRMATION_PIPS * 0.0001):
                breakout_direction = "SELL"
                breakout_level = self._daily_levels.low

        return {
            "tradeable": True,
            "daily_high": self._daily_levels.high,
            "daily_low": self._daily_levels.low,
            "daily_range": self._daily_levels.range_pips,
            "current_price": current_price,
            "breakout_direction": breakout_direction,
            "breakout_level": breakout_level,
        }

    def should_trade(self, analysis: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if breakout conditions are met."""
        if not analysis.get("tradeable"):
            return False, analysis.get("reason", "Not tradeable")

        breakout_direction = analysis.get("breakout_direction")
        if breakout_direction is None:
            return False, "No breakout detected"

        return True, f"{breakout_direction} breakout at {analysis.get('breakout_level'):.5f}"

    def generate_signal(self, analysis: Dict[str, Any], current_price: float) -> Optional[TradeSignal]:
        """Generate breakout signal."""
        direction = analysis.get("breakout_direction")
        if direction is None:
            return None

        breakout_level = analysis.get("breakout_level")

        # Calculate SL/TP
        pip_value = 0.0001  # EUR/USD pip

        if direction == "BUY":
            stop_loss = current_price - (self.STOP_LOSS_PIPS * pip_value)
            take_profit = current_price + (self.TAKE_PROFIT_PIPS * pip_value)
            self._breakout_triggered["high"] = True
        else:  # SELL
            stop_loss = current_price + (self.STOP_LOSS_PIPS * pip_value)
            take_profit = current_price - (self.TAKE_PROFIT_PIPS * pip_value)
            self._breakout_triggered["low"] = True

        # Get account info for position sizing
        account_info = self.get_account_info()
        balance = account_info.get("balance", 15000)

        lot_size = self.calculate_lot_size(balance, self.STOP_LOSS_PIPS)

        signal = TradeSignal(
            bot_name=self.config.bot_name,
            symbol=self.config.symbol,
            direction=direction,
            entry_price=current_price,
            stop_loss=round(stop_loss, 5),
            take_profit=round(take_profit, 5),
            lot_size=lot_size,
            reason=f"Daily {direction.lower()} breakout at {breakout_level:.5f}",
            confidence=0.65,
        )

        logger.info(
            f"[{self.config.bot_name}] Signal: {direction} @ {current_price:.5f} "
            f"(level: {breakout_level:.5f}, SL: {stop_loss:.5f}, TP: {take_profit:.5f})"
        )

        return signal

    def _is_trading_hours(self, now: datetime) -> bool:
        """Check if within trading hours (London + NY sessions)."""
        return self.TRADE_START_HOUR <= now.hour < self.TRADE_END_HOUR

    def _update_daily_levels(self, market_data: Dict[str, Any]) -> None:
        """Update yesterday's high/low levels."""
        today = datetime.now(timezone.utc).date()

        # Reset breakout flags at start of new day
        if self._levels_date != today:
            self._breakout_triggered = {"high": False, "low": False}

        # Only update once per day
        if self._levels_date == today and self._daily_levels is not None:
            return

        # Get daily candles
        d1_candles = market_data.get("d1_candles", [])

        if d1_candles and len(d1_candles) >= 2:
            # Yesterday's candle (second to last in the list)
            yesterday = d1_candles[-2]
            high = yesterday.get("high", 0)
            low = yesterday.get("low", 0)

            if high > 0 and low > 0:
                range_pips = (high - low) / 0.0001
                self._daily_levels = DailyLevels(
                    high=high,
                    low=low,
                    date=datetime.now(timezone.utc),
                    range_pips=range_pips,
                )
                self._levels_date = today

                logger.info(
                    f"[{self.config.bot_name}] Daily levels updated: "
                    f"High: {high:.5f}, Low: {low:.5f}, Range: {range_pips:.0f} pips"
                )
                return

        # Fallback: try to calculate from M5 candles (last 24 hours)
        m5_candles = market_data.get("m5_candles", market_data.get("candles", []))
        if m5_candles and len(m5_candles) >= 288:  # 24 hours of M5 candles
            yesterday_candles = m5_candles[-576:-288]  # 24-48 hours ago
            if yesterday_candles:
                high = max(c.get("high", 0) for c in yesterday_candles)
                low = min(c.get("low", float("inf")) for c in yesterday_candles)

                if high > 0 and low < float("inf"):
                    range_pips = (high - low) / 0.0001
                    self._daily_levels = DailyLevels(
                        high=high,
                        low=low,
                        date=datetime.now(timezone.utc),
                        range_pips=range_pips,
                    )
                    self._levels_date = today

                    logger.info(
                        f"[{self.config.bot_name}] Daily levels (from M5): "
                        f"High: {high:.5f}, Low: {low:.5f}"
                    )

    def _fetch_market_data(self) -> Dict[str, Any]:
        """Fetch EUR/USD candles."""
        try:
            # Fetch M5 candles
            url = f"{self.MT5_EXECUTOR_URL}/candles/{self.config.symbol}"
            headers = {"Authorization": f"Bearer {self.MT5_API_SECRET}"}

            response = requests.get(
                url, headers=headers,
                params={"timeframe": "M5", "count": 600},
                timeout=30
            )
            m5_data = response.json()

            # Fetch D1 candles
            response_d1 = requests.get(
                url, headers=headers,
                params={"timeframe": "D1", "count": 5},
                timeout=30
            )
            d1_data = response_d1.json()

            return {
                "m5_candles": m5_data.get("candles", []),
                "d1_candles": d1_data.get("candles", []),
                "candles": m5_data.get("candles", []),
            }

        except Exception as e:
            logger.error(f"[{self.config.bot_name}] Failed to fetch data: {e}")
            return {}

    def check_and_trade(self) -> Optional[TradeSignal]:
        """Main method for scheduler."""
        now = datetime.now(timezone.utc)

        if not self._is_trading_hours(now):
            return None

        return self.run_cycle()


# Singleton
_eurusd_bot: Optional[EURUSDBreakoutBot] = None
_bot_lock = threading.Lock()


def get_eurusd_bot(paper_mode: bool = True) -> EURUSDBreakoutBot:
    """Get EUR/USD breakout bot singleton."""
    global _eurusd_bot
    if _eurusd_bot is None:
        with _bot_lock:
            if _eurusd_bot is None:
                _eurusd_bot = EURUSDBreakoutBot(paper_mode=paper_mode)
    return _eurusd_bot
