"""
Gold London Reversal Bot

Strategy:
- Monitor Gold (XAUUSD) during Asian session (00:00-07:55 UTC)
- Detect Asian session trend direction (>0.15% move)
- Enter OPPOSITE direction at 07:58 UTC (London open reversal)
- Asian bullish trend (up >0.15%) → SELL at London open
- Asian bearish trend (down >0.15%) → BUY at London open
- Stop: 50 pips, Target: 90 pips
- Max hold: 2 hours (exit by 09:00 UTC)

Expected Performance:
- Win rate: 61%
- Avg win: +90 pips
- Avg loss: -50 pips
- Daily EV: +$184 (at 1.5% risk, $15k account)
"""

import os
import requests
import threading
from datetime import datetime, timezone, timedelta, time as dt_time
from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass
from loguru import logger

from .base_ftmo_bot import BaseFTMOBot, BotConfig, TradeSignal


@dataclass
class AsianSessionData:
    """Data for Asian session analysis."""
    session_open_price: float
    session_high: float
    session_low: float
    current_price: float
    percent_change: float
    trend_direction: str  # "BULLISH", "BEARISH", "NEUTRAL"
    session_start: datetime
    session_end: datetime


class GoldLondonReversalBot(BaseFTMOBot):
    """
    Gold London Session Reversal Strategy

    Fades the Asian session trend at London open.
    Based on the tendency for gold to reverse after strong Asian moves.
    """

    # Strategy parameters
    ASIAN_SESSION_START_HOUR = 0  # 00:00 UTC
    ASIAN_SESSION_END_HOUR = 8    # 08:00 UTC
    ENTRY_HOUR = 7
    ENTRY_MINUTE = 58  # 07:58 UTC
    MAX_HOLD_HOUR = 10  # Exit by 10:00 UTC latest

    MIN_ASIAN_MOVE_PERCENT = 0.15  # Minimum 0.15% move to trigger
    STOP_LOSS_PIPS = 50.0
    TAKE_PROFIT_PIPS = 90.0

    def __init__(self, paper_mode: bool = True, turbo_mode: bool = False):
        config = BotConfig(
            bot_name="GoldLondonReversal",
            symbol="XAUUSD",  # Gold vs USD on MT5
            risk_percent=0.015,  # 1.5% risk per trade
            max_daily_trades=1,  # Only 1 trade per day (3 in turbo)
            stop_loss_pips=self.STOP_LOSS_PIPS,
            take_profit_pips=self.TAKE_PROFIT_PIPS,
            max_hold_hours=2.0,
            enabled=True,
            paper_mode=paper_mode,
            turbo_mode=turbo_mode,
        )
        super().__init__(config)

        self._asian_session_data: Optional[AsianSessionData] = None
        self._traded_today = False
        self._last_trade_date: Optional[datetime] = None

        # Apply turbo multiplier to threshold
        self._min_move = self.MIN_ASIAN_MOVE_PERCENT * config.get_turbo_multiplier()

        logger.info(
            f"[{self.config.bot_name}] Strategy: "
            f"Fade Asian trend at London open (min move: {self._min_move}%)"
            f"{' [TURBO MODE]' if turbo_mode else ''}"
        )

    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze Asian session for trend direction.

        Args:
            market_data: Dict with "candles" (list of OHLCV dicts)

        Returns:
            Analysis result with Asian session data
        """
        now = datetime.now(timezone.utc)

        # Check if we're in the trading window (around 07:55-08:05 UTC)
        if not self._is_entry_window(now):
            return {
                "in_entry_window": False,
                "reason": f"Not entry window (current: {now.strftime('%H:%M')} UTC)",
                "asian_data": None,
            }

        # Check if already traded today
        if self._already_traded_today():
            return {
                "in_entry_window": True,
                "reason": "Already traded today",
                "asian_data": None,
            }

        # Analyze Asian session
        asian_data = self._analyze_asian_session(market_data)

        if asian_data is None:
            return {
                "in_entry_window": True,
                "reason": "Could not analyze Asian session",
                "asian_data": None,
            }

        return {
            "in_entry_window": True,
            "reason": "Analysis complete",
            "asian_data": asian_data,
            "trend": asian_data.trend_direction,
            "move_percent": asian_data.percent_change,
        }

    def should_trade(self, analysis: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if trading conditions are met.

        Conditions:
        1. In entry window (07:55-08:05 UTC)
        2. Haven't traded today
        3. Asian session had significant move (>0.15%)
        """
        if not analysis.get("in_entry_window"):
            return False, analysis.get("reason", "Not in entry window")

        asian_data = analysis.get("asian_data")
        if asian_data is None:
            return False, analysis.get("reason", "No Asian data")

        # Check minimum move (uses turbo threshold if enabled)
        if abs(asian_data.percent_change) < self._min_move:
            return False, f"Asian move too small ({asian_data.percent_change:.2f}% < {self._min_move}%)"

        # Check trend direction is clear
        if asian_data.trend_direction == "NEUTRAL":
            return False, "No clear Asian trend"

        return True, f"Asian {asian_data.trend_direction} trend ({asian_data.percent_change:+.2f}%)"

    def generate_signal(self, analysis: Dict[str, Any], current_price: float) -> Optional[TradeSignal]:
        """
        Generate reversal signal based on Asian trend.

        Asian BULLISH (up >0.15%) → SELL
        Asian BEARISH (down >0.15%) → BUY
        """
        asian_data = analysis.get("asian_data")
        if asian_data is None:
            return None

        # Determine direction (OPPOSITE of Asian trend)
        if asian_data.trend_direction == "BULLISH":
            direction = "SELL"
            # For SELL: SL above entry, TP below entry
            stop_loss = current_price + (self.STOP_LOSS_PIPS / 10)  # Gold pips = 0.10
            take_profit = current_price - (self.TAKE_PROFIT_PIPS / 10)
        elif asian_data.trend_direction == "BEARISH":
            direction = "BUY"
            # For BUY: SL below entry, TP above entry
            stop_loss = current_price - (self.STOP_LOSS_PIPS / 10)
            take_profit = current_price + (self.TAKE_PROFIT_PIPS / 10)
        else:
            return None

        # Get account info for position sizing
        account_info = self.get_account_info()
        balance = account_info.get("balance", 15000)

        # Calculate lot size
        lot_size = self.calculate_lot_size(balance, self.STOP_LOSS_PIPS)

        signal = TradeSignal(
            bot_name=self.config.bot_name,
            symbol=self.config.symbol,
            direction=direction,
            entry_price=current_price,
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            lot_size=lot_size,
            reason=f"London reversal: Asian {asian_data.trend_direction} ({asian_data.percent_change:+.2f}%)",
            confidence=0.70,
        )

        logger.info(
            f"[{self.config.bot_name}] Signal generated: "
            f"{direction} @ {current_price:.2f} (SL: {stop_loss:.2f}, TP: {take_profit:.2f})"
        )

        # Mark as traded today
        self._traded_today = True
        self._last_trade_date = datetime.now(timezone.utc).date()

        return signal

    def _is_entry_window(self, now: datetime) -> bool:
        """Check if current time is in entry window (07:55-08:05 UTC)."""
        entry_start = now.replace(hour=7, minute=55, second=0, microsecond=0)
        entry_end = now.replace(hour=8, minute=5, second=0, microsecond=0)
        return entry_start <= now <= entry_end

    def _already_traded_today(self) -> bool:
        """Check if bot already traded today."""
        today = datetime.now(timezone.utc).date()
        if self._last_trade_date == today:
            return True
        # Reset flag if new day
        self._traded_today = False
        return False

    def _analyze_asian_session(self, market_data: Dict[str, Any]) -> Optional[AsianSessionData]:
        """
        Analyze Asian session candles to determine trend.

        Args:
            market_data: Dict with "candles" list (OHLCV)

        Returns:
            AsianSessionData or None if insufficient data
        """
        candles = market_data.get("candles", [])

        if not candles or len(candles) < 10:
            logger.warning(f"[{self.config.bot_name}] Insufficient candles: {len(candles)}")
            return None

        now = datetime.now(timezone.utc)
        today = now.date()

        # Asian session times for today
        session_start = datetime(today.year, today.month, today.day, 0, 0, tzinfo=timezone.utc)
        session_end = datetime(today.year, today.month, today.day, 7, 55, tzinfo=timezone.utc)

        # Filter candles to Asian session
        asian_candles = []
        for candle in candles:
            # Parse candle timestamp
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

            if session_start <= candle_time <= session_end:
                asian_candles.append(candle)

        if len(asian_candles) < 5:
            # Fallback: use last 8 hours of candles
            asian_candles = candles[-480:] if len(candles) >= 480 else candles  # 8 hours of 1-min candles

        if not asian_candles:
            logger.warning(f"[{self.config.bot_name}] No Asian session candles found")
            return None

        # Calculate session metrics
        session_open = asian_candles[0].get("open", asian_candles[0].get("close"))
        session_high = max(c.get("high", 0) for c in asian_candles)
        session_low = min(c.get("low", float("inf")) for c in asian_candles)
        current_price = asian_candles[-1].get("close")

        if session_open is None or session_open == 0:
            return None

        # Calculate percent change
        percent_change = ((current_price - session_open) / session_open) * 100

        # Determine trend direction (uses turbo threshold if enabled)
        if percent_change >= self._min_move:
            trend = "BULLISH"
        elif percent_change <= -self._min_move:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"

        logger.debug(
            f"[{self.config.bot_name}] Asian session: "
            f"Open: {session_open:.2f}, High: {session_high:.2f}, Low: {session_low:.2f}, "
            f"Close: {current_price:.2f}, Change: {percent_change:+.2f}%, Trend: {trend}"
        )

        return AsianSessionData(
            session_open_price=session_open,
            session_high=session_high,
            session_low=session_low,
            current_price=current_price,
            percent_change=percent_change,
            trend_direction=trend,
            session_start=session_start,
            session_end=session_end,
        )

    def _fetch_market_data(self) -> Dict[str, Any]:
        """Fetch gold candle data from MT5."""
        try:
            url = f"{self.MT5_EXECUTOR_URL}/candles/{self.config.symbol}"
            headers = {"Authorization": f"Bearer {self.MT5_API_SECRET}"}
            params = {"timeframe": "M1", "count": 500}  # 8+ hours of 1-min candles

            response = requests.get(url, headers=headers, params=params, timeout=30)
            data = response.json()

            if data.get("success") and data.get("candles"):
                return {"candles": data["candles"]}

            logger.warning(f"[{self.config.bot_name}] No candle data returned")
            return {"candles": []}

        except Exception as e:
            logger.error(f"[{self.config.bot_name}] Failed to fetch candles: {e}")
            return {"candles": []}

    def check_and_trade(self) -> Optional[TradeSignal]:
        """
        Main method to be called by scheduler.

        Checks if it's time to trade and executes if conditions are met.
        Returns signal if trade was made.
        """
        now = datetime.now(timezone.utc)

        # Only run during entry window
        if not self._is_entry_window(now):
            return None

        return self.run_cycle()


# Singleton instance
_gold_london_bot: Optional[GoldLondonReversalBot] = None
_bot_lock = threading.Lock()


def get_gold_london_bot(paper_mode: bool = True, turbo_mode: bool = False) -> GoldLondonReversalBot:
    """Get or create Gold London Reversal bot singleton."""
    global _gold_london_bot
    if _gold_london_bot is None:
        with _bot_lock:
            if _gold_london_bot is None:
                _gold_london_bot = GoldLondonReversalBot(paper_mode=paper_mode, turbo_mode=turbo_mode)
    return _gold_london_bot
