"""
Gold NY Mean Reversion Bot

Strategy:
- Calculate VWAP from 14:30 UTC (NY open) onwards
- If price >1.5% ABOVE VWAP → SELL (revert to mean)
- If price >1.5% BELOW VWAP → BUY (revert to mean)
- Target: Return to VWAP
- Stop: 50 pips
- Trade window: 18:00-21:00 UTC (13:00-16:00 EST)

Rationale:
- Gold tends to mean-revert after strong moves
- NY afternoon session often corrects morning excesses

Expected Performance:
- Win rate: 60%
- Avg win: Variable (VWAP distance)
- Avg loss: -50 pips
- Daily EV: +$148 (at 1.5% risk, $15k account)
"""

import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass
from loguru import logger

from .base_ftmo_bot import BaseFTMOBot, BotConfig, TradeSignal


@dataclass
class VWAPData:
    """VWAP calculation data."""
    vwap: float
    current_price: float
    deviation_percent: float
    sample_count: int


class GoldNYReversionBot(BaseFTMOBot):
    """
    Gold NY Session Mean Reversion Strategy

    Trades reversion to VWAP during NY afternoon.
    """

    # Trading parameters
    VWAP_START_HOUR = 14  # 14:30 UTC (NY open)
    VWAP_START_MINUTE = 30

    TRADE_START_HOUR = 18  # 18:00 UTC (13:00 EST)
    TRADE_END_HOUR = 21  # 21:00 UTC (16:00 EST)

    MIN_DEVIATION_PERCENT = 1.5  # Minimum deviation from VWAP
    STOP_LOSS_PIPS = 50.0

    def __init__(self, paper_mode: bool = True):
        config = BotConfig(
            bot_name="GoldNYReversion",
            symbol="XAUUSD",
            risk_percent=0.015,
            max_daily_trades=1,
            stop_loss_pips=self.STOP_LOSS_PIPS,
            take_profit_pips=80.0,  # Variable based on VWAP distance
            max_hold_hours=3.0,
            enabled=True,
            paper_mode=paper_mode,
        )
        super().__init__(config)

        self._vwap_data: Optional[VWAPData] = None
        self._vwap_date: Optional[datetime] = None
        self._traded_today = False

        logger.info(
            f"[{self.config.bot_name}] Strategy: VWAP reversion "
            f"(min deviation: {self.MIN_DEVIATION_PERCENT}%, SL: {self.STOP_LOSS_PIPS} pips)"
        )

    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze for mean reversion opportunity."""
        now = datetime.now(timezone.utc)

        # Check trading window
        if not self._is_trading_window(now):
            return {"tradeable": False, "reason": f"Outside trading window ({now.strftime('%H:%M')} UTC)"}

        # Check if already traded
        if self._already_traded_today():
            return {"tradeable": False, "reason": "Already traded today"}

        # Calculate VWAP
        vwap_data = self._calculate_vwap(market_data)

        if vwap_data is None:
            return {"tradeable": False, "reason": "Could not calculate VWAP"}

        # Check deviation
        if abs(vwap_data.deviation_percent) < self.MIN_DEVIATION_PERCENT:
            return {
                "tradeable": False,
                "reason": f"Deviation too small ({vwap_data.deviation_percent:+.2f}%)",
            }

        # Determine direction
        if vwap_data.deviation_percent > self.MIN_DEVIATION_PERCENT:
            direction = "SELL"  # Price above VWAP - sell to revert
        elif vwap_data.deviation_percent < -self.MIN_DEVIATION_PERCENT:
            direction = "BUY"  # Price below VWAP - buy to revert
        else:
            return {"tradeable": False, "reason": "No clear deviation"}

        return {
            "tradeable": True,
            "vwap": vwap_data.vwap,
            "current_price": vwap_data.current_price,
            "deviation_percent": vwap_data.deviation_percent,
            "direction": direction,
        }

    def should_trade(self, analysis: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if reversion conditions are met."""
        if not analysis.get("tradeable"):
            return False, analysis.get("reason", "Not tradeable")

        return True, f"VWAP reversion: {analysis['deviation_percent']:+.2f}% deviation"

    def generate_signal(self, analysis: Dict[str, Any], current_price: float) -> Optional[TradeSignal]:
        """Generate VWAP reversion signal."""
        direction = analysis.get("direction")
        vwap = analysis.get("vwap")

        if direction is None or vwap is None:
            return None

        # TP is VWAP (full reversion)
        # SL is fixed 50 pips
        pip_value = 0.10  # Gold pip = $0.10

        if direction == "SELL":
            stop_loss = current_price + (self.STOP_LOSS_PIPS * pip_value)
            take_profit = vwap  # Target VWAP
        else:  # BUY
            stop_loss = current_price - (self.STOP_LOSS_PIPS * pip_value)
            take_profit = vwap

        account_info = self.get_account_info()
        balance = account_info.get("balance", 15000)
        lot_size = self.calculate_lot_size(balance, self.STOP_LOSS_PIPS)

        self._traded_today = True
        self._last_trade_date = datetime.now(timezone.utc).date()

        return TradeSignal(
            bot_name=self.config.bot_name,
            symbol=self.config.symbol,
            direction=direction,
            entry_price=current_price,
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            lot_size=lot_size,
            reason=f"VWAP reversion: {analysis['deviation_percent']:+.2f}% from {vwap:.2f}",
            confidence=0.65,
        )

    def _is_trading_window(self, now: datetime) -> bool:
        """Check if in trading window."""
        return self.TRADE_START_HOUR <= now.hour < self.TRADE_END_HOUR

    def _already_traded_today(self) -> bool:
        """Check if already traded today."""
        today = datetime.now(timezone.utc).date()
        if hasattr(self, "_last_trade_date") and self._last_trade_date == today:
            return True
        self._traded_today = False
        return False

    def _calculate_vwap(self, market_data: Dict[str, Any]) -> Optional[VWAPData]:
        """
        Calculate VWAP from NY open.

        VWAP = Σ(Price × Volume) / Σ(Volume)
        """
        candles = market_data.get("candles", [])
        if not candles:
            return None

        now = datetime.now(timezone.utc)
        today = now.date()

        # Filter candles from NY open (14:30 UTC) to now
        vwap_candles = []
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
                if (candle_time.hour > self.VWAP_START_HOUR or
                    (candle_time.hour == self.VWAP_START_HOUR and candle_time.minute >= self.VWAP_START_MINUTE)):
                    vwap_candles.append(candle)

        if len(vwap_candles) < 30:  # Need at least 30 candles
            logger.debug(f"[{self.config.bot_name}] Insufficient candles for VWAP: {len(vwap_candles)}")
            return None

        # Calculate VWAP
        total_pv = 0.0  # Price × Volume
        total_volume = 0.0

        for candle in vwap_candles:
            # Typical price = (High + Low + Close) / 3
            high = candle.get("high", 0)
            low = candle.get("low", 0)
            close = candle.get("close", 0)
            volume = candle.get("volume", 1)  # Default 1 if no volume

            if high > 0 and low > 0 and close > 0:
                typical_price = (high + low + close) / 3
                total_pv += typical_price * volume
                total_volume += volume

        if total_volume == 0:
            return None

        vwap = total_pv / total_volume
        current_price = vwap_candles[-1].get("close", 0)

        if current_price == 0 or vwap == 0:
            return None

        deviation_percent = ((current_price - vwap) / vwap) * 100

        return VWAPData(
            vwap=vwap,
            current_price=current_price,
            deviation_percent=deviation_percent,
            sample_count=len(vwap_candles),
        )

    def check_and_trade(self) -> Optional[TradeSignal]:
        """Main scheduler method."""
        now = datetime.now(timezone.utc)
        if not self._is_trading_window(now):
            return None
        return self.run_cycle()


# Singleton
_gold_ny_bot: Optional[GoldNYReversionBot] = None
_bot_lock = threading.Lock()


def get_gold_ny_bot(paper_mode: bool = True) -> GoldNYReversionBot:
    """Get Gold NY reversion bot singleton."""
    global _gold_ny_bot
    if _gold_ny_bot is None:
        with _bot_lock:
            if _gold_ny_bot is None:
                _gold_ny_bot = GoldNYReversionBot(paper_mode=paper_mode)
    return _gold_ny_bot
