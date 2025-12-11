"""
Gold NY Mean Reversion Bot (v2 - with trend filter)

Strategy:
- Calculate VWAP from 14:30 UTC (NY open) onwards
- Add VWAP bands at ±1 and ±2 standard deviations
- Entry when z-score >2.0 (extreme deviation)
- Skip first 15 min after NY open (chaotic price action)
- NEW: Trend filter - only trade reversions in direction of daily trend
- Target: Return to VWAP
- Stop: 75 pips
- Trade window: 18:00-21:00 UTC (13:00-16:00 EST)

VWAP BAND STRATEGY (based on MQL5 research):
- z-score >2.0: Extreme deviation, high probability reversion
- Entry at ±2 band, target VWAP or ±1 band
- Skip first 15 min after 14:30 UTC (chaotic)

v2 IMPROVEMENTS (2025-12-11):
- Added 50 SMA trend filter: only trade reversions toward the trend
- BUY only when price < 50 SMA (uptrend pullback)
- SELL only when price > 50 SMA (downtrend pullback)
- This filters out counter-trend reversions that often fail

Rationale:
- Gold tends to mean-revert after strong moves
- NY afternoon session often corrects morning excesses
- Trading with the trend increases win probability

Historical Performance (pre-v2):
- Win rate: 47.6% (21 trades)
- P&L: $184.63

Expected Performance (post-v2):
- Win rate: 55-60% (trading with trend only)
- Fewer trades but higher quality
"""

import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass
from loguru import logger

from .base_ftmo_bot import BaseFTMOBot, BotConfig, TradeSignal


@dataclass
class VWAPData:
    """VWAP calculation data with bands."""
    vwap: float
    current_price: float
    deviation_percent: float
    sample_count: int
    # VWAP bands (standard deviations)
    std_dev: float = 0.0
    upper_band_1: float = 0.0  # VWAP + 1 std
    lower_band_1: float = 0.0  # VWAP - 1 std
    upper_band_2: float = 0.0  # VWAP + 2 std
    lower_band_2: float = 0.0  # VWAP - 2 std
    z_score: float = 0.0       # How many std devs from VWAP


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

    # Skip first 15 min after NY open (chaotic price action)
    SKIP_FIRST_MINUTES = 15

    # Z-score based entry (more reliable than % deviation)
    MIN_ZSCORE = 2.0  # Entry at ±2 standard deviations
    MIN_DEVIATION_PERCENT = 0.7  # Lowered from 1.5% (z-score is primary)
    STOP_LOSS_PIPS = 75.0  # Increased from 50 (backtest showed 41-48 pip losses hitting SL)

    # v2: Trend filter (trade with trend only)
    SMA_PERIOD = 50  # 50-period SMA for trend direction
    USE_TREND_FILTER = True  # Enable/disable trend filter

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
        self._daily_sma: Optional[float] = None  # v2: Trend filter

        logger.info(
            f"[{self.config.bot_name}] Strategy: VWAP reversion "
            f"(min deviation: {self.MIN_DEVIATION_PERCENT}%, SL: {self.STOP_LOSS_PIPS} pips)"
        )

    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze for mean reversion opportunity using z-score and VWAP bands."""
        now = datetime.now(timezone.utc)

        # Check trading window
        if not self._is_trading_window(now):
            return {"tradeable": False, "reason": f"Outside trading window ({now.strftime('%H:%M')} UTC)"}

        # Skip first 15 min after NY open (chaotic price action)
        if self._is_chaotic_period(now):
            return {"tradeable": False, "reason": "Skipping first 15 min after NY open (chaotic)"}

        # Check if already traded
        if self._already_traded_today():
            return {"tradeable": False, "reason": "Already traded today"}

        # Calculate VWAP with bands
        vwap_data = self._calculate_vwap(market_data)

        if vwap_data is None:
            return {"tradeable": False, "reason": "Could not calculate VWAP"}

        # v2: Calculate SMA for trend filter
        candles = market_data.get("candles", [])
        self._calculate_sma(candles)

        # Primary: Check z-score for extreme deviation (>2 std devs)
        if abs(vwap_data.z_score) < self.MIN_ZSCORE:
            return {
                "tradeable": False,
                "reason": f"Z-score too low ({vwap_data.z_score:+.2f}, need >{self.MIN_ZSCORE})",
            }

        # Secondary: Check minimum % deviation
        if abs(vwap_data.deviation_percent) < self.MIN_DEVIATION_PERCENT:
            return {
                "tradeable": False,
                "reason": f"Deviation too small ({vwap_data.deviation_percent:+.2f}%)",
            }

        # Determine direction based on z-score
        if vwap_data.z_score > self.MIN_ZSCORE:
            direction = "SELL"  # Price above +2 std - sell to revert
        elif vwap_data.z_score < -self.MIN_ZSCORE:
            direction = "BUY"  # Price below -2 std - buy to revert
        else:
            return {"tradeable": False, "reason": "No clear deviation"}

        # v2: Apply trend filter - only trade reversions toward the trend
        if self.USE_TREND_FILTER and self._daily_sma is not None:
            trend_aligned = self._check_trend_alignment(direction, vwap_data.current_price)
            if not trend_aligned:
                return {
                    "tradeable": False,
                    "reason": f"Trend filter: {direction} blocked (price vs SMA mismatch)",
                    "z_score": vwap_data.z_score,
                    "sma": self._daily_sma,
                }

        return {
            "tradeable": True,
            "vwap": vwap_data.vwap,
            "current_price": vwap_data.current_price,
            "deviation_percent": vwap_data.deviation_percent,
            "z_score": vwap_data.z_score,
            "upper_band_1": vwap_data.upper_band_1,
            "lower_band_1": vwap_data.lower_band_1,
            "direction": direction,
            "sma": self._daily_sma,
        }

    def should_trade(self, analysis: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if reversion conditions are met."""
        if not analysis.get("tradeable"):
            return False, analysis.get("reason", "Not tradeable")

        z_score = analysis.get("z_score", 0)
        return True, f"VWAP reversion: z={z_score:+.2f}, {analysis['deviation_percent']:+.2f}% deviation"

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
        if not account_info:
            logger.warning(f"[{self.config.bot_name}] Skipping trade - account info unavailable")
            return None
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

    def _is_chaotic_period(self, now: datetime) -> bool:
        """Check if in first 15 min after NY open (14:30-14:45 UTC) - chaotic price action."""
        if now.hour == self.VWAP_START_HOUR and now.minute < (self.VWAP_START_MINUTE + self.SKIP_FIRST_MINUTES):
            return True
        return False

    def _already_traded_today(self) -> bool:
        """Check if already traded today."""
        today = datetime.now(timezone.utc).date()
        if hasattr(self, "_last_trade_date") and self._last_trade_date == today:
            return True
        self._traded_today = False
        return False

    def _calculate_vwap(self, market_data: Dict[str, Any]) -> Optional[VWAPData]:
        """
        Calculate VWAP from NY open with standard deviation bands.

        VWAP = Σ(Price × Volume) / Σ(Volume)
        Bands = VWAP ± n × StdDev(TypicalPrice)
        Z-score = (Price - VWAP) / StdDev
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

        # Calculate VWAP and collect typical prices for std dev
        total_pv = 0.0  # Price × Volume
        total_volume = 0.0
        typical_prices = []

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
                typical_prices.append(typical_price)

        if total_volume == 0 or len(typical_prices) < 10:
            return None

        vwap = total_pv / total_volume
        current_price = vwap_candles[-1].get("close", 0)

        if current_price == 0 or vwap == 0:
            return None

        # Calculate standard deviation of typical prices
        mean_price = sum(typical_prices) / len(typical_prices)
        variance = sum((p - mean_price) ** 2 for p in typical_prices) / len(typical_prices)
        std_dev = variance ** 0.5

        # Calculate bands and z-score
        if std_dev > 0:
            z_score = (current_price - vwap) / std_dev
        else:
            z_score = 0.0

        deviation_percent = ((current_price - vwap) / vwap) * 100

        return VWAPData(
            vwap=vwap,
            current_price=current_price,
            deviation_percent=deviation_percent,
            sample_count=len(vwap_candles),
            std_dev=std_dev,
            upper_band_1=vwap + std_dev,
            lower_band_1=vwap - std_dev,
            upper_band_2=vwap + (2 * std_dev),
            lower_band_2=vwap - (2 * std_dev),
            z_score=z_score,
        )

    def _calculate_sma(self, candles: List[Dict]) -> None:
        """Calculate N-period SMA for trend filter."""
        if len(candles) < self.SMA_PERIOD:
            self._daily_sma = None
            return

        # Use last N candles' close prices
        closes = [c.get("close", 0) for c in candles[-self.SMA_PERIOD:]]
        if all(c > 0 for c in closes):
            self._daily_sma = sum(closes) / len(closes)
        else:
            self._daily_sma = None

    def _check_trend_alignment(self, direction: str, current_price: float) -> bool:
        """
        Check if reversion direction aligns with the trend.

        For mean reversion:
        - BUY (price below VWAP): should be buying dips in an uptrend (price below SMA = pullback)
        - SELL (price above VWAP): should be selling rallies in a downtrend (price above SMA = rally)

        Logic:
        - BUY allowed when price is BELOW the SMA (pullback in uptrend, or near bottom)
        - SELL allowed when price is ABOVE the SMA (rally in downtrend, or near top)
        """
        if self._daily_sma is None:
            return True  # No filter if SMA not available

        if direction == "BUY":
            # BUY is for price below VWAP (oversold)
            # Allow BUY when price is below SMA (confirms oversold condition)
            is_aligned = current_price < self._daily_sma
            if not is_aligned:
                logger.info(
                    f"[{self.config.bot_name}] BUY blocked by trend filter "
                    f"(price {current_price:.2f} > SMA {self._daily_sma:.2f})"
                )
            return is_aligned

        else:  # SELL
            # SELL is for price above VWAP (overbought)
            # Allow SELL when price is above SMA (confirms overbought condition)
            is_aligned = current_price > self._daily_sma
            if not is_aligned:
                logger.info(
                    f"[{self.config.bot_name}] SELL blocked by trend filter "
                    f"(price {current_price:.2f} < SMA {self._daily_sma:.2f})"
                )
            return is_aligned

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
