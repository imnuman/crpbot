"""
High-Frequency Scalper Bot

Strategy:
- Monitor multiple pairs for momentum/volatility setups
- Enter on strong momentum candles (body > 60% of range)
- Quick scalps with tight stops (10-15 pips) and targets (15-25 pips)
- Trade frequently to generate L3 metalearning data
- Max trades per day: 10 (30 in turbo mode)

SESSION-AWARE TRADING (based on 126-trade analysis):
- London (08:00-16:00 UTC): Best session, 57.1% WR, $1,047 P&L → Normal trading
- NY (13:00-21:00 UTC): Good session, 53.3% WR, $654 P&L → Normal trading
- Overlap (13:00-16:00 UTC): Underperforms, 42.9% WR, $98 P&L → Higher threshold
- Asian (22:00-06:00 UTC): Limited data → Reduced activity

Expected Performance:
- Win rate: 52%
- Avg win: +18 pips
- Avg loss: -12 pips
- High trade volume for L3 data collection
"""

import os
import requests
import threading
import random
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass
from loguru import logger

from .base_ftmo_bot import BaseFTMOBot, BotConfig, TradeSignal


@dataclass
class MomentumSetup:
    """Momentum-based scalp setup."""
    symbol: str
    direction: str  # "BUY" or "SELL"
    momentum_score: float  # 0-1
    volatility: float
    entry_price: float
    reason: str


class HFScalperBot(BaseFTMOBot):
    """
    High-Frequency Scalper for L3 Data Generation

    Trades momentum setups on multiple pairs with tight stops.
    Designed to generate high trade volume for metalearning training.
    """

    # Strategy parameters
    STOP_LOSS_PIPS = 12.0
    TAKE_PROFIT_PIPS = 18.0
    MIN_MOMENTUM_BODY_RATIO = 0.60  # Candle body must be 60% of range
    MIN_VOLATILITY_PIPS = 5.0  # Minimum recent volatility
    MAX_SPREAD_PIPS = 3.0  # Maximum spread to enter

    # Trading hours (extended for more trades)
    TRADE_START_HOUR = 6  # 06:00 UTC
    TRADE_END_HOUR = 22  # 22:00 UTC

    # Tradeable symbols
    SYMBOLS = ["XAUUSD", "EURUSD", "GBPUSD"]

    # Cooldown between trades (minutes)
    MIN_TRADE_INTERVAL = 15

    # Session-aware trading adjustments (based on 126-trade analysis)
    # Overlap (13:00-16:00 UTC) has 42.9% WR vs 57.1% London - be more selective
    SESSION_OVERLAP_START = 13  # 13:00 UTC
    SESSION_OVERLAP_END = 16    # 16:00 UTC
    OVERLAP_MOMENTUM_MULTIPLIER = 1.3  # Require 30% higher momentum during overlap
    OVERLAP_COOLDOWN_MULTIPLIER = 1.5  # 50% longer cooldown during overlap

    def __init__(self, paper_mode: bool = True, turbo_mode: bool = False):
        # DISABLED 2025-12-11: 20% WR today (-$227) - buying in downtrend
        # TODO: Add trend filter before re-enabling
        config = BotConfig(
            bot_name="HFScalper",
            symbol="MULTI",  # Multi-symbol bot
            risk_percent=0.005,  # 0.5% risk per trade (Phase 9: halved from 1%)
            max_daily_trades=10,  # High frequency (30 in turbo)
            stop_loss_pips=self.STOP_LOSS_PIPS,
            take_profit_pips=self.TAKE_PROFIT_PIPS,
            max_hold_hours=1.0,  # Quick scalps
            enabled=False,  # DISABLED - bleeding money
            paper_mode=paper_mode,
            turbo_mode=turbo_mode,
        )
        super().__init__(config)

        self._last_trade_times: Dict[str, datetime] = {}
        self._current_symbol: Optional[str] = None

        # Apply turbo adjustments
        self._min_body_ratio = self.MIN_MOMENTUM_BODY_RATIO * config.get_turbo_multiplier()
        self._min_volatility = self.MIN_VOLATILITY_PIPS * config.get_turbo_multiplier()
        self._min_interval = self.MIN_TRADE_INTERVAL // 2 if turbo_mode else self.MIN_TRADE_INTERVAL

        logger.info(
            f"[{self.config.bot_name}] HF Scalper initialized "
            f"(symbols: {self.SYMBOLS}, max trades: {config.get_turbo_max_trades()}/day)"
            f"{' [TURBO MODE]' if turbo_mode else ''}"
        )

    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze multiple pairs for momentum setups.

        Args:
            market_data: Dict with "candles" per symbol

        Returns:
            Analysis with best setup if found
        """
        now = datetime.now(timezone.utc)

        # Check trading hours
        if not self._is_trading_hours(now):
            return {
                "tradeable": False,
                "reason": f"Outside trading hours ({now.strftime('%H:%M')} UTC)",
            }

        # Find best momentum setup across all symbols
        setups: List[MomentumSetup] = []

        for symbol in self.SYMBOLS:
            # Check cooldown for this symbol
            if not self._can_trade_symbol(symbol, now):
                continue

            # Get candles for symbol - MUST use symbol-specific candles only
            symbol_candles = market_data.get(f"{symbol}_candles", [])
            if not symbol_candles:
                # Skip if no candles for this specific symbol (don't use fallback to avoid price mixing)
                logger.debug(f"[{self.config.bot_name}] No candles for {symbol}, skipping")
                continue

            # Analyze momentum
            setup = self._analyze_momentum(symbol, symbol_candles)
            if setup:
                setups.append(setup)

        if not setups:
            return {
                "tradeable": False,
                "reason": "No momentum setups found",
            }

        # Select best setup by momentum score
        best_setup = max(setups, key=lambda s: s.momentum_score)

        return {
            "tradeable": True,
            "setup": best_setup,
            "all_setups": len(setups),
        }

    def should_trade(self, analysis: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if momentum setup is strong enough (session-aware)."""
        if not analysis.get("tradeable"):
            return False, analysis.get("reason", "Not tradeable")

        setup: MomentumSetup = analysis.get("setup")
        if not setup:
            return False, "No setup found"

        now = datetime.now(timezone.utc)
        session = self._get_session_name(now)

        # Require minimum momentum score (higher during overlap due to 42.9% WR)
        min_score = 0.5 * self.config.get_turbo_multiplier()
        if self._is_overlap_session(now):
            min_score *= self.OVERLAP_MOMENTUM_MULTIPLIER
            logger.debug(f"[{self.config.bot_name}] Overlap session - raised threshold to {min_score:.2f}")

        if setup.momentum_score < min_score:
            return False, f"Momentum too weak ({setup.momentum_score:.2f} < {min_score}) [{session}]"

        return True, f"{setup.direction} on {setup.symbol} (momentum: {setup.momentum_score:.2f}) [{session}]"

    def generate_signal(self, analysis: Dict[str, Any], current_price: float) -> Optional[TradeSignal]:
        """Generate scalp signal from momentum setup."""
        setup: MomentumSetup = analysis.get("setup")
        if not setup:
            return None

        # Update current symbol
        self._current_symbol = setup.symbol

        # Get pip value for symbol
        pip_value = self._get_pip_value(setup.symbol)

        # Calculate SL/TP based on volatility
        sl_pips = max(self.STOP_LOSS_PIPS, setup.volatility * 1.5)
        tp_pips = max(self.TAKE_PROFIT_PIPS, setup.volatility * 2.0)

        # Cap at reasonable levels
        sl_pips = min(sl_pips, 25.0)
        tp_pips = min(tp_pips, 40.0)

        if setup.direction == "BUY":
            stop_loss = setup.entry_price - (sl_pips * pip_value)
            take_profit = setup.entry_price + (tp_pips * pip_value)
        else:  # SELL
            stop_loss = setup.entry_price + (sl_pips * pip_value)
            take_profit = setup.entry_price - (tp_pips * pip_value)

        # Get account info for position sizing
        account_info = self.get_account_info()
        if not account_info:
            logger.warning(f"[{self.config.bot_name}] Skipping trade - account info unavailable")
            return None
        balance = account_info.get("balance", 15000)

        lot_size = self.calculate_lot_size(balance, sl_pips)

        signal = TradeSignal(
            bot_name=self.config.bot_name,
            symbol=setup.symbol,
            direction=setup.direction,
            entry_price=setup.entry_price,
            stop_loss=round(stop_loss, 5 if "USD" in setup.symbol else 2),
            take_profit=round(take_profit, 5 if "USD" in setup.symbol else 2),
            lot_size=lot_size,
            reason=setup.reason,
            confidence=setup.momentum_score,
        )

        # Record trade time for cooldown
        now = datetime.now(timezone.utc)
        self._last_trade_times[setup.symbol] = now
        session = self._get_session_name(now)

        logger.info(
            f"[{self.config.bot_name}] SCALP SIGNAL: {setup.direction} {setup.symbol} "
            f"@ {setup.entry_price:.5f} (SL: {sl_pips:.0f} pips, TP: {tp_pips:.0f} pips, "
            f"momentum: {setup.momentum_score:.2f}) [{session}]"
        )

        return signal

    def _is_trading_hours(self, now: datetime) -> bool:
        """Check if within trading hours."""
        return self.TRADE_START_HOUR <= now.hour < self.TRADE_END_HOUR

    def _is_overlap_session(self, now: datetime) -> bool:
        """Check if in London-NY overlap (13:00-16:00 UTC) - underperforming session."""
        return self.SESSION_OVERLAP_START <= now.hour < self.SESSION_OVERLAP_END

    def _get_session_name(self, now: datetime) -> str:
        """Get current session name for logging."""
        hour = now.hour
        if hour >= 22 or hour < 6:
            return "Asian"
        elif 8 <= hour < 13:
            return "London"
        elif 13 <= hour < 16:
            return "Overlap"
        elif 16 <= hour < 21:
            return "NY"
        else:
            return "Transition"

    def _can_trade_symbol(self, symbol: str, now: datetime) -> bool:
        """Check if symbol is off cooldown (extended during overlap session)."""
        last_trade = self._last_trade_times.get(symbol)
        if last_trade is None:
            return True

        elapsed = (now - last_trade).total_seconds() / 60

        # Apply longer cooldown during overlap session (42.9% WR vs 57.1% London)
        interval = self._min_interval
        if self._is_overlap_session(now):
            interval = self._min_interval * self.OVERLAP_COOLDOWN_MULTIPLIER

        return elapsed >= interval

    def _analyze_momentum(self, symbol: str, candles: List[Dict]) -> Optional[MomentumSetup]:
        """
        Analyze recent candles for momentum setup.

        Looks for:
        - Strong candle with body > 60% of range
        - Recent volatility above threshold
        - Clear direction
        """
        if len(candles) < 10:
            return None

        # Get recent candles
        recent = candles[-10:]
        last_candle = recent[-1]

        # Calculate candle metrics
        open_price = last_candle.get("open", 0)
        close_price = last_candle.get("close", 0)
        high_price = last_candle.get("high", 0)
        low_price = last_candle.get("low", 0)

        if high_price == low_price or open_price == 0:
            return None

        # Calculate body ratio
        body = abs(close_price - open_price)
        range_total = high_price - low_price
        body_ratio = body / range_total if range_total > 0 else 0

        # Calculate recent volatility (ATR-like)
        pip_value = self._get_pip_value(symbol)
        ranges = [(c.get("high", 0) - c.get("low", 0)) / pip_value for c in recent]
        avg_volatility = sum(ranges) / len(ranges) if ranges else 0

        # Check minimum volatility (with turbo adjustment)
        if avg_volatility < self._min_volatility:
            return None

        # Check body ratio threshold (with turbo adjustment)
        if body_ratio < self._min_body_ratio:
            return None

        # Determine direction
        if close_price > open_price:
            direction = "BUY"
            momentum_bias = (close_price - low_price) / range_total
        else:
            direction = "SELL"
            momentum_bias = (high_price - close_price) / range_total

        # Calculate momentum score
        momentum_score = (body_ratio + momentum_bias) / 2

        # Add some noise/variation for L3 learning
        momentum_score *= random.uniform(0.9, 1.1)
        momentum_score = min(1.0, max(0.0, momentum_score))

        return MomentumSetup(
            symbol=symbol,
            direction=direction,
            momentum_score=momentum_score,
            volatility=avg_volatility,
            entry_price=close_price,
            reason=f"Momentum scalp: body={body_ratio:.0%}, vol={avg_volatility:.1f} pips"
        )

    def _get_pip_value(self, symbol: str) -> float:
        """Get pip value for symbol."""
        if symbol.startswith("XAU"):
            return 0.10  # Gold: $0.10 = 1 pip
        elif symbol.startswith("US30") or symbol.startswith("NAS"):
            return 1.0  # Indices: 1 point = 1 pip
        else:
            return 0.0001  # Forex: 0.0001 = 1 pip

    def _fetch_market_data(self) -> Dict[str, Any]:
        """Fetch candles for all symbols."""
        data = {}

        for symbol in self.SYMBOLS:
            try:
                url = f"{self.MT5_EXECUTOR_URL}/candles/{symbol}"
                headers = {"Authorization": f"Bearer {self.MT5_API_SECRET}"}
                params = {"timeframe": "M5", "count": 50}

                response = requests.get(url, headers=headers, params=params, timeout=15)
                result = response.json()

                if result.get("success") and result.get("candles"):
                    data[f"{symbol}_candles"] = result["candles"]

            except Exception as e:
                logger.warning(f"[{self.config.bot_name}] Failed to fetch {symbol}: {e}")

        return data

    def check_and_trade(self) -> Optional[TradeSignal]:
        """Main method for scheduler - runs every cycle."""
        now = datetime.now(timezone.utc)

        if not self._is_trading_hours(now):
            return None

        return self.run_cycle()

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Override to use the current symbol being analyzed."""
        # Handle MULTI placeholder - use current symbol or default to first in list
        if symbol == "MULTI":
            actual_symbol = self._current_symbol or self.SYMBOLS[0]
        else:
            actual_symbol = self._current_symbol or symbol
        return super().get_current_price(actual_symbol)


# Singleton
_hf_scalper: Optional[HFScalperBot] = None
_bot_lock = threading.Lock()


def get_hf_scalper(paper_mode: bool = True, turbo_mode: bool = False) -> HFScalperBot:
    """Get HF Scalper bot singleton."""
    global _hf_scalper
    if _hf_scalper is None:
        with _bot_lock:
            if _hf_scalper is None:
                _hf_scalper = HFScalperBot(paper_mode=paper_mode, turbo_mode=turbo_mode)
    return _hf_scalper
