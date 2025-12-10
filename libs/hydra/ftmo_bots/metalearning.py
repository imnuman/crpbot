"""
FTMO Metalearning Module

L1: Adaptive Position Sizing - Adjusts lot sizes based on recent performance
L2: Volatility Regime Detection - Adjusts parameters based on market conditions

This module enables bots to learn and adapt in real-time without retraining.
"""

import os
import json
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
from loguru import logger
import numpy as np


# =============================================================================
# L1: ADAPTIVE POSITION SIZING
# =============================================================================

@dataclass
class TradeResult:
    """Record of a completed trade."""
    bot_name: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    lot_size: float
    pnl_pips: float
    pnl_dollars: float
    is_win: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class PerformanceWindow:
    """Performance metrics over a rolling window."""
    trades: List[TradeResult] = field(default_factory=list)
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.5
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 1.0
    kelly_fraction: float = 0.02  # Base 2% risk


class AdaptivePositionSizer:
    """
    L1: Adaptive Position Sizing based on recent performance.

    Core Logic:
    - After wins: Gradually increase position size (up to 2x base)
    - After losses: Quickly decrease position size (down to 0.5x base)
    - Uses modified Kelly Criterion with safety caps
    - Tracks per-bot performance separately

    FTMO Compliance:
    - Never exceeds 2% risk per trade
    - Reduces risk after drawdowns
    - Conservative during losing streaks
    """

    # Configuration
    BASE_RISK_PERCENT = 0.015  # 1.5% base risk
    MIN_RISK_PERCENT = 0.005   # 0.5% minimum (very conservative)
    MAX_RISK_PERCENT = 0.02    # 2.0% maximum (FTMO limit)

    WINDOW_SIZE = 20           # Number of trades to consider
    WIN_STREAK_BOOST = 0.1     # 10% boost per consecutive win
    LOSS_STREAK_CUT = 0.15     # 15% cut per consecutive loss
    MAX_STREAK_EFFECT = 3      # Max streak multiplier effect

    def __init__(self, data_dir: str = "data/hydra/ftmo"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._performance: Dict[str, PerformanceWindow] = {}
        self._streak: Dict[str, int] = {}  # Positive = wins, negative = losses
        self._last_sizes: Dict[str, float] = {}  # Last calculated multiplier

        self._load_history()
        logger.info("[L1 AdaptivePositionSizer] Initialized")

    def _load_history(self):
        """Load trade history from disk."""
        history_file = self.data_dir / "trade_history.jsonl"
        if not history_file.exists():
            return

        try:
            with open(history_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    result = TradeResult(
                        bot_name=data["bot_name"],
                        symbol=data["symbol"],
                        direction=data["direction"],
                        entry_price=data["entry_price"],
                        exit_price=data["exit_price"],
                        lot_size=data["lot_size"],
                        pnl_pips=data["pnl_pips"],
                        pnl_dollars=data["pnl_dollars"],
                        is_win=data["is_win"],
                        timestamp=datetime.fromisoformat(data["timestamp"])
                    )
                    self._add_to_window(result)
            logger.info(f"[L1] Loaded {sum(len(p.trades) for p in self._performance.values())} trades from history")
        except Exception as e:
            logger.error(f"[L1] Error loading history: {e}")

    def _save_trade(self, result: TradeResult):
        """Save trade result to disk."""
        history_file = self.data_dir / "trade_history.jsonl"
        try:
            with open(history_file, "a") as f:
                f.write(json.dumps(result.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"[L1] Error saving trade: {e}")

    def _add_to_window(self, result: TradeResult):
        """Add trade result to performance window."""
        bot = result.bot_name

        if bot not in self._performance:
            self._performance[bot] = PerformanceWindow()

        perf = self._performance[bot]
        perf.trades.append(result)

        # Maintain window size
        if len(perf.trades) > self.WINDOW_SIZE:
            old = perf.trades.pop(0)
            if old.is_win:
                perf.wins -= 1
            else:
                perf.losses -= 1
            perf.total_pnl -= old.pnl_dollars

        # Update metrics
        if result.is_win:
            perf.wins += 1
        else:
            perf.losses += 1
        perf.total_pnl += result.pnl_dollars

        # Recalculate derived metrics
        self._recalculate_metrics(bot)

        # Update streak
        if result.is_win:
            self._streak[bot] = max(1, self._streak.get(bot, 0) + 1)
        else:
            self._streak[bot] = min(-1, self._streak.get(bot, 0) - 1)

    def _recalculate_metrics(self, bot: str):
        """Recalculate performance metrics for a bot."""
        perf = self._performance.get(bot)
        if not perf or not perf.trades:
            return

        total = len(perf.trades)
        perf.win_rate = perf.wins / total if total > 0 else 0.5

        wins = [t for t in perf.trades if t.is_win]
        losses = [t for t in perf.trades if not t.is_win]

        perf.avg_win = np.mean([t.pnl_dollars for t in wins]) if wins else 0
        perf.avg_loss = abs(np.mean([t.pnl_dollars for t in losses])) if losses else 0

        # Profit factor
        total_wins = sum(t.pnl_dollars for t in wins) if wins else 0
        total_losses = abs(sum(t.pnl_dollars for t in losses)) if losses else 1
        perf.profit_factor = total_wins / total_losses if total_losses > 0 else 1.0

        # Kelly Criterion (with safety fraction)
        if perf.avg_loss > 0 and perf.win_rate > 0:
            # Kelly = W - (1-W)/R where W=win rate, R=avg_win/avg_loss
            r = perf.avg_win / perf.avg_loss if perf.avg_loss > 0 else 1
            kelly = perf.win_rate - (1 - perf.win_rate) / r if r > 0 else 0
            # Use half-Kelly for safety
            perf.kelly_fraction = max(0.005, min(0.02, kelly * 0.5))
        else:
            perf.kelly_fraction = self.BASE_RISK_PERCENT

    def record_trade(self, result: TradeResult):
        """Record a completed trade and update performance."""
        with self._lock:
            self._add_to_window(result)
            self._save_trade(result)

            logger.info(
                f"[L1] Recorded {result.bot_name}: {'WIN' if result.is_win else 'LOSS'} "
                f"{result.pnl_pips:.1f} pips (${result.pnl_dollars:.2f})"
            )

    def get_risk_multiplier(self, bot_name: str) -> float:
        """
        Get position size multiplier for a bot.

        Returns:
            Multiplier between 0.5 and 2.0 (1.0 = base risk)
        """
        with self._lock:
            perf = self._performance.get(bot_name)
            streak = self._streak.get(bot_name, 0)

            # Base multiplier from Kelly fraction
            if perf and len(perf.trades) >= 5:
                # Use Kelly-based sizing
                kelly_mult = perf.kelly_fraction / self.BASE_RISK_PERCENT
                base_mult = max(0.5, min(2.0, kelly_mult))
            else:
                # Not enough data, use base risk
                base_mult = 1.0

            # Apply streak adjustment
            if streak > 0:
                # Winning streak: gentle boost
                streak_adj = min(self.MAX_STREAK_EFFECT, streak) * self.WIN_STREAK_BOOST
                streak_mult = 1 + streak_adj
            elif streak < 0:
                # Losing streak: aggressive cut
                streak_adj = min(self.MAX_STREAK_EFFECT, abs(streak)) * self.LOSS_STREAK_CUT
                streak_mult = 1 - streak_adj
            else:
                streak_mult = 1.0

            # Combine multipliers
            final_mult = base_mult * streak_mult

            # Enforce limits
            final_mult = max(0.33, min(1.5, final_mult))  # Conservative caps

            self._last_sizes[bot_name] = final_mult

            logger.debug(
                f"[L1] {bot_name}: base={base_mult:.2f}, streak={streak}x{streak_mult:.2f}, "
                f"final={final_mult:.2f}"
            )

            return final_mult

    def get_adjusted_risk_percent(self, bot_name: str) -> float:
        """Get adjusted risk percentage for a bot."""
        mult = self.get_risk_multiplier(bot_name)
        adjusted = self.BASE_RISK_PERCENT * mult
        return max(self.MIN_RISK_PERCENT, min(self.MAX_RISK_PERCENT, adjusted))

    def get_stats(self, bot_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._lock:
            if bot_name:
                perf = self._performance.get(bot_name)
                if not perf:
                    return {"error": "No data"}
                return {
                    "bot": bot_name,
                    "trades": len(perf.trades),
                    "win_rate": perf.win_rate,
                    "profit_factor": perf.profit_factor,
                    "kelly_fraction": perf.kelly_fraction,
                    "streak": self._streak.get(bot_name, 0),
                    "risk_multiplier": self._last_sizes.get(bot_name, 1.0),
                }
            else:
                return {
                    bot: self.get_stats(bot)
                    for bot in self._performance.keys()
                }


# =============================================================================
# L2: VOLATILITY REGIME DETECTION
# =============================================================================

@dataclass
class VolatilityRegime:
    """Current volatility regime for a symbol."""
    symbol: str
    regime: str  # "low", "normal", "high", "extreme"
    atr: float
    atr_percentile: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class VolatilityRegimeDetector:
    """
    L2: Volatility Regime Detection and Parameter Adjustment.

    Detects market volatility regime and adjusts trading parameters:
    - LOW volatility: Tighter stops, smaller targets
    - NORMAL volatility: Standard parameters
    - HIGH volatility: Wider stops, larger targets
    - EXTREME volatility: Skip trading or reduce size

    Uses ATR (Average True Range) as primary volatility measure.
    """

    # Regime thresholds (percentile-based)
    REGIMES = {
        "low": (0, 25),       # Bottom 25% of ATR
        "normal": (25, 75),   # Middle 50%
        "high": (75, 90),     # High volatility
        "extreme": (90, 100), # Extreme - be careful
    }

    # Parameter adjustments by regime
    ADJUSTMENTS = {
        "low": {
            "sl_multiplier": 0.7,    # Tighter SL
            "tp_multiplier": 0.6,    # Smaller TP
            "size_multiplier": 1.2,  # Slightly larger size (lower risk)
            "should_trade": True,
        },
        "normal": {
            "sl_multiplier": 1.0,
            "tp_multiplier": 1.0,
            "size_multiplier": 1.0,
            "should_trade": True,
        },
        "high": {
            "sl_multiplier": 1.3,    # Wider SL
            "tp_multiplier": 1.5,    # Larger TP potential
            "size_multiplier": 0.8,  # Smaller size (higher risk)
            "should_trade": True,
        },
        "extreme": {
            "sl_multiplier": 1.5,
            "tp_multiplier": 2.0,
            "size_multiplier": 0.5,  # Half size
            "should_trade": False,   # Skip extreme volatility
        },
    }

    # ATR lookback period
    ATR_PERIOD = 14
    HISTORY_SIZE = 100  # Number of ATR values to keep for percentile calc

    def __init__(self, data_dir: str = "data/hydra/ftmo"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._atr_history: Dict[str, List[float]] = {}  # ATR values per symbol
        self._current_regime: Dict[str, VolatilityRegime] = {}

        self._load_history()
        logger.info("[L2 VolatilityRegimeDetector] Initialized")

    def _load_history(self):
        """Load ATR history from disk."""
        history_file = self.data_dir / "volatility_history.json"
        if not history_file.exists():
            return

        try:
            with open(history_file, "r") as f:
                data = json.load(f)
                self._atr_history = data.get("atr_history", {})
            logger.info(f"[L2] Loaded volatility history for {len(self._atr_history)} symbols")
        except Exception as e:
            logger.error(f"[L2] Error loading history: {e}")

    def _save_history(self):
        """Save ATR history to disk."""
        history_file = self.data_dir / "volatility_history.json"
        try:
            with open(history_file, "w") as f:
                json.dump({"atr_history": self._atr_history}, f)
        except Exception as e:
            logger.error(f"[L2] Error saving history: {e}")

    def calculate_atr(self, candles: List[Dict[str, float]]) -> float:
        """
        Calculate ATR from candles.

        Args:
            candles: List of dicts with 'high', 'low', 'close' keys

        Returns:
            ATR value
        """
        if len(candles) < self.ATR_PERIOD + 1:
            return 0.0

        true_ranges = []
        for i in range(1, len(candles)):
            high = candles[i]["high"]
            low = candles[i]["low"]
            prev_close = candles[i - 1]["close"]

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)

        # Use last ATR_PERIOD values
        return np.mean(true_ranges[-self.ATR_PERIOD:])

    def update_regime(self, symbol: str, atr: float) -> VolatilityRegime:
        """
        Update volatility regime for a symbol.

        Args:
            symbol: Trading symbol
            atr: Current ATR value

        Returns:
            Current volatility regime
        """
        with self._lock:
            # Add to history
            if symbol not in self._atr_history:
                self._atr_history[symbol] = []

            self._atr_history[symbol].append(atr)

            # Maintain history size
            if len(self._atr_history[symbol]) > self.HISTORY_SIZE:
                self._atr_history[symbol].pop(0)

            # Calculate percentile
            history = self._atr_history[symbol]
            if len(history) >= 10:
                percentile = (sum(1 for x in history if x <= atr) / len(history)) * 100
            else:
                percentile = 50  # Default to normal if not enough data

            # Determine regime
            regime = "normal"
            for regime_name, (low, high) in self.REGIMES.items():
                if low <= percentile < high:
                    regime = regime_name
                    break

            # Create regime object
            vol_regime = VolatilityRegime(
                symbol=symbol,
                regime=regime,
                atr=atr,
                atr_percentile=percentile,
            )

            self._current_regime[symbol] = vol_regime
            self._save_history()

            logger.debug(
                f"[L2] {symbol}: ATR={atr:.2f}, percentile={percentile:.1f}%, regime={regime}"
            )

            return vol_regime

    def get_regime(self, symbol: str) -> Optional[VolatilityRegime]:
        """Get current volatility regime for a symbol."""
        with self._lock:
            return self._current_regime.get(symbol)

    def get_adjustments(self, symbol: str) -> Dict[str, Any]:
        """
        Get parameter adjustments for current volatility regime.

        Returns:
            Dict with sl_multiplier, tp_multiplier, size_multiplier, should_trade
        """
        regime = self.get_regime(symbol)
        if not regime:
            return self.ADJUSTMENTS["normal"]

        return self.ADJUSTMENTS.get(regime.regime, self.ADJUSTMENTS["normal"])

    def should_trade(self, symbol: str) -> Tuple[bool, str]:
        """Check if trading is advisable given volatility."""
        regime = self.get_regime(symbol)
        if not regime:
            return True, "No volatility data"

        adjustments = self.ADJUSTMENTS.get(regime.regime, self.ADJUSTMENTS["normal"])

        if not adjustments["should_trade"]:
            return False, f"Extreme volatility: ATR={regime.atr:.2f} (p{regime.atr_percentile:.0f})"

        return True, f"{regime.regime} volatility"

    def get_stats(self) -> Dict[str, Any]:
        """Get volatility statistics for all symbols."""
        with self._lock:
            return {
                symbol: {
                    "regime": regime.regime,
                    "atr": regime.atr,
                    "percentile": regime.atr_percentile,
                    "adjustments": self.ADJUSTMENTS.get(regime.regime, {}),
                }
                for symbol, regime in self._current_regime.items()
            }


# =============================================================================
# L3: MARKET REGIME DETECTION (ENGINE-POWERED)
# =============================================================================

@dataclass
class MarketRegime:
    """Current market regime from engine analysis."""
    regime: str  # "risk_on", "risk_off", "neutral", "uncertain"
    confidence: float  # 0-1
    drivers: List[str]  # Key factors driving the regime
    recommended_action: str  # "aggressive", "normal", "conservative", "skip"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MarketRegimeAnalyzer:
    """
    L3: Market Regime Detection using HYDRA Engine Analysis.

    Analyzes macro conditions to determine overall market sentiment:
    - RISK_ON: Bull market, low fear, engines see opportunity
    - RISK_OFF: Bear market, high fear, engines advise caution
    - NEUTRAL: Sideways/choppy, normal trading conditions
    - UNCERTAIN: Conflicting signals, reduce exposure

    Uses simplified heuristics when engines are unavailable,
    but can integrate with full HYDRA engine analysis for deeper insights.
    """

    # Regime definitions and adjustments
    REGIME_ADJUSTMENTS = {
        "risk_on": {
            "action": "aggressive",
            "size_multiplier": 1.3,     # Increase position size
            "confidence_boost": 0.1,     # Lower confidence threshold
            "tp_multiplier": 1.2,        # Larger targets
            "sl_multiplier": 0.9,        # Tighter stops (risk reversal less likely)
            "max_trades_boost": 1.5,     # Allow more trades
        },
        "risk_off": {
            "action": "conservative",
            "size_multiplier": 0.6,      # Reduce position size
            "confidence_boost": -0.1,    # Higher confidence required
            "tp_multiplier": 0.8,        # Smaller targets
            "sl_multiplier": 1.3,        # Wider stops (high volatility)
            "max_trades_boost": 0.5,     # Fewer trades
        },
        "neutral": {
            "action": "normal",
            "size_multiplier": 1.0,
            "confidence_boost": 0.0,
            "tp_multiplier": 1.0,
            "sl_multiplier": 1.0,
            "max_trades_boost": 1.0,
        },
        "uncertain": {
            "action": "skip",
            "size_multiplier": 0.4,      # Very small positions
            "confidence_boost": -0.2,    # Much higher confidence needed
            "tp_multiplier": 0.7,        # Tight targets
            "sl_multiplier": 1.5,        # Wide stops
            "max_trades_boost": 0.3,     # Very few trades
        },
    }

    # Asset correlations for regime detection
    # Note: Uses FTMO broker symbol names (US30.cash, US100.cash)
    RISK_ASSETS = ["BTC-USD", "ETH-USD", "SOL-USD", "US100.cash", "US30.cash", "EURUSD", "GBPUSD"]
    SAFE_ASSETS = ["XAUUSD", "USDJPY"]  # Gold and JPY as safe havens

    # Update frequency (don't spam engine calls)
    UPDATE_INTERVAL_MINUTES = 15
    CACHE_TTL_MINUTES = 30

    def __init__(self, data_dir: str = "data/hydra/ftmo"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._current_regime: Optional[MarketRegime] = None
        self._last_update: Optional[datetime] = None
        self._regime_history: List[Dict] = []

        # Engine availability flag
        self._engines_available = False
        self._check_engine_availability()

        self._load_history()
        logger.info("[L3 MarketRegimeAnalyzer] Initialized")

    def _check_engine_availability(self):
        """Check if HYDRA engines are available for deep analysis."""
        try:
            # Try to import engine modules
            from libs.hydra.mother_ai import MotherAI
            self._engines_available = True
            logger.info("[L3] HYDRA engines available for regime analysis")
        except ImportError:
            self._engines_available = False
            logger.info("[L3] HYDRA engines not available, using heuristic mode")

    def _load_history(self):
        """Load regime history from disk."""
        history_file = self.data_dir / "regime_history.json"
        if not history_file.exists():
            return

        try:
            with open(history_file, "r") as f:
                data = json.load(f)
                self._regime_history = data.get("history", [])[-100:]  # Keep last 100
            logger.info(f"[L3] Loaded {len(self._regime_history)} regime records")
        except Exception as e:
            logger.error(f"[L3] Error loading history: {e}")

    def _save_history(self):
        """Save regime history to disk."""
        history_file = self.data_dir / "regime_history.json"
        try:
            with open(history_file, "w") as f:
                json.dump({"history": self._regime_history[-100:]}, f)
        except Exception as e:
            logger.error(f"[L3] Error saving history: {e}")

    def analyze_market_regime(
        self,
        price_data: Optional[Dict[str, List[float]]] = None,
        force_update: bool = False
    ) -> MarketRegime:
        """
        Analyze current market regime.

        Args:
            price_data: Optional dict of {symbol: [prices]} for analysis
            force_update: Force regime recalculation

        Returns:
            Current MarketRegime
        """
        with self._lock:
            now = datetime.now(timezone.utc)

            # Check if we can use cached regime
            if (
                not force_update
                and self._current_regime
                and self._last_update
                and (now - self._last_update).total_seconds() < self.CACHE_TTL_MINUTES * 60
            ):
                return self._current_regime

            # Perform analysis
            if self._engines_available:
                regime = self._analyze_with_engines(price_data)
            else:
                regime = self._analyze_heuristic(price_data)

            self._current_regime = regime
            self._last_update = now

            # Record history
            self._regime_history.append({
                "timestamp": now.isoformat(),
                "regime": regime.regime,
                "confidence": regime.confidence,
                "drivers": regime.drivers,
            })
            self._save_history()

            logger.info(
                f"[L3] Market regime: {regime.regime.upper()} "
                f"(confidence: {regime.confidence:.0%}, action: {regime.recommended_action})"
            )

            return regime

    def _analyze_with_engines(self, price_data: Optional[Dict] = None) -> MarketRegime:
        """
        Analyze market regime using HYDRA engines.

        This queries the gladiator engines for their macro view.
        """
        try:
            from libs.hydra.mother_ai import MotherAI

            # Get engine consensus on market conditions
            # For now, use simplified logic - can be extended to full engine queries

            drivers = []
            regime_votes = {"risk_on": 0, "risk_off": 0, "neutral": 0}

            # Use price momentum as primary signal
            if price_data:
                for symbol, prices in price_data.items():
                    if len(prices) >= 20:
                        # Calculate short-term momentum
                        short_ma = np.mean(prices[-5:])
                        long_ma = np.mean(prices[-20:])
                        momentum = (short_ma - long_ma) / long_ma if long_ma > 0 else 0

                        if symbol in self.RISK_ASSETS:
                            if momentum > 0.02:
                                regime_votes["risk_on"] += 1
                                drivers.append(f"{symbol} bullish (+{momentum:.1%})")
                            elif momentum < -0.02:
                                regime_votes["risk_off"] += 1
                                drivers.append(f"{symbol} bearish ({momentum:.1%})")
                            else:
                                regime_votes["neutral"] += 1

            # Determine regime
            total_votes = sum(regime_votes.values()) or 1
            max_regime = max(regime_votes, key=regime_votes.get)
            confidence = regime_votes[max_regime] / total_votes

            # Low confidence = uncertain
            if confidence < 0.5:
                regime = "uncertain"
                confidence = 0.4
            else:
                regime = max_regime

            return MarketRegime(
                regime=regime,
                confidence=confidence,
                drivers=drivers[:5],  # Top 5 drivers
                recommended_action=self.REGIME_ADJUSTMENTS[regime]["action"],
            )

        except Exception as e:
            logger.warning(f"[L3] Engine analysis failed: {e}, falling back to heuristic")
            return self._analyze_heuristic(price_data)

    def _analyze_heuristic(self, price_data: Optional[Dict] = None) -> MarketRegime:
        """
        Analyze market regime using simple heuristics (no engines needed).

        Uses:
        - Time of day (session activity)
        - Day of week (weekend risk)
        - Recent volatility patterns
        """
        drivers = []
        now = datetime.now(timezone.utc)

        # Day of week factor
        day = now.weekday()
        if day == 4:  # Friday
            drivers.append("Friday: reduced exposure")
            regime = "conservative"
        elif day in [5, 6]:  # Weekend
            drivers.append("Weekend: minimal trading")
            regime = "skip"
        else:
            regime = "neutral"

        # Time of day factor
        hour = now.hour
        if 8 <= hour <= 11:  # London session
            drivers.append("London session: normal activity")
        elif 13 <= hour <= 16:  # NY overlap
            drivers.append("NY overlap: high activity")
        elif 0 <= hour <= 3:  # Asia session
            drivers.append("Asia session: lower activity")
            if regime == "neutral":
                regime = "conservative"
        elif 22 <= hour or hour <= 5:  # Off-hours
            drivers.append("Off-hours: reduced liquidity")
            regime = "conservative"

        # Price data analysis if available
        if price_data:
            bullish_count = 0
            bearish_count = 0

            for symbol, prices in price_data.items():
                if len(prices) >= 10:
                    change = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
                    if change > 0.005:
                        bullish_count += 1
                    elif change < -0.005:
                        bearish_count += 1

            if bullish_count > bearish_count * 1.5:
                drivers.append(f"Bullish momentum ({bullish_count} assets)")
                regime = "risk_on"
            elif bearish_count > bullish_count * 1.5:
                drivers.append(f"Bearish pressure ({bearish_count} assets)")
                regime = "risk_off"

        # Map regime to proper type
        regime_map = {
            "aggressive": "risk_on",
            "conservative": "risk_off",
            "neutral": "neutral",
            "skip": "uncertain",
        }
        if regime in regime_map:
            regime = regime_map.get(regime, regime)

        # Default confidence for heuristic
        confidence = 0.6 if regime != "uncertain" else 0.4

        return MarketRegime(
            regime=regime,
            confidence=confidence,
            drivers=drivers,
            recommended_action=self.REGIME_ADJUSTMENTS.get(regime, self.REGIME_ADJUSTMENTS["neutral"])["action"],
        )

    def get_regime(self) -> Optional[MarketRegime]:
        """Get current market regime (cached)."""
        with self._lock:
            return self._current_regime

    def get_adjustments(self, regime: Optional[MarketRegime] = None) -> Dict[str, Any]:
        """Get parameter adjustments for current regime."""
        if regime is None:
            regime = self._current_regime

        if regime is None:
            return self.REGIME_ADJUSTMENTS["neutral"]

        return self.REGIME_ADJUSTMENTS.get(regime.regime, self.REGIME_ADJUSTMENTS["neutral"])

    def should_trade(self, min_confidence: float = 0.4) -> Tuple[bool, str]:
        """Check if trading is advisable given market regime."""
        regime = self._current_regime

        if regime is None:
            # No regime data - analyze now
            regime = self.analyze_market_regime()

        adjustments = self.get_adjustments(regime)

        if adjustments["action"] == "skip":
            return False, f"Market regime: {regime.regime} (skip recommended)"

        if regime.confidence < min_confidence:
            return False, f"Low regime confidence: {regime.confidence:.0%}"

        return True, f"Regime: {regime.regime} ({regime.confidence:.0%} conf)"

    def get_stats(self) -> Dict[str, Any]:
        """Get regime statistics."""
        with self._lock:
            if not self._current_regime:
                return {"status": "no_data"}

            return {
                "current_regime": self._current_regime.regime,
                "confidence": self._current_regime.confidence,
                "recommended_action": self._current_regime.recommended_action,
                "drivers": self._current_regime.drivers,
                "last_update": self._last_update.isoformat() if self._last_update else None,
                "history_count": len(self._regime_history),
                "engines_available": self._engines_available,
            }


# =============================================================================
# METALEARNER: COMBINES L1 + L2 + L3
# =============================================================================

class FTMOMetalearner:
    """
    Combined Metalearner for FTMO bots.

    Integrates:
    - L1: Adaptive Position Sizing (performance-based)
    - L2: Volatility Regime Detection (market volatility)
    - L3: Market Regime Detection (macro conditions via HYDRA engines)

    Usage:
        metalearner = get_ftmo_metalearner()

        # Before trade: Get adjusted parameters
        params = metalearner.get_trade_parameters(bot_name, symbol, candles)
        # params = {risk_percent, sl_multiplier, tp_multiplier, should_trade, reason}

        # After trade: Record result
        metalearner.record_trade(result)

        # Update market regime (periodically)
        metalearner.update_market_regime(price_data)
    """

    def __init__(self, data_dir: str = "data/hydra/ftmo"):
        self.data_dir = Path(data_dir)
        self.position_sizer = AdaptivePositionSizer(data_dir)
        self.volatility_detector = VolatilityRegimeDetector(data_dir)
        self.regime_analyzer = MarketRegimeAnalyzer(data_dir)

        logger.info("[Metalearner] FTMO Metalearner initialized (L1 + L2 + L3)")

    def update_volatility(self, symbol: str, candles: List[Dict[str, float]]) -> VolatilityRegime:
        """Update volatility regime for a symbol."""
        atr = self.volatility_detector.calculate_atr(candles)
        return self.volatility_detector.update_regime(symbol, atr)

    def update_market_regime(self, price_data: Optional[Dict[str, List[float]]] = None) -> MarketRegime:
        """
        Update market regime using engine analysis.

        Args:
            price_data: Dict of {symbol: [recent_prices]} for multi-asset analysis

        Returns:
            Current MarketRegime
        """
        return self.regime_analyzer.analyze_market_regime(price_data)

    def record_trade(self, result: TradeResult):
        """Record a completed trade."""
        self.position_sizer.record_trade(result)

    def get_trade_parameters(
        self,
        bot_name: str,
        symbol: str,
        candles: Optional[List[Dict[str, float]]] = None,
        price_data: Optional[Dict[str, List[float]]] = None,
        base_sl_pips: float = 50,
        base_tp_pips: float = 90,
    ) -> Dict[str, Any]:
        """
        Get adjusted trade parameters based on performance, volatility, and market regime.

        Combines three layers of intelligence:
        - L1: Adaptive position sizing based on bot's recent performance
        - L2: Volatility regime adjustments based on ATR
        - L3: Market regime adjustments based on engine macro analysis

        Args:
            bot_name: Name of the trading bot
            symbol: Trading symbol
            candles: Optional candle data for volatility update
            price_data: Optional multi-asset price data for regime analysis
            base_sl_pips: Base stop loss in pips
            base_tp_pips: Base take profit in pips

        Returns:
            Dict with:
                - risk_percent: Adjusted risk percentage
                - sl_pips: Adjusted stop loss
                - tp_pips: Adjusted take profit
                - size_multiplier: Combined size multiplier
                - should_trade: Whether to trade
                - reason: Explanation
                - market_regime: Current macro regime
        """
        # Update volatility if candles provided
        if candles and len(candles) > 15:
            self.update_volatility(symbol, candles)

        # Update market regime if price data provided (or use cached)
        if price_data:
            self.update_market_regime(price_data)

        # L1: Get performance-based risk adjustment
        perf_risk = self.position_sizer.get_adjusted_risk_percent(bot_name)
        perf_mult = self.position_sizer.get_risk_multiplier(bot_name)

        # L2: Get volatility-based adjustments
        vol_adj = self.volatility_detector.get_adjustments(symbol)
        can_trade_vol, vol_reason = self.volatility_detector.should_trade(symbol)

        # L3: Get market regime adjustments
        market_regime = self.regime_analyzer.get_regime()
        regime_adj = self.regime_analyzer.get_adjustments(market_regime)
        can_trade_regime, regime_reason = self.regime_analyzer.should_trade()

        # Combine all adjustments
        # Size: L1 (performance) × L2 (volatility) × L3 (regime)
        final_size_mult = perf_mult * vol_adj["size_multiplier"] * regime_adj["size_multiplier"]

        # SL/TP: L2 (volatility) × L3 (regime) adjustments
        final_sl = base_sl_pips * vol_adj["sl_multiplier"] * regime_adj["sl_multiplier"]
        final_tp = base_tp_pips * vol_adj["tp_multiplier"] * regime_adj["tp_multiplier"]

        # Should trade: All layers must agree
        should_trade = can_trade_vol and can_trade_regime

        # Enforce limits
        final_size_mult = max(0.25, min(2.0, final_size_mult))  # Extended range with L3
        final_sl = max(10, min(100, final_sl))  # Cap SL range
        final_tp = max(15, min(200, final_tp))  # Cap TP range

        vol_regime = self.volatility_detector.get_regime(symbol)
        vol_regime_name = vol_regime.regime if vol_regime else "unknown"
        market_regime_name = market_regime.regime if market_regime else "neutral"

        params = {
            "risk_percent": perf_risk,
            "sl_pips": final_sl,
            "tp_pips": final_tp,
            "size_multiplier": final_size_mult,
            "should_trade": should_trade,
            "market_regime": market_regime_name,
            "reason": (
                f"L1: {perf_mult:.2f}x, "
                f"L2: {vol_regime_name} ({vol_adj['size_multiplier']:.2f}x), "
                f"L3: {market_regime_name} ({regime_adj['size_multiplier']:.2f}x)"
            ),
            "details": {
                "l1_multiplier": perf_mult,
                "l1_risk_percent": perf_risk,
                "l2_regime": vol_regime_name,
                "l2_sl_mult": vol_adj["sl_multiplier"],
                "l2_tp_mult": vol_adj["tp_multiplier"],
                "l2_size_mult": vol_adj["size_multiplier"],
                "l3_regime": market_regime_name,
                "l3_action": regime_adj["action"],
                "l3_size_mult": regime_adj["size_multiplier"],
                "l3_sl_mult": regime_adj["sl_multiplier"],
                "l3_tp_mult": regime_adj["tp_multiplier"],
                "l3_drivers": market_regime.drivers if market_regime else [],
            }
        }

        logger.info(
            f"[Metalearner] {bot_name}/{symbol}: "
            f"risk={perf_risk*100:.2f}%, SL={final_sl:.0f}, TP={final_tp:.0f}, "
            f"size={final_size_mult:.2f}x, regime={market_regime_name}, trade={should_trade}"
        )

        return params

    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics from all layers."""
        return {
            "l1_performance": self.position_sizer.get_stats(),
            "l2_volatility": self.volatility_detector.get_stats(),
            "l3_market_regime": self.regime_analyzer.get_stats(),
        }


# Singleton
_metalearner: Optional[FTMOMetalearner] = None
_meta_lock = threading.Lock()


def get_ftmo_metalearner() -> FTMOMetalearner:
    """Get FTMO Metalearner singleton."""
    global _metalearner
    if _metalearner is None:
        with _meta_lock:
            if _metalearner is None:
                _metalearner = FTMOMetalearner()
    return _metalearner
