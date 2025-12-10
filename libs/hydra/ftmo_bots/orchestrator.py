"""
FTMO Bot Orchestrator

Manages all FTMO bots with unified risk management:
- Runs all 5 bots on their appropriate schedules
- Enforces max 3 concurrent positions
- Daily drawdown limit: 4.5%
- Total drawdown limit: 8.5%
- Correlation checking between positions
- Telegram alerts for all trades

Usage:
    orchestrator = get_ftmo_orchestrator()
    orchestrator.run()  # Runs continuously
    # or
    orchestrator.run_single_cycle()  # Runs one check cycle
"""

import os
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from loguru import logger

from .gold_london_reversal import get_gold_london_bot
from .eurusd_breakout import get_eurusd_bot
from .us30_orb import get_us30_bot
from .nas100_gap import get_nas100_bot
from .gold_ny_reversion import get_gold_ny_bot
from .hf_scalper import get_hf_scalper
from .base_ftmo_bot import TradeSignal
from .metalearning import get_ftmo_metalearner, TradeResult, FTMOMetalearner

# Import ZMQ client for MT5 connection
try:
    from libs.brokers.mt5_zmq_client import MT5ZMQClient, get_mt5_client
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    logger.warning("[Orchestrator] ZMQ client not available, falling back to HTTP")


@dataclass
class RiskLimits:
    """Risk management limits for FTMO compliance."""
    max_daily_loss_percent: float = 4.5  # FTMO daily limit: 5%
    max_total_drawdown_percent: float = 8.5  # FTMO total limit: 10%
    max_concurrent_positions: int = 3
    max_correlated_positions: int = 2
    max_risk_per_trade_percent: float = 1.5


@dataclass
class DailyStats:
    """Daily trading statistics."""
    date: datetime
    starting_balance: float
    current_balance: float
    trades_taken: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0


class FTMOOrchestrator:
    """
    Master orchestrator for all FTMO bots.

    Handles scheduling, risk management, and position tracking.
    """

    # Correlation matrix for position management
    # Highly correlated pairs should not be traded together
    # Note: Uses FTMO broker symbol names (US30.cash, US100.cash)
    CORRELATIONS = {
        ("XAUUSD", "XAUUSD"): 1.0,
        ("XAUUSD", "US30.cash"): -0.3,  # Gold vs Dow - low inverse
        ("XAUUSD", "US100.cash"): -0.2,  # Gold vs NASDAQ - low inverse
        ("XAUUSD", "EURUSD"): 0.4,  # Gold vs EUR - moderate positive
        ("US30.cash", "US100.cash"): 0.85,  # Dow vs NASDAQ - high positive
        ("EURUSD", "US30.cash"): -0.2,
        ("EURUSD", "US100.cash"): -0.2,
        ("EURUSD", "GBPUSD"): 0.85,  # EUR vs GBP - highly correlated
        ("GBPUSD", "XAUUSD"): 0.35,  # GBP vs Gold - moderate
        ("GBPUSD", "US30.cash"): -0.25,  # GBP vs Dow
        ("GBPUSD", "US100.cash"): -0.20,  # GBP vs NASDAQ
    }

    MT5_EXECUTOR_URL = os.getenv("MT5_EXECUTOR_URL", "http://45.82.167.195:5000")
    MT5_API_SECRET = os.getenv("MT5_API_SECRET", "hydra_secret_2024")

    def __init__(self, paper_mode: bool = True, enable_metalearning: bool = True, use_zmq: bool = True, turbo_mode: bool = False):
        self.paper_mode = paper_mode
        self.turbo_mode = turbo_mode
        self.limits = RiskLimits()
        self.enable_metalearning = enable_metalearning
        self.use_zmq = use_zmq and ZMQ_AVAILABLE

        # Initialize ZMQ client for MT5 connection
        self._zmq_client: Optional[MT5ZMQClient] = None
        if self.use_zmq:
            try:
                self._zmq_client = get_mt5_client()
                if self._zmq_client.connect():
                    logger.info("[Orchestrator] ZMQ client connected to MT5 executor")
                else:
                    logger.warning("[Orchestrator] ZMQ client failed to connect, falling back to HTTP")
                    self.use_zmq = False
            except Exception as e:
                logger.warning(f"[Orchestrator] ZMQ init failed: {e}, falling back to HTTP")
                self.use_zmq = False

        # Initialize all bots (pass turbo_mode to bots that support it)
        self.bots = {
            "gold_london": get_gold_london_bot(paper_mode, turbo_mode=turbo_mode),
            "eurusd": get_eurusd_bot(paper_mode),
            "us30": get_us30_bot(paper_mode),
            "nas100": get_nas100_bot(paper_mode),
            "gold_ny": get_gold_ny_bot(paper_mode),
            "hf_scalper": get_hf_scalper(paper_mode, turbo_mode=turbo_mode),
        }

        if turbo_mode:
            logger.info("[Orchestrator] TURBO MODE enabled - thresholds lowered, max trades increased")

        # Initialize metalearner (L1 + L2)
        self._metalearner: Optional[FTMOMetalearner] = None
        if enable_metalearning:
            self._metalearner = get_ftmo_metalearner()
            logger.info("[Orchestrator] Metalearning enabled (L1: Adaptive Sizing, L2: Volatility Regimes)")

        # State tracking
        self._open_positions: List[Dict[str, Any]] = []
        self._pending_trades: Dict[str, TradeSignal] = {}  # Track pending trades for P&L
        self._daily_stats: Optional[DailyStats] = None
        self._starting_balance: float = 15000.0  # Default FTMO starting
        self._kill_switch = False

        # Thread safety
        self._lock = threading.Lock()
        self._running = False

        logger.info(
            f"[Orchestrator] Initialized with {len(self.bots)} bots "
            f"(paper_mode: {paper_mode}, max_positions: {self.limits.max_concurrent_positions})"
        )

    def run(self, cycle_interval_seconds: int = 60):
        """
        Run orchestrator continuously.

        Args:
            cycle_interval_seconds: Time between cycles (default 60s)
        """
        self._running = True
        logger.info("[Orchestrator] Starting continuous run...")

        while self._running and not self._kill_switch:
            try:
                self.run_single_cycle()
                time.sleep(cycle_interval_seconds)
            except KeyboardInterrupt:
                logger.info("[Orchestrator] Stopped by user")
                break
            except Exception as e:
                logger.error(f"[Orchestrator] Cycle error: {e}")
                time.sleep(cycle_interval_seconds)

        logger.info("[Orchestrator] Stopped")

    def stop(self):
        """Stop the orchestrator."""
        self._running = False

    def run_single_cycle(self) -> List[TradeSignal]:
        """
        Run one cycle of all bots with metalearning adjustments.

        Returns:
            List of signals generated this cycle
        """
        # Update daily stats
        self._update_daily_stats()

        # Check risk limits
        if not self._check_risk_limits():
            return []

        # Update open positions
        self._update_open_positions()

        signals = []

        # Run each bot
        for bot_name, bot in self.bots.items():
            try:
                # Check if we can take more positions
                if len(self._open_positions) >= self.limits.max_concurrent_positions:
                    logger.debug(f"[Orchestrator] Max positions reached, skipping {bot_name}")
                    continue

                # Check correlation with existing positions
                if not self._check_correlation(bot.config.symbol):
                    logger.debug(f"[Orchestrator] Correlation check failed for {bot_name}")
                    continue

                # Fetch candles via ZMQ (for both metalearning and passing to bot)
                candles = self._get_candles(bot.config.symbol, count=500, timeframe="M1")

                # Apply L2 volatility check before running bot
                if self._metalearner:
                    if candles and len(candles) > 15:
                        self._metalearner.update_volatility(bot.config.symbol, candles)

                    # Check if volatility regime allows trading
                    can_trade, vol_reason = self._metalearner.volatility_detector.should_trade(bot.config.symbol)
                    if not can_trade:
                        logger.info(f"[Orchestrator] Skipping {bot_name}: {vol_reason}")
                        continue

                    # Apply L1 adaptive risk adjustment to bot config
                    adjusted_risk = self._metalearner.position_sizer.get_adjusted_risk_percent(bot_name)
                    original_risk = bot.config.risk_percent
                    bot.config.risk_percent = adjusted_risk
                    logger.debug(
                        f"[Orchestrator] {bot_name} risk adjusted: "
                        f"{original_risk*100:.2f}% -> {adjusted_risk*100:.2f}%"
                    )

                # Run bot cycle with market data (avoids bot's HTTP fallback)
                market_data = {"candles": candles} if candles else None
                signal = bot.run_cycle(market_data)

                # Restore original risk after trade
                if self._metalearner:
                    bot.config.risk_percent = 0.015  # Reset to base

                if signal:
                    # Track for P&L recording
                    self._pending_trades[f"{bot_name}_{signal.timestamp.timestamp()}"] = signal

                    signals.append(signal)
                    self._record_trade(signal)
                    self._send_telegram_alert(signal)

            except Exception as e:
                logger.error(f"[Orchestrator] Error running {bot_name}: {e}")

        return signals

    def _check_risk_limits(self) -> bool:
        """Check if we're within risk limits."""
        if self._kill_switch:
            logger.warning("[Orchestrator] Kill switch active - no trading")
            return False

        if self._daily_stats is None:
            return True

        # Check daily loss limit
        daily_pnl_percent = (
            (self._daily_stats.current_balance - self._daily_stats.starting_balance)
            / self._daily_stats.starting_balance * 100
        )

        if daily_pnl_percent <= -self.limits.max_daily_loss_percent:
            logger.warning(
                f"[Orchestrator] DAILY LOSS LIMIT HIT: {daily_pnl_percent:.2f}% "
                f"(limit: -{self.limits.max_daily_loss_percent}%)"
            )
            self._kill_switch = True
            self._send_alert("DAILY LOSS LIMIT HIT - Trading stopped")
            return False

        # Check total drawdown
        account = self._get_account_info()
        if account:
            total_dd = (self._starting_balance - account.get("balance", 0)) / self._starting_balance * 100
            if total_dd >= self.limits.max_total_drawdown_percent:
                logger.warning(
                    f"[Orchestrator] TOTAL DRAWDOWN LIMIT: {total_dd:.2f}% "
                    f"(limit: {self.limits.max_total_drawdown_percent}%)"
                )
                self._kill_switch = True
                self._send_alert("TOTAL DRAWDOWN LIMIT HIT - Trading stopped")
                return False

        return True

    def _check_correlation(self, symbol: str) -> bool:
        """
        Check if new position would violate correlation limits.

        Prevents holding multiple highly correlated positions.
        """
        correlated_count = 0

        for position in self._open_positions:
            pos_symbol = position.get("symbol", "")
            correlation = self._get_correlation(symbol, pos_symbol)

            if abs(correlation) > 0.7:
                correlated_count += 1

        return correlated_count < self.limits.max_correlated_positions

    def _get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols."""
        key = (symbol1, symbol2)
        if key in self.CORRELATIONS:
            return self.CORRELATIONS[key]

        # Try reverse key
        key_rev = (symbol2, symbol1)
        if key_rev in self.CORRELATIONS:
            return self.CORRELATIONS[key_rev]

        return 0.0  # Unknown correlation

    def _update_open_positions(self):
        """Update list of open positions from MT5."""
        try:
            if self.use_zmq and self._zmq_client:
                # Use ZMQ client
                positions = self._zmq_client.get_positions()
                self._open_positions = positions
                logger.debug(f"[Orchestrator] {len(self._open_positions)} open positions (ZMQ)")
            else:
                # Fallback to HTTP
                import requests
                url = f"{self.MT5_EXECUTOR_URL}/positions"
                headers = {"Authorization": f"Bearer {self.MT5_API_SECRET}"}

                response = requests.get(url, headers=headers, timeout=10)
                data = response.json()

                if data.get("success"):
                    self._open_positions = data.get("positions", [])
                    logger.debug(f"[Orchestrator] {len(self._open_positions)} open positions (HTTP)")

        except Exception as e:
            logger.warning(f"[Orchestrator] Failed to update positions: {e}")

    def _update_daily_stats(self):
        """Update daily statistics."""
        today = datetime.now(timezone.utc).date()

        # Reset stats at start of new day
        if self._daily_stats is None or self._daily_stats.date.date() != today:
            account = self._get_account_info()
            balance = account.get("balance", 15000) if account else 15000

            self._daily_stats = DailyStats(
                date=datetime.now(timezone.utc),
                starting_balance=balance,
                current_balance=balance,
            )

            # Reset kill switch at new day
            self._kill_switch = False

            logger.info(f"[Orchestrator] New day - starting balance: ${balance:.2f}")

        else:
            # Update current balance
            account = self._get_account_info()
            if account:
                self._daily_stats.current_balance = account.get("balance", self._daily_stats.current_balance)

    def _get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get account info from MT5."""
        try:
            if self.use_zmq and self._zmq_client:
                # Use ZMQ client
                account = self._zmq_client.get_account()
                return account
            else:
                # Fallback to HTTP
                import requests
                url = f"{self.MT5_EXECUTOR_URL}/account"
                headers = {"Authorization": f"Bearer {self.MT5_API_SECRET}"}
                response = requests.get(url, headers=headers, timeout=10)
                return response.json()
        except Exception as e:
            logger.warning(f"[Orchestrator] Failed to get account info: {e}")
            return None

    def _get_candles(self, symbol: str, count: int = 100, timeframe: str = "H1") -> List[Dict[str, float]]:
        """Fetch candles for volatility analysis."""
        try:
            # Use ZMQ for candles if available
            if self.use_zmq and self._zmq_client:
                candles = self._zmq_client.get_candles(symbol, timeframe, count)
                if candles:
                    logger.debug(f"[Orchestrator] Got {len(candles)} candles via ZMQ for {symbol}")
                return candles

            # HTTP fallback
            import requests
            url = f"{self.MT5_EXECUTOR_URL}/candles/{symbol}"
            headers = {"Authorization": f"Bearer {self.MT5_API_SECRET}"}
            params = {"count": count, "timeframe": timeframe}
            response = requests.get(url, headers=headers, params=params, timeout=10)
            data = response.json()
            return data.get("candles", [])
        except Exception as e:
            logger.warning(f"[Orchestrator] Failed to get candles for {symbol}: {e}")
            return []

    def _apply_metalearning(self, bot_name: str, symbol: str, signal: TradeSignal) -> TradeSignal:
        """Apply metalearning adjustments to a trade signal."""
        if not self._metalearner:
            return signal

        # Fetch candles for volatility analysis
        candles = self._get_candles(symbol)

        # Get adjusted parameters
        params = self._metalearner.get_trade_parameters(
            bot_name=bot_name,
            symbol=symbol,
            candles=candles,
            base_sl_pips=abs(signal.entry_price - signal.stop_loss),
            base_tp_pips=abs(signal.take_profit - signal.entry_price),
        )

        # Check if we should trade
        if not params["should_trade"]:
            logger.warning(
                f"[Orchestrator] Metalearning blocked trade for {bot_name}/{symbol}: "
                f"{params['reason']}"
            )
            return None

        # Adjust lot size
        adjusted_lot = signal.lot_size * params["size_multiplier"]
        # SAFETY: Max 0.5 lots after $503 loss on 2025-12-10
        adjusted_lot = round(max(0.01, min(0.5, adjusted_lot)), 2)

        # Create adjusted signal
        adjusted_signal = TradeSignal(
            bot_name=signal.bot_name,
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            lot_size=adjusted_lot,
            reason=f"{signal.reason} | ML: {params['reason']}",
            confidence=signal.confidence,
            timestamp=signal.timestamp,
        )

        logger.info(
            f"[Orchestrator] Metalearning adjusted {bot_name}: "
            f"lot {signal.lot_size} -> {adjusted_lot} ({params['size_multiplier']:.2f}x)"
        )

        return adjusted_signal

    def record_trade_result(
        self,
        signal: TradeSignal,
        exit_price: float,
        pnl_pips: float,
        pnl_dollars: float,
    ):
        """Record trade result for metalearning."""
        if not self._metalearner:
            return

        result = TradeResult(
            bot_name=signal.bot_name,
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            exit_price=exit_price,
            lot_size=signal.lot_size,
            pnl_pips=pnl_pips,
            pnl_dollars=pnl_dollars,
            is_win=pnl_dollars > 0,
        )

        self._metalearner.record_trade(result)
        logger.info(
            f"[Orchestrator] Recorded trade result: {signal.bot_name} "
            f"{'WIN' if result.is_win else 'LOSS'} ${pnl_dollars:.2f}"
        )

    def _record_trade(self, signal: TradeSignal):
        """Record trade in daily stats."""
        if self._daily_stats:
            self._daily_stats.trades_taken += 1

    def _send_telegram_alert(self, signal: TradeSignal):
        """Send Telegram alert for new trade."""
        try:
            from libs.notifications.telegram_bot import send_ftmo_trade_alert

            send_ftmo_trade_alert(
                bot_name=signal.bot_name,
                symbol=signal.symbol,
                direction=signal.direction,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                lot_size=signal.lot_size,
                reason=signal.reason,
                paper_mode=self.paper_mode
            )

        except Exception as e:
            logger.warning(f"[Orchestrator] Failed to send Telegram alert: {e}")

    def _send_alert(self, message: str):
        """Send critical alert."""
        try:
            from libs.notifications.telegram_bot import send_ftmo_kill_switch_alert

            # Get current balance and drawdown
            account = self._get_account_info()
            balance = account.get("balance", 15000) if account else 15000
            dd = 0
            if self._daily_stats:
                dd = max(
                    0,
                    (self._daily_stats.starting_balance - self._daily_stats.current_balance)
                    / self._daily_stats.starting_balance * 100
                )

            send_ftmo_kill_switch_alert(message, balance, dd)
        except Exception:
            pass

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status summary."""
        status = {
            "running": self._running,
            "paper_mode": self.paper_mode,
            "kill_switch": self._kill_switch,
            "open_positions": len(self._open_positions),
            "max_positions": self.limits.max_concurrent_positions,
            "metalearning_enabled": self.enable_metalearning,
            "zmq_enabled": self.use_zmq,
            "zmq_connected": self._zmq_client.is_connected if self._zmq_client else False,
            "daily_stats": {
                "trades": self._daily_stats.trades_taken if self._daily_stats else 0,
                "starting_balance": self._daily_stats.starting_balance if self._daily_stats else 0,
                "current_balance": self._daily_stats.current_balance if self._daily_stats else 0,
            },
            "bots": {name: bot.get_status() for name, bot in self.bots.items()},
        }

        # Add metalearning stats if enabled
        if self._metalearner:
            status["metalearning"] = self._metalearner.get_stats()

        return status

    def set_paper_mode(self, paper_mode: bool):
        """Set paper mode for all bots."""
        self.paper_mode = paper_mode
        for bot in self.bots.values():
            bot.config.paper_mode = paper_mode
        logger.info(f"[Orchestrator] Paper mode set to: {paper_mode}")


# Singleton
_orchestrator: Optional[FTMOOrchestrator] = None
_orch_lock = threading.Lock()


def get_ftmo_orchestrator(paper_mode: bool = True) -> FTMOOrchestrator:
    """Get FTMO orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        with _orch_lock:
            if _orchestrator is None:
                _orchestrator = FTMOOrchestrator(paper_mode=paper_mode)
    return _orchestrator
