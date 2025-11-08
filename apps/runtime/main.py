"""Runtime loop: scanning coins, generating signals, enforcing FTMO rules.

This is the production runtime that continuously scans coins and emits trading signals
when confidence thresholds are met and FTMO rules allow.
"""
import os
import time
from datetime import datetime

from loguru import logger

from apps.runtime.ftmo_rules import (
    check_daily_loss_limit,
    check_position_size,
    check_total_loss_limit,
)
from apps.runtime.rate_limiter import RateLimiter
from libs.config.config import Settings
from libs.constants import (
    CONFIDENCE_THRESHOLD,
    INITIAL_BALANCE,
    MAX_HIGH_TIER_SIGNALS_PER_HOUR,
    MAX_SIGNALS_PER_HOUR,
    RISK_PER_TRADE,
    TIER_HIGH_CONFIDENCE,
    TIER_MEDIUM_CONFIDENCE,
)
from libs.db.models import RiskBookSnapshot, Signal, create_tables, get_session


class TradingRuntime:
    """Main trading runtime loop."""

    def __init__(self, config: Settings | None = None):
        """
        Initialize trading runtime.

        Args:
            config: Settings object (if None, loads from environment)
        """
        self.config = config or Settings()
        self.kill_switch = self.config.kill_switch
        self.confidence_threshold = self.config.confidence_threshold

        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            max_signals_per_hour=MAX_SIGNALS_PER_HOUR,
            max_high_tier_per_hour=MAX_HIGH_TIER_SIGNALS_PER_HOUR,
        )

        # Account state
        self.initial_balance = INITIAL_BALANCE
        self.current_balance = INITIAL_BALANCE
        self.daily_pnl = 0.0

        # Database
        create_tables(self.config.db_url)
        logger.info(f"✅ Database initialized: {self.config.db_url}")

    def classify_tier(self, confidence: float) -> str:
        """
        Classify signal tier based on confidence.

        Args:
            confidence: Signal confidence (0-1)

        Returns:
            Tier string ('high', 'medium', or 'low')
        """
        if confidence >= TIER_HIGH_CONFIDENCE:
            return "high"
        elif confidence >= TIER_MEDIUM_CONFIDENCE:
            return "medium"
        else:
            return "low"

    def generate_mock_signal(self) -> dict:
        """
        Generate a mock trading signal.

        TODO: Replace this with actual model inference in production.

        Returns:
            Dictionary containing signal information
        """
        import random

        symbols = ["BTC-USD", "ETH-USD"]
        symbol = random.choice(symbols)

        # Mock model predictions (TODO: Replace with real model inference)
        lstm_pred = random.uniform(0.50, 0.85)
        transformer_pred = random.uniform(0.50, 0.85)
        rl_pred = random.uniform(0.50, 0.85)

        # Ensemble prediction (weighted average)
        ensemble = lstm_pred * 0.35 + transformer_pred * 0.40 + rl_pred * 0.25

        direction = "long" if ensemble >= 0.5 else "short"
        tier = self.classify_tier(ensemble)

        return {
            "symbol": symbol,
            "confidence": ensemble,
            "tier": tier,
            "direction": direction,
            "lstm_prediction": lstm_pred,
            "transformer_prediction": transformer_pred,
            "rl_prediction": rl_pred,
            "entry_price": 50000.0 if symbol == "BTC-USD" else 3000.0,  # Mock price
        }

    def check_ftmo_rules(self) -> bool:
        """
        Check if FTMO rules allow trading.

        Returns:
            True if all rules pass, False otherwise
        """
        # Check daily loss limit
        if not check_daily_loss_limit(self.current_balance, self.daily_pnl):
            return False

        # Check total loss limit
        if not check_total_loss_limit(self.initial_balance, self.current_balance):
            return False

        # Check position sizing
        position_size = self.current_balance * RISK_PER_TRADE
        if not check_position_size(self.current_balance, position_size):
            return False

        return True

    def record_signal_to_db(self, signal_data: dict) -> None:
        """
        Record signal to database.

        Args:
            signal_data: Signal information dictionary
        """
        try:
            session = get_session(self.config.db_url)

            signal = Signal(
                timestamp=datetime.now(),
                symbol=signal_data["symbol"],
                direction=signal_data["direction"],
                confidence=signal_data["confidence"],
                tier=signal_data["tier"],
                lstm_prediction=signal_data.get("lstm_prediction"),
                transformer_prediction=signal_data.get("transformer_prediction"),
                rl_prediction=signal_data.get("rl_prediction"),
                ensemble_prediction=signal_data["confidence"],
                entry_price=signal_data.get("entry_price"),
                latency_ms=signal_data.get("latency_ms", 0.0),
                model_version=self.config.model_version,
                result="pending",
            )

            session.add(signal)
            session.commit()
            logger.info(f"✅ Signal recorded to database: {signal}")
            session.close()

        except Exception as e:
            logger.error(f"❌ Failed to record signal to database: {e}")

    def record_risk_snapshot(self) -> None:
        """Record current risk book state to database."""
        try:
            session = get_session(self.config.db_url)
            stats = self.rate_limiter.get_stats()

            snapshot = RiskBookSnapshot(
                timestamp=datetime.now(),
                balance=self.current_balance,
                equity=self.current_balance,  # Simplified: no open positions
                daily_pnl=self.daily_pnl,
                total_pnl=self.current_balance - self.initial_balance,
                daily_loss_limit=self.current_balance * 0.05,
                total_loss_limit=self.initial_balance * 0.10,
                daily_loss_used_pct=(abs(self.daily_pnl) / (self.current_balance * 0.05))
                if self.daily_pnl < 0
                else 0.0,
                total_loss_used_pct=(
                    abs(self.current_balance - self.initial_balance) / (self.initial_balance * 0.10)
                )
                if self.current_balance < self.initial_balance
                else 0.0,
                signals_last_hour=stats["total_signals_last_hour"],
                high_tier_signals_last_hour=stats["high_tier_signals_last_hour"],
            )

            session.add(snapshot)
            session.commit()
            session.close()

        except Exception as e:
            logger.error(f"❌ Failed to record risk snapshot: {e}")

    def loop_once(self) -> None:
        """Execute one iteration of the runtime loop."""
        if self.kill_switch:
            logger.warning("🛑 Kill-switch is ACTIVE - no signals will be emitted")
            return

        # Generate signal (TODO: Replace with actual model inference)
        signal_data = self.generate_mock_signal()

        logger.info(
            f"📡 Generated signal: {signal_data['symbol']} {signal_data['direction']} "
            f"@ {signal_data['confidence']:.2%} confidence (tier: {signal_data['tier']})"
        )

        # Check confidence threshold
        if signal_data["confidence"] < self.confidence_threshold:
            logger.info(
                f"⏭️  Signal below threshold: {signal_data['confidence']:.2%} < "
                f"{self.confidence_threshold:.2%}"
            )
            return

        # Check FTMO rules
        if not self.check_ftmo_rules():
            logger.warning("⏭️  Signal blocked by FTMO rules")
            return

        # Check rate limits
        if not self.rate_limiter.can_emit_signal(signal_data["tier"]):
            logger.warning("⏭️  Signal blocked by rate limiter")
            return

        # Emit signal
        logger.info(
            f"✅ EMIT SIGNAL: {signal_data['symbol']} {signal_data['direction']} "
            f"@ {signal_data['confidence']:.2%} [{signal_data['tier'].upper()}]"
        )

        # Record to database
        self.record_signal_to_db(signal_data)

        # Update rate limiter
        self.rate_limiter.record_signal(signal_data["tier"])

        # TODO: Send to Telegram bot
        # TODO: Send to MT5 bridge

    def run(self, iterations: int = 10, sleep_seconds: int = 5) -> None:
        """
        Run the trading loop for a specified number of iterations.

        Args:
            iterations: Number of loop iterations (use -1 for infinite loop)
            sleep_seconds: Seconds to sleep between iterations
        """
        logger.info("🚀 Trading runtime starting")
        logger.info(f"   Confidence threshold: {self.confidence_threshold:.2%}")
        logger.info(f"   Kill-switch: {'🛑 ACTIVE' if self.kill_switch else '✅ INACTIVE'}")
        logger.info(f"   Database: {self.config.db_url}")

        iteration = 0
        while iterations == -1 or iteration < iterations:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Iteration {iteration + 1}")
                logger.info(f"{'='*60}")

                self.loop_once()

                # Record risk snapshot every 10 iterations
                if (iteration + 1) % 10 == 0:
                    self.record_risk_snapshot()

                time.sleep(sleep_seconds)
                iteration += 1

            except KeyboardInterrupt:
                logger.info("\n⏹️  Shutting down gracefully...")
                break
            except Exception as e:
                logger.error(f"❌ Error in runtime loop: {e}", exc_info=True)
                time.sleep(sleep_seconds)

        logger.info("👋 Trading runtime exiting")


if __name__ == "__main__":
    # Initialize runtime
    runtime = TradingRuntime()

    # Run for limited iterations (use -1 for infinite loop in production)
    runtime.run(iterations=10, sleep_seconds=3)
