"""Runtime loop: scanning coins, generating signals, enforcing FTMO rules.

This is the production runtime that continuously scans coins and emits trading signals
when confidence thresholds are met and FTMO rules allow.
"""
import argparse
import asyncio
import os
import time
from datetime import datetime

from loguru import logger

from apps.runtime.ensemble import load_ensemble
from apps.runtime.data_fetcher import get_data_fetcher
from apps.runtime.telegram_bot import init_bot, send_message
from apps.runtime.runtime_features import engineer_runtime_features
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
        self.runtime_mode = self.config.runtime_mode.lower().replace("-", "")

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

        # Market data fetcher
        self.data_fetcher = get_data_fetcher(self.config)
        logger.info("✅ Market data fetcher initialized")

        # Load V5 FIXED ensemble models
        self.symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
        self.ensembles = {}

        logger.info("Loading V5 FIXED models...")
        for symbol in self.symbols:
            try:
                ensemble = load_ensemble(symbol, model_dir="models/promoted")
                self.ensembles[symbol] = ensemble
                logger.info(f"✅ Loaded {symbol}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load {symbol}: {e}")

        # Initialize Telegram bot (will be started in main_async)
        self.telegram_bot = init_bot(self.config)
        if self.telegram_bot and self.telegram_bot.token:
            logger.info("✅ Telegram bot initialized")
        else:
            logger.warning("⚠️  Telegram bot not configured")

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

    def generate_v6_signal(self) -> dict:
        """
        Generate trading signal using V6 Statistical models.
        
        Returns:
            Dictionary containing signal information
        """
        try:
            # Get live market data using data fetcher
            from apps.runtime.data_fetcher import get_data_fetcher
            logger.info("Fetching live market data for V6...")
            
            fetcher = get_data_fetcher()
            features = fetcher.get_live_features()
            
            if not features:
                logger.warning("No live data available, skipping signal generation")
                return None
            
            # Get V6 statistical predictions
            from apps.runtime.ensemble import get_v6_ensemble
            logger.info("Running V6 statistical inference...")
            
            v6_ensemble = get_v6_ensemble()
            predictions = v6_ensemble.predict_v6(features)
            
            if not predictions:
                logger.info("No V6 predictions available")
                return None
            
            # Select best prediction
            best_symbol = None
            best_confidence = 0.0
            
            for symbol, pred in predictions.items():
                if pred['confidence'] > best_confidence:
                    best_confidence = pred['confidence']
                    best_symbol = symbol
            
            if not best_symbol or best_confidence < 0.5:
                logger.info("V6 predictions below confidence threshold")
                return None
            
            best_pred = predictions[best_symbol]
            
            # Get current price (mock for now)
            mock_prices = {"BTC-USD": 50000.0, "ETH-USD": 3000.0, "SOL-USD": 150.0}
            entry_price = mock_prices.get(best_symbol, 1000.0)
            
            # Format signal
            direction = "long" if best_pred['signal'] == 'BUY' else "short"
            tier = self.classify_tier(best_pred['confidence'])
            
            signal_data = {
                "symbol": best_symbol,
                "confidence": best_pred['confidence'],
                "tier": tier,
                "direction": direction,
                "lstm_prediction": best_pred['probability'],
                "transformer_prediction": 0.0,  # V6 statistical only
                "rl_prediction": 0.0,  # V6 statistical only
                "entry_price": entry_price,
                "v6_predictions": predictions,
                "model_type": "v6_statistical"
            }
            
            logger.info(f"Generated V6 signal: {best_symbol} {direction.upper()} @ {best_pred['confidence']:.1%}")
            return signal_data
            
        except Exception as e:
            logger.error(f"V6 signal generation failed: {e}")
            return None

    def generate_v5_signal(self) -> dict:
        """Generate trading signal using V5 FIXED models.

        Returns:
            Dictionary containing signal information or None
        """
        # Scan all symbols and pick best prediction
        best_signal = None
        best_confidence = 0.0

        for symbol in self.symbols:
            if symbol not in self.ensembles:
                continue

            try:
                # Fetch latest market data
                df_raw = self.data_fetcher.fetch_latest_candles(symbol, num_candles=150)

                if df_raw.empty or len(df_raw) < 100:
                    logger.debug(f"Insufficient data for {symbol}: {len(df_raw)} candles")
                    continue

                # Engineer ALL features (including multi-TF) to match training
                df_features = engineer_runtime_features(
                    df=df_raw,
                    symbol=symbol,
                    data_fetcher=self.data_fetcher,
                    include_multi_tf=True,
                    include_coingecko=False  # Use placeholders for now
                )

                if df_features.empty or len(df_features) < 60:
                    logger.debug(f"Feature engineering failed for {symbol}")
                    continue

                # Get V5 ensemble prediction
                ensemble = self.ensembles[symbol]
                prediction = ensemble.predict(df_features)

                confidence = prediction['confidence']

                # Track best prediction
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_signal = {
                        "symbol": symbol,
                        "confidence": confidence,
                        "tier": self.classify_tier(confidence),
                        "direction": prediction['direction'],
                        "lstm_prediction": prediction['lstm_prediction'],
                        "transformer_prediction": prediction.get('transformer_prediction', 0.5),
                        "rl_prediction": prediction.get('rl_prediction', 0.5),
                        "entry_price": float(df_raw['close'].iloc[-1])
                    }

                logger.debug(
                    f"{symbol}: {prediction['direction']} @ {confidence:.1%} "
                    f"(LSTM: {prediction['lstm_prediction']:.3f})"
                )

            except Exception as e:
                logger.warning(f"Failed to generate signal for {symbol}: {e}")
                continue

        if best_signal:
            logger.info(
                f"✅ Best signal: {best_signal['symbol']} {best_signal['direction'].upper()} "
                f"@ {best_signal['confidence']:.1%} [{best_signal['tier'].upper()}]"
            )

        return best_signal

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

    async def send_telegram_notification(self, signal_data: dict) -> None:
        """Send Telegram notification for signal.

        Args:
            signal_data: Signal information dictionary
        """
        if not self.telegram_bot or not self.telegram_bot.token:
            return

        try:
            # Format signal message
            message = (
                f"🚨 **TRADING SIGNAL** 🚨\n\n"
                f"**Symbol:** {signal_data['symbol']}\n"
                f"**Direction:** {signal_data['direction'].upper()}\n"
                f"**Confidence:** {signal_data['confidence']:.2%}\n"
                f"**Tier:** {signal_data['tier'].upper()}\n"
                f"**Entry Price:** ${signal_data.get('entry_price', 0):.2f}\n\n"
                f"**Model Predictions:**\n"
                f"• LSTM: {signal_data.get('lstm_prediction', 0):.3f}\n"
                f"• Transformer: {signal_data.get('transformer_prediction', 0):.3f}\n"
                f"• RL: {signal_data.get('rl_prediction', 0):.3f}\n\n"
                f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

            await send_message(message, mode=self.runtime_mode)
            logger.info("✅ Telegram notification sent")
        except Exception as e:
            logger.error(f"❌ Failed to send Telegram notification: {e}")

    def loop_once(self) -> None:
        """Execute one iteration of the runtime loop."""
        if self.kill_switch:
            logger.warning("🛑 Kill-switch is ACTIVE - no signals will be emitted")
            return

        # Generate signal using V5 FIXED models
        signal_data = self.generate_v5_signal()

        if signal_data is None:
            logger.info("No signal generated this cycle")
            return

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

        # Send Telegram notification (run in background thread to avoid blocking)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.send_telegram_notification(signal_data))
            else:
                asyncio.run(self.send_telegram_notification(signal_data))
        except Exception as e:
            logger.warning(f"Could not send Telegram notification: {e}")

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
        logger.info(f"   Runtime mode: {self.runtime_mode.upper()}")
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


async def main_async():
    """Async main entry point for runtime with Telegram bot support."""
    parser = argparse.ArgumentParser(description="CRPBot trading runtime")
    parser.add_argument(
        "--mode",
        default=os.environ.get("RUNTIME_MODE", "dryrun"),
        choices=["dryrun", "dry-run", "live"],
        help="Runtime mode (default: dryrun)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of loop iterations (-1 for infinite)",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=3.0,
        help="Seconds to sleep between iterations",
    )
    parser.add_argument(
        "--log-level",
        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Override log level for this run",
    )

    args = parser.parse_args()
    normalized_mode = args.mode.lower().replace("-", "")
    os.environ["RUNTIME_MODE"] = normalized_mode

    if args.log_level:
        os.environ["LOG_LEVEL"] = args.log_level

    runtime = TradingRuntime()

    # Start Telegram bot
    if runtime.telegram_bot and runtime.telegram_bot.token:
        await runtime.telegram_bot.start()

        # Send startup notification
        await send_message(
            f"🚀 **Trading Bot Started**\n\n"
            f"Mode: {normalized_mode.upper()}\n"
            f"Confidence Threshold: {runtime.confidence_threshold:.0%}\n"
            f"Kill-switch: {'🛑 ACTIVE' if runtime.kill_switch else '✅ INACTIVE'}",
            mode=normalized_mode
        )

    try:
        runtime.run(iterations=args.iterations, sleep_seconds=args.sleep_seconds)
    finally:
        # Stop Telegram bot on exit
        if runtime.telegram_bot:
            await runtime.telegram_bot.stop()


if __name__ == "__main__":
    asyncio.run(main_async())
