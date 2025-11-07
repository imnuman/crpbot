"""Runtime loop: scanning + signals + auto-learning."""
import asyncio
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

from apps.runtime.confidence import score_confidence
from apps.runtime.ftmo_rules import FTMOState, check_ftmo_limits
from apps.runtime.logging_config import log_signal, setup_structured_logging
from apps.runtime.rate_limiter import RateLimiter
from apps.runtime.signal import generate_signal
from apps.runtime.telegram_bot import init_bot, send_message
from libs.config.config import Settings

# Global state
runtime_state = {
    "kill_switch": False,
    "running": True,
    "ftmo_state": None,
    "rate_limiter": None,
    "config": None,
    "bot": None,
}


def signal_handler(sig, frame):
    """Handle shutdown signals."""
    logger.info("Received shutdown signal, stopping runtime...")
    runtime_state["running"] = False


async def scan_coins(config: Settings) -> list[dict[str, float]]:
    """
    Scan coins for trading opportunities.

    TODO: This is a stub implementation. In production, this would:
    1. Load models from disk
    2. Fetch latest market data
    3. Run predictions through models
    4. Return predictions

    Args:
        config: Application settings

    Returns:
        List of predictions: [{"pair": "BTC-USD", "lstm": 0.72, "transformer": 0.70, "direction": "long"}, ...]
    """
    # Stub: Return mock predictions for dry-run
    return [
        {"pair": "BTC-USD", "lstm": 0.72, "transformer": 0.70, "rl": 0.0, "direction": "long", "entry_price": 50000.0},
        {"pair": "ETH-USD", "lstm": 0.68, "transformer": 0.65, "rl": 0.0, "direction": "short", "entry_price": 3000.0},
    ]


async def loop_once(config: Settings, mode: str = "dryrun") -> None:
    """Execute one iteration of the runtime loop."""
    # Check kill switch
    if runtime_state["kill_switch"]:
        logger.warning("Kill-switch is ACTIVE - no signals will be emitted")
        return

    # Scan coins
    logger.debug("Scanning coins for trading opportunities...")
    predictions = await scan_coins(config)

    # Process each prediction
    for pred in predictions:
        # Score confidence
        confidence = score_confidence(
            lstm_pred=pred.get("lstm", 0.0),
            transformer_pred=pred.get("transformer", 0.0),
            rl_pred=pred.get("rl", 0.0),
            ensemble_weights=config.ensemble_weights_parsed.__dict__,
        )

        # Check confidence threshold
        if confidence < config.confidence_threshold:
            logger.debug(f"Skipping {pred['pair']}: confidence {confidence:.2%} < threshold {config.confidence_threshold:.2%}")
            continue

        # Determine tier
        from apps.runtime.signal import determine_tier

        tier = determine_tier(confidence)

        # Check rate limiting
        can_emit, reason = runtime_state["rate_limiter"].can_emit(tier)
        if not can_emit:
            logger.warning(f"Rate limit exceeded for {pred['pair']}: {reason}")
            continue

        # Check FTMO limits (dry-run mode: always pass)
        if mode == "live" and runtime_state["ftmo_state"]:
            ftmo_ok, ftmo_reason = check_ftmo_limits(
                runtime_state["ftmo_state"], runtime_state["ftmo_state"].account_balance
            )
            if not ftmo_ok:
                logger.warning(f"FTMO limit check failed for {pred['pair']}: {ftmo_reason}")
                continue

        # Generate signal
        signal = generate_signal(
            pair=pred["pair"],
            direction=pred["direction"],
            entry_price=pred["entry_price"],
            confidence=confidence,
            mode=mode,
            latency_ms=0.0,  # TODO: Measure actual latency
            spread_bps=12.0,  # TODO: Use execution model
            slippage_bps=3.0,  # TODO: Use execution model
        )

        # Log signal
        log_signal(signal)

        # Send Telegram message
        message = signal.format_message()
        await send_message(message, mode=mode)

        # Record signal in rate limiter
        runtime_state["rate_limiter"].record_signal(tier)

        logger.info(f"Emitted {mode} signal: {signal.pair} {signal.direction} @ {signal.confidence:.1%} ({tier})")


async def run_runtime(config: Settings, mode: str = "dryrun", scan_interval: int = 120) -> None:
    """
    Main runtime loop.

    Args:
        config: Application settings
        mode: Runtime mode ('dryrun' or 'live')
        scan_interval: Scanning interval in seconds (default: 120 = 2 minutes)
    """
    logger.info(f"Starting runtime in {mode} mode (scan interval: {scan_interval}s)")

    # Initialize components
    runtime_state["config"] = config
    runtime_state["kill_switch"] = config.kill_switch
    runtime_state["ftmo_state"] = FTMOState(account_balance=10000.0)
    runtime_state["rate_limiter"] = RateLimiter(
        max_signals_per_hour=config.max_signals_per_hour,
        max_signals_per_hour_high=config.max_signals_per_hour_high,
    )

    # Initialize Telegram bot
    if config.telegram_token and config.telegram_chat_id:
        runtime_state["bot"] = init_bot(config)
        await runtime_state["bot"].start()
        await send_message(f"🤖 Runtime started in {mode.upper()} mode", mode=mode)
    else:
        logger.warning("Telegram bot not configured. Notifications will be logged only.")

    # Main loop
    iteration = 0
    while runtime_state["running"]:
        try:
            iteration += 1
            logger.debug(f"Runtime iteration {iteration}")

            # Update kill switch from config (allows runtime updates)
            runtime_state["kill_switch"] = config.kill_switch

            # Run one iteration
            await loop_once(config, mode=mode)

            # Wait for next scan
            if runtime_state["running"]:
                await asyncio.sleep(scan_interval)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            runtime_state["running"] = False
            break
        except Exception as e:
            logger.error(f"Error in runtime loop: {e}", exc_info=True)
            # Continue running (don't crash on errors)
            await asyncio.sleep(scan_interval)

    # Cleanup
    logger.info("Runtime stopping...")
    if runtime_state["bot"]:
        await send_message("🛑 Runtime stopped", mode=mode)
        await runtime_state["bot"].stop()

    logger.info("Runtime stopped")


def main():
    """Main entry point."""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Load config
    config = Settings()
    config.validate()

    # Setup logging
    setup_structured_logging(log_format=config.log_format, log_level=config.log_level)

    # Determine mode
    mode = os.getenv("RUNTIME_MODE", "dryrun").lower()
    if mode not in ["dryrun", "live"]:
        logger.warning(f"Invalid mode '{mode}', defaulting to 'dryrun'")
        mode = "dryrun"

    # Run runtime
    try:
        asyncio.run(run_runtime(config, mode=mode))
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
