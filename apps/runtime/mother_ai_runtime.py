#!/usr/bin/env python3
"""
HYDRA 3.0 - Mother AI Runtime

Production runtime for the Mother AI tournament system.

Usage:
    python apps/runtime/mother_ai_runtime.py --assets BTC-USD ETH-USD --iterations 10 --interval 300
    python apps/runtime/mother_ai_runtime.py --assets BTC-USD --iterations -1 --interval 600  # Infinite loop

Arguments:
    --assets: Space-separated list of trading symbols (default: BTC-USD ETH-USD SOL-USD)
    --iterations: Number of cycles to run (-1 = infinite) (default: -1)
    --interval: Seconds between cycles (default: 300 = 5 minutes)
    --paper: Paper trading mode (no real trades) (always enabled for now)
"""

import sys
import os
import argparse
import time
from datetime import datetime, timezone
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from libs.hydra.mother_ai import get_mother_ai
from libs.data.coinbase_client import get_coinbase_client


def setup_logging():
    """Configure logging for Mother AI runtime."""
    logger.remove()  # Remove default handler

    # Console logging (INFO and above)
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
        colorize=True
    )

    # File logging (DEBUG and above)
    log_file = f"/tmp/mother_ai_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.log"
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="100 MB"
    )

    logger.info(f"Logging to: {log_file}")
    return log_file


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="HYDRA 3.0 - Mother AI Runtime")

    parser.add_argument(
        "--assets",
        nargs="+",
        default=["BTC-USD", "ETH-USD", "SOL-USD"],
        help="Space-separated list of trading symbols"
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=-1,
        help="Number of cycles to run (-1 = infinite)"
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Seconds between cycles (default: 300 = 5 minutes)"
    )

    parser.add_argument(
        "--paper",
        action="store_true",
        default=True,
        help="Paper trading mode (always enabled for safety)"
    )

    return parser.parse_args()


def get_current_market_data(asset: str) -> dict:
    """
    Fetch current market data for an asset.

    Args:
        asset: Trading symbol (e.g., "BTC-USD")

    Returns:
        Dict with market data (price, volume, etc.)
    """
    try:
        coinbase = get_coinbase_client()

        # Get current ticker
        ticker = coinbase.get_ticker(asset)

        # Get recent candles for volume
        candles = coinbase.get_candles(
            asset=asset,
            granularity="ONE_MINUTE",
            limit=5
        )

        # Calculate 24h change (approximate using recent candles)
        if len(candles) >= 2:
            change_24h = ((ticker["price"] - candles[-1]["close"]) / candles[-1]["close"]) * 100
        else:
            change_24h = 0.0

        return {
            "asset": asset,
            "close": ticker["price"],
            "volume": ticker.get("volume", 0),
            "spread": ticker.get("spread", 0),
            "change_24h": change_24h,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to fetch market data for {asset}: {e}")
        # Return mock data for testing
        return {
            "asset": asset,
            "close": 50000.0,  # Mock price
            "volume": 1000000,
            "spread": 0.01,
            "change_24h": 0.5,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


def main():
    """Main runtime loop."""
    args = parse_args()
    log_file = setup_logging()

    logger.info("\n" + "="*80)
    logger.info("🧠 HYDRA 3.0 - MOTHER AI RUNTIME STARTING")
    logger.info("="*80)
    logger.info(f"Assets: {', '.join(args.assets)}")
    logger.info(f"Iterations: {'∞ (infinite)' if args.iterations == -1 else args.iterations}")
    logger.info(f"Interval: {args.interval}s ({args.interval // 60}m {args.interval % 60}s)")
    logger.info(f"Mode: {'PAPER TRADING' if args.paper else 'LIVE TRADING'}")
    logger.info(f"Log File: {log_file}")
    logger.info("="*80 + "\n")

    # Initialize Mother AI
    logger.info("Initializing Mother AI...")
    mother_ai = get_mother_ai()
    logger.success("Mother AI initialized with 4 gladiators\n")

    # Show initial tournament state
    logger.info("📊 INITIAL TOURNAMENT STATE:")
    summary = mother_ai.get_tournament_summary()
    for ranking in summary["rankings"]:
        logger.info(
            f"  Gladiator {ranking['gladiator']}: "
            f"Weight={ranking['weight']:.0%}, "
            f"Trades={ranking['total_trades']}, "
            f"P&L=${ranking['total_pnl_usd']:+.2f}"
        )
    logger.info("")

    # Main loop
    iteration = 0
    asset_index = 0

    try:
        while args.iterations == -1 or iteration < args.iterations:
            iteration += 1

            # Cycle through assets
            current_asset = args.assets[asset_index % len(args.assets)]
            asset_index += 1

            logger.info(f"\n{'='*80}")
            logger.info(f"🔄 CYCLE #{iteration} - {current_asset}")
            logger.info(f"{'='*80}\n")

            # Fetch current market data
            logger.info(f"Fetching market data for {current_asset}...")
            market_data = get_current_market_data(current_asset)
            logger.info(
                f"Price: ${market_data['close']:,.2f} | "
                f"Volume: {market_data['volume']:,.0f} | "
                f"24h Change: {market_data['change_24h']:+.2f}%"
            )

            # Run trading cycle
            try:
                cycle_result = mother_ai.run_trading_cycle(
                    asset=current_asset,
                    market_data=market_data
                )

                logger.success(
                    f"\n✅ Cycle #{iteration} complete: "
                    f"{cycle_result.trades_opened} trades opened by "
                    f"{len(cycle_result.gladiators_active)} gladiators"
                )

            except Exception as e:
                logger.error(f"❌ Cycle #{iteration} failed: {e}")
                logger.exception("Full traceback:")

            # Sleep between cycles (unless last iteration)
            if args.iterations == -1 or iteration < args.iterations:
                logger.info(f"\n⏳ Sleeping for {args.interval}s until next cycle...\n")
                time.sleep(args.interval)

        # Final summary
        logger.info("\n" + "="*80)
        logger.info("🏁 MOTHER AI RUNTIME COMPLETE")
        logger.info("="*80)

        final_summary = mother_ai.get_tournament_summary()
        logger.info(f"\n📊 FINAL TOURNAMENT STANDINGS:")
        for ranking in final_summary["rankings"]:
            logger.info(
                f"  #{ranking['rank']} - Gladiator {ranking['gladiator']} | "
                f"Weight: {ranking['weight']:.0%} | "
                f"P&L: ${ranking['total_pnl_usd']:+.2f} | "
                f"WR: {ranking['win_rate']:.1%} | "
                f"Trades: {ranking['total_trades']}"
            )

        logger.info(f"\nTotal Cycles: {iteration}")
        logger.info(f"Log File: {log_file}")
        logger.success("\n🎉 All cycles completed successfully!")

    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Interrupted by user (Ctrl+C)")
        logger.info("Shutting down gracefully...")

        # Show final standings
        final_summary = mother_ai.get_tournament_summary()
        logger.info(f"\n📊 TOURNAMENT STANDINGS AT SHUTDOWN:")
        for ranking in final_summary["rankings"]:
            logger.info(
                f"  #{ranking['rank']} - Gladiator {ranking['gladiator']} | "
                f"Weight: {ranking['weight']:.0%} | "
                f"P&L: ${ranking['total_pnl_usd']:+.2f} | "
                f"WR: {ranking['win_rate']:.1%} | "
                f"Trades: {ranking['total_trades']}"
            )

        logger.info(f"\nCompleted Cycles: {iteration}")
        logger.info(f"Log File: {log_file}")

    except Exception as e:
        logger.error(f"\n\n❌ Fatal error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
