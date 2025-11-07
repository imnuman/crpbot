#!/usr/bin/env python3
"""Nightly job to recompute execution metrics from FTMO bridge."""
import sys
from pathlib import Path

from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from apps.mt5_bridge.interface import MT5BridgeInterface, MockMT5Bridge
from libs.config.config import Settings
from libs.rl_env.execution_metrics import measure_execution_metrics, save_execution_metrics


def main():
    """Run nightly execution metrics measurement."""
    logger.info("Starting nightly execution metrics measurement...")

    # Load config
    config = Settings()

    # Create MT5 bridge (will use MockMT5Bridge if FTMO not available)
    # TODO: Replace with actual FTMO bridge when available
    try:
        # Try to create real FTMO bridge
        # bridge = create_ftmo_bridge(config)
        # For now, use mock bridge
        bridge = MockMT5Bridge()
        logger.info("Using MockMT5Bridge (FTMO bridge not yet implemented)")
    except Exception as e:
        logger.warning(f"Failed to create FTMO bridge: {e}. Using MockMT5Bridge.")
        bridge = MockMT5Bridge()

    # Symbols to measure
    symbols = ["BTC-USD", "ETH-USD", "BNB-USD"]

    # Measure metrics over last 7 days
    metrics = measure_execution_metrics(
        bridge=bridge, symbols=symbols, days=7, min_samples_per_session=10
    )

    if not metrics:
        logger.error("No metrics collected. Exiting.")
        sys.exit(1)

    # Save metrics
    metrics_file = save_execution_metrics(metrics)
    logger.info(f"✅ Execution metrics saved to {metrics_file}")

    logger.info("Nightly execution metrics measurement complete!")


if __name__ == "__main__":
    main()

