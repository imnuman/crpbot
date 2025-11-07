"""Execution metrics measurement and storage system."""
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from apps.mt5_bridge.interface import MT5BridgeInterface, MockMT5Bridge
from apps.trainer.features import get_trading_session
from libs.config.config import Settings
from libs.rl_env.execution_model import ExecutionMetrics


def measure_execution_metrics(
    bridge: MT5BridgeInterface,
    symbols: list[str],
    days: int = 7,
    min_samples_per_session: int = 10,
) -> dict[str, dict[str, ExecutionMetrics]]:
    """
    Measure execution metrics (spreads/slippage) from FTMO bridge.

    Args:
        bridge: MT5 bridge interface (can be MockMT5Bridge for testing)
        symbols: List of trading pairs to measure
        days: Number of days of historical data to analyze
        min_samples_per_session: Minimum samples required per session

    Returns:
        Dictionary: {symbol: {session: ExecutionMetrics}}
    """
    logger.info(f"Measuring execution metrics for {symbols} over {days} days")

    all_metrics: dict[str, dict[str, ExecutionMetrics]] = {}

    for symbol in symbols:
        logger.info(f"Measuring {symbol}...")
        symbol_metrics: dict[str, ExecutionMetrics] = {}

        # Collect spread and slippage samples per session
        sessions_data: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {"spreads": [], "slippages": []}
        )

        # Get historical data from bridge
        # Note: This is a placeholder - actual implementation depends on bridge interface
        # For now, we'll use mock data or estimate from available data

        try:
            # TODO: Implement actual measurement from FTMO bridge
            # For now, estimate from Coinbase data or use mock values
            # This will be replaced when FTMO bridge is connected

            # Placeholder: Generate mock samples for each session
            for session in ["tokyo", "london", "new_york"]:
                # Mock data generation (replace with real FTMO measurements)
                n_samples = min_samples_per_session * 2  # Generate enough samples

                # Generate realistic spread/slippage samples
                # These will be replaced by actual FTMO measurements
                if symbol == "BTC-USD":
                    spread_base = 10.0  # Base spread in bps
                    slippage_base = 2.5  # Base slippage in bps
                elif symbol == "ETH-USD":
                    spread_base = 12.0
                    slippage_base = 3.0
                else:
                    spread_base = 15.0
                    slippage_base = 4.0

                # Session-specific adjustments
                session_multipliers = {"tokyo": 1.0, "london": 1.2, "new_york": 1.1}
                multiplier = session_multipliers.get(session, 1.0)

                spreads = np.random.normal(spread_base * multiplier, 2.0, n_samples)
                spreads = np.clip(spreads, 1.0, 50.0)  # Reasonable bounds

                slippages = np.random.normal(slippage_base * multiplier, 1.0, n_samples)
                slippages = np.clip(slippages, 0.5, 15.0)

                sessions_data[session]["spreads"] = spreads.tolist()
                sessions_data[session]["slippages"] = slippages.tolist()

        except Exception as e:
            logger.error(f"Error measuring {symbol}: {e}")
            continue

        # Calculate statistics for each session
        for session, data in sessions_data.items():
            spreads = data["spreads"]
            slippages = data["slippages"]

            if len(spreads) < min_samples_per_session:
                logger.warning(
                    f"Insufficient samples for {symbol}/{session}: "
                    f"{len(spreads)} < {min_samples_per_session}"
                )
                continue

            # Calculate percentiles
            spread_p50 = np.percentile(spreads, 50)
            spread_p90 = np.percentile(spreads, 90)
            slippage_p50 = np.percentile(slippages, 50)
            slippage_p90 = np.percentile(slippages, 90)

            metrics = ExecutionMetrics(
                spread_bps_mean=float(np.mean(spreads)),
                spread_bps_p50=float(spread_p50),
                spread_bps_p90=float(spread_p90),
                slippage_bps_mean=float(np.mean(slippages)),
                slippage_bps_p50=float(slippage_p50),
                slippage_bps_p90=float(slippage_p90),
                sample_count=len(spreads),
                last_updated=datetime.now(timezone.utc).isoformat(),
            )

            symbol_metrics[session] = metrics

            logger.info(
                f"  {symbol}/{session}: "
                f"spread={metrics.spread_bps_mean:.2f}bps (p90={metrics.spread_bps_p90:.2f}), "
                f"slippage={metrics.slippage_bps_mean:.2f}bps (p90={metrics.slippage_bps_p90:.2f}), "
                f"n={metrics.sample_count}"
            )

        all_metrics[symbol] = symbol_metrics

    return all_metrics


def save_execution_metrics(
    metrics: dict[str, dict[str, ExecutionMetrics]], version: str | None = None
) -> Path:
    """
    Save execution metrics to versioned JSON file.

    Args:
        metrics: Dictionary of metrics {symbol: {session: ExecutionMetrics}}
        version: Version string (if None, uses date-based version)

    Returns:
        Path to saved file
    """
    output_dir = Path("data/execution_metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate version if not provided
    if version is None:
        version = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Create filename
    filename = f"execution_metrics_{version}.json"
    filepath = output_dir / filename

    # Convert to JSON-serializable format
    json_data: dict[str, dict[str, dict[str, Any]]] = {}
    for symbol, sessions in metrics.items():
        json_data[symbol] = {}
        for session, metrics_obj in sessions.items():
            json_data[symbol][session] = metrics_obj.to_dict()

    # Save to file
    with open(filepath, "w") as f:
        json.dump(json_data, f, indent=2)

    logger.info(f"Saved execution metrics to {filepath}")

    # Update symlink to latest
    latest_link = output_dir / "execution_metrics.json"
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(filepath.name)

    logger.info(f"Updated symlink: {latest_link} -> {filepath.name}")

    return filepath


def load_execution_metrics(version: str = "latest") -> dict[str, dict[str, ExecutionMetrics]]:
    """
    Load execution metrics from JSON file.

    Args:
        version: Version string or 'latest' for symlink

    Returns:
        Dictionary of metrics {symbol: {session: ExecutionMetrics}}
    """
    output_dir = Path("data/execution_metrics")

    if version == "latest":
        filepath = output_dir / "execution_metrics.json"
    else:
        filepath = output_dir / f"execution_metrics_{version}.json"

    if not filepath.exists():
        raise FileNotFoundError(f"Execution metrics file not found: {filepath}")

    with open(filepath, "r") as f:
        json_data = json.load(f)

    # Convert from JSON to ExecutionMetrics objects
    metrics: dict[str, dict[str, ExecutionMetrics]] = {}
    for symbol, sessions in json_data.items():
        metrics[symbol] = {}
        for session, metrics_data in sessions.items():
            metrics[symbol][session] = ExecutionMetrics.from_dict(metrics_data)

    logger.info(f"Loaded execution metrics from {filepath}")
    return metrics

