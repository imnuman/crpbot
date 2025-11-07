#!/usr/bin/env python3
"""Test the execution model with mock metrics."""
from datetime import datetime, timezone

from loguru import logger

from libs.rl_env.execution_model import ExecutionModel
from libs.rl_env.execution_metrics import (
    ExecutionMetrics,
    measure_execution_metrics,
    save_execution_metrics,
)
from apps.mt5_bridge.interface import MockMT5Bridge


def test_execution_model():
    """Test execution model functionality."""
    logger.info("Testing execution model...")

    # Create mock bridge
    bridge = MockMT5Bridge()

    # Generate mock metrics
    logger.info("Generating mock execution metrics...")
    metrics = measure_execution_metrics(
        bridge=bridge, symbols=["BTC-USD", "ETH-USD"], days=7, min_samples_per_session=10
    )

    # Save metrics
    metrics_file = save_execution_metrics(metrics)
    logger.info(f"✅ Saved metrics to {metrics_file}")

    # Load execution model
    logger.info("Loading execution model...")
    exec_model = ExecutionModel(metrics_file=metrics_file)

    # Test getting metrics
    logger.info("\n📊 Testing metric retrieval:")
    for symbol in ["BTC-USD", "ETH-USD"]:
        for session in ["tokyo", "london", "new_york"]:
            metrics_obj = exec_model.get_metrics(symbol, session)
            logger.info(
                f"  {symbol}/{session}: "
                f"spread={metrics_obj.spread_bps_mean:.2f}bps (p90={metrics_obj.spread_bps_p90:.2f}), "
                f"slippage={metrics_obj.slippage_bps_mean:.2f}bps (p90={metrics_obj.slippage_bps_p90:.2f})"
            )

    # Test sampling
    logger.info("\n🎲 Testing spread/slippage sampling:")
    symbol = "BTC-USD"
    session = "london"
    timestamp = datetime.now(timezone.utc)

    spreads = [exec_model.sample_spread(symbol, session, timestamp) for _ in range(10)]
    slippages = [
        exec_model.sample_slippage(symbol, session, timestamp) for _ in range(10)
    ]

    logger.info(f"  Spread samples: mean={sum(spreads)/len(spreads):.2f}bps, "
                f"min={min(spreads):.2f}bps, max={max(spreads):.2f}bps")
    logger.info(f"  Slippage samples: mean={sum(slippages)/len(slippages):.2f}bps, "
                f"min={min(slippages):.2f}bps, max={max(slippages):.2f}bps")

    # Test execution cost calculation
    logger.info("\n💰 Testing execution cost calculation:")
    entry_price = 50000.0  # BTC price
    costs_long = []
    costs_short = []

    for _ in range(10):
        cost = exec_model.calculate_execution_cost(entry_price, symbol, session, timestamp)
        costs_long.append(cost)

        actual_entry_long = exec_model.apply_execution_cost(
            entry_price, symbol, session, timestamp, direction="long"
        )
        actual_entry_short = exec_model.apply_execution_cost(
            entry_price, symbol, session, timestamp, direction="short"
        )
        costs_short.append(actual_entry_short)

    avg_cost = sum(costs_long) / len(costs_long)
    logger.info(f"  Entry price: ${entry_price:,.2f}")
    logger.info(f"  Average execution cost: ${avg_cost:.2f} ({avg_cost/entry_price*10000:.2f}bps)")
    logger.info(
        f"  Long entry: ${sum([exec_model.apply_execution_cost(entry_price, symbol, session, timestamp, direction='long') for _ in range(10)])/10:.2f}"
    )
    logger.info(
        f"  Short entry: ${sum([exec_model.apply_execution_cost(entry_price, symbol, session, timestamp, direction='short') for _ in range(10)])/10:.2f}"
    )

    # Test latency penalty
    logger.info("\n⏱️  Testing latency penalty:")
    normal_latency = 200  # ms
    high_latency = 600  # ms (exceeds 500ms budget)

    slippage_normal = exec_model.sample_slippage(
        symbol, session, timestamp, latency_ms=normal_latency, latency_budget_ms=500
    )
    slippage_high = exec_model.sample_slippage(
        symbol, session, timestamp, latency_ms=high_latency, latency_budget_ms=500
    )

    logger.info(f"  Normal latency ({normal_latency}ms): slippage={slippage_normal:.2f}bps")
    logger.info(f"  High latency ({high_latency}ms): slippage={slippage_high:.2f}bps (p90)")
    logger.info(
        f"  Latency penalty: {slippage_high - slippage_normal:.2f}bps additional slippage"
    )

    logger.info("\n✅ Execution model tests complete!")


if __name__ == "__main__":
    test_execution_model()

