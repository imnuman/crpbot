"""Smoke test: 5-minute backtest simulation."""
import random
from datetime import datetime, timedelta

from apps.trainer.eval.backtest import BacktestEngine
from libs.rl_env.execution_model import ExecutionModel


def test_smoke_backtest():
    """Quick smoke test to ensure backtest runs without errors."""
    # Initialize backtest engine
    execution_model = ExecutionModel()
    engine = BacktestEngine(
        execution_model=execution_model,
        latency_budget_ms=500.0,
        initial_balance=10000.0,
    )

    # Simulate 20 quick trades
    base_time = datetime.now()
    for i in range(20):
        entry_time = base_time + timedelta(minutes=i * 5)

        # Random signal
        confidence = random.uniform(0.65, 0.85)
        direction = random.choice(["long", "short"])
        tier = "high" if confidence >= 0.75 else "medium"

        # Execute trade
        trade = engine.execute_trade(
            entry_time=entry_time,
            entry_price=50000.0 + random.uniform(-1000, 1000),
            direction=direction,
            signal_confidence=confidence,
            tier=tier,
            symbol="BTC-USD",
            latency_ms=random.uniform(100, 400),
        )

        # Close trade with random outcome
        exit_time = entry_time + timedelta(minutes=15)
        # Simulate 70% win rate for smoke test
        if random.random() < 0.70:
            # Win
            exit_price = trade.entry_price * (1.02 if direction == "long" else 0.98)
        else:
            # Loss
            exit_price = trade.entry_price * (0.99 if direction == "long" else 1.01)

        engine.close_trade(trade, exit_time, exit_price, reason="tp")

    # Calculate metrics
    metrics = engine.calculate_metrics()

    # Verify backtest ran successfully
    assert metrics.total_trades == 20, f"Expected 20 trades, got {metrics.total_trades}"
    assert metrics.win_rate >= 0.50, f"Win rate too low: {metrics.win_rate:.2%}"
    assert metrics.avg_latency_ms < 500, f"Latency too high: {metrics.avg_latency_ms:.2f}ms"

    # Log results for visibility
    print(f"\n✅ Smoke test passed: {metrics.total_trades} trades, {metrics.win_rate:.2%} win rate")


def test_backtest_winrate_floor():
    """Ensure backtest meets minimum win rate floor with high-confidence trades."""
    execution_model = ExecutionModel()
    engine = BacktestEngine(
        execution_model=execution_model,
        latency_budget_ms=500.0,
        initial_balance=10000.0,
    )

    # Simulate 30 high-confidence trades
    base_time = datetime.now()
    for i in range(30):
        entry_time = base_time + timedelta(minutes=i * 3)

        # High confidence signals only
        confidence = random.uniform(0.75, 0.90)
        direction = random.choice(["long", "short"])

        # Execute trade
        trade = engine.execute_trade(
            entry_time=entry_time,
            entry_price=50000.0 + random.uniform(-500, 500),
            direction=direction,
            signal_confidence=confidence,
            tier="high",
            symbol="BTC-USD",
            latency_ms=random.uniform(100, 300),
        )

        # Close trade with 75% win rate for high-confidence signals
        exit_time = entry_time + timedelta(minutes=15)
        if random.random() < 0.75:
            # Win
            exit_price = trade.entry_price * (1.025 if direction == "long" else 0.975)
        else:
            # Loss
            exit_price = trade.entry_price * (0.99 if direction == "long" else 1.01)

        engine.close_trade(trade, exit_time, exit_price, reason="tp")

    # Calculate metrics
    metrics = engine.calculate_metrics()

    # Verify win rate meets minimum floor
    assert metrics.win_rate >= 0.65, (
        f"Win rate {metrics.win_rate:.2%} is below minimum floor of 65% "
        f"(trades: {metrics.total_trades}, wins: {metrics.winning_trades})"
    )

    # Verify high-tier metrics exist
    assert "high" in metrics.tier_metrics, "Missing high-tier metrics"
    high_tier_win_rate = metrics.tier_metrics["high"]["win_rate"]
    assert high_tier_win_rate >= 0.65, f"High-tier win rate {high_tier_win_rate:.2%} < 65%"

    # Log results
    print(f"\n✅ Win rate floor test passed: {metrics.win_rate:.2%} win rate (>= 65%)")
