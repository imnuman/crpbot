"""Smoke tests for backtest engine."""
from datetime import datetime, timedelta

import numpy as np

from apps.trainer.eval.backtest import BacktestEngine


def _run_backtest(num_trades: int, win_ratio: float) -> BacktestEngine:
    np.random.seed(42)
    engine = BacktestEngine()
    start_time = datetime(2025, 1, 1, 9, 0)

    for i in range(num_trades):
        entry_time = start_time + timedelta(minutes=5 * i)
        direction = "long" if i % 2 == 0 else "short"
        confidence = 0.8 if direction == "long" else 0.7
        tier = "high" if confidence >= 0.75 else "medium"

        trade = engine.execute_trade(
            entry_time=entry_time,
            entry_price=50000.0,
            direction=direction,
            signal_confidence=confidence,
            tier=tier,
            symbol="BTC-USD",
            latency_ms=200.0,
        )

        if i < int(num_trades * win_ratio):
            exit_price = trade.entry_price * (1.01 if direction == "long" else 0.99)
        else:
            exit_price = trade.entry_price * (0.99 if direction == "long" else 1.01)

        engine.close_trade(trade, entry_time + timedelta(minutes=15), exit_price, reason="tp")

    return engine


def test_smoke_backtest_runs():
    engine = _run_backtest(num_trades=10, win_ratio=0.7)
    metrics = engine.calculate_metrics()

    assert metrics.total_trades == 10
    assert metrics.win_rate >= 0.6
    assert metrics.latency_penalized_pnl != 0.0


def test_backtest_meets_winrate_floor():
    engine = _run_backtest(num_trades=10, win_ratio=0.7)
    metrics = engine.calculate_metrics()

    assert metrics.win_rate >= 0.65
    assert metrics.calibration_error <= 0.30
