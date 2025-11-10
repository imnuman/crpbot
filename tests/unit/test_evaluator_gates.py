"""Unit tests for model evaluator promotion gates."""

from apps.trainer.eval.backtest import BacktestMetrics
from apps.trainer.eval.evaluator import ModelEvaluator


def _dummy_metrics(**overrides):
    base = {
        "total_trades": 50,
        "winning_trades": 40,
        "losing_trades": 10,
        "win_rate": 0.8,
        "total_pnl": 1000.0,
        "avg_pnl_per_trade": 20.0,
        "max_drawdown": 0.05,
        "avg_drawdown": 0.02,
        "sharpe_ratio": 1.5,
        "tier_metrics": {
            "high": {"trades": 20, "win_rate": 0.8, "total_pnl": 600.0, "avg_pnl": 30.0}
        },
        "session_metrics": {
            "london": {"trades": 20, "win_rate": 0.8, "total_pnl": 600.0, "avg_pnl": 30.0}
        },
        "brier_score": 0.02,
        "calibration_error": 0.03,
        "avg_latency_ms": 120.0,
        "p90_latency_ms": 200.0,
        "latency_penalized_pnl": 950.0,
        "hit_rate_by_session": {"london": 0.8},
    }
    base.update(overrides)
    return BacktestMetrics(**base)


def test_promotion_gates_pass():
    evaluator = ModelEvaluator()
    metrics = _dummy_metrics()
    passed, failures = evaluator.check_promotion_gates(metrics, symbol="BTC-USD")

    assert passed is True
    assert failures == []


def test_promotion_gates_fail_calibration():
    evaluator = ModelEvaluator()
    metrics = _dummy_metrics(calibration_error=0.15)
    passed, failures = evaluator.check_promotion_gates(metrics, symbol="BTC-USD")

    assert passed is False
    assert any("Calibration error" in failure for failure in failures)


def test_promotion_gates_fail_accuracy():
    evaluator = ModelEvaluator()
    metrics = _dummy_metrics(win_rate=0.5)
    passed, failures = evaluator.check_promotion_gates(metrics, symbol="BTC-USD")

    assert passed is False
    assert any("Win rate" in failure for failure in failures)
