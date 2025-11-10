"""Integration tests for runtime guardrails."""

from apps.runtime import main as runtime_main
from apps.runtime.rate_limiter import RateLimiter
from libs.config.config import Settings


def test_runtime_blocks_signal_when_ftmo_limit_breached(monkeypatch):
    """Runtime should block signals when FTMO limits are breached."""
    config = Settings()
    config.confidence_threshold = 0.6

    runtime = runtime_main.TradingRuntime(config)
    runtime.kill_switch = False
    runtime.rate_limiter = RateLimiter(max_signals_per_hour=10, max_signals_per_hour_high=10)

    # Force FTMO breach (daily loss > 4.5%)
    runtime.current_balance = 9400.0
    runtime.daily_pnl = -600.0  # 6% drawdown

    emitted = []

    monkeypatch.setattr(
        runtime,
        "generate_mock_signal",
        lambda: {
            "symbol": "BTC-USD",
            "confidence": 0.9,
            "tier": "high",
            "direction": "long",
            "lstm_prediction": 0.9,
            "transformer_prediction": 0.9,
            "rl_prediction": 0.9,
            "entry_price": 50000.0,
        },
    )
    monkeypatch.setattr(runtime, "record_signal_to_db", lambda data: emitted.append(data))

    runtime.loop_once()

    assert len(runtime.rate_limiter.signal_history) == 0
    assert emitted == []


def test_runtime_emits_signal_when_limits_ok(monkeypatch):
    """Runtime should emit signals when all guardrails pass."""
    config = Settings()
    config.confidence_threshold = 0.6

    runtime = runtime_main.TradingRuntime(config)
    runtime.kill_switch = False
    runtime.rate_limiter = RateLimiter(max_signals_per_hour=10, max_signals_per_hour_high=10)

    runtime.current_balance = 10000.0
    runtime.daily_pnl = 0.0

    emitted = []

    monkeypatch.setattr(
        runtime,
        "generate_mock_signal",
        lambda: {
            "symbol": "BTC-USD",
            "confidence": 0.9,
            "tier": "high",
            "direction": "long",
            "lstm_prediction": 0.9,
            "transformer_prediction": 0.9,
            "rl_prediction": 0.9,
            "entry_price": 50000.0,
        },
    )
    monkeypatch.setattr(runtime, "record_signal_to_db", lambda data: emitted.append(data))

    runtime.loop_once()

    assert len(runtime.rate_limiter.signal_history) == 1
    assert len(emitted) == 1
