"""Integration tests for runtime guardrails."""

import asyncio

import pytest

from apps.runtime import main as runtime_main
from apps.runtime.ftmo_rules import FTMOState
from apps.runtime.rate_limiter import RateLimiter
from libs.config.config import Settings


@pytest.mark.asyncio
async def test_runtime_blocks_signal_when_ftmo_limit_breached(monkeypatch):
    """Runtime loop should skip emission when FTMO limits are violated."""
    config = Settings()
    config.confidence_threshold = 0.6  # allow high-confidence trades

    runtime_main.runtime_state["kill_switch"] = False
    runtime_main.runtime_state["rate_limiter"] = RateLimiter(
        max_signals_per_hour=10, max_signals_per_hour_high=10
    )
    state = FTMOState(account_balance=10000.0)
    state.update_balance(9400.0)  # daily loss > 4.5%
    runtime_main.runtime_state["ftmo_state"] = state

    emitted = []

    async def fake_send_message(message: str, mode: str = "dryrun") -> None:
        emitted.append(message)

    async def fake_scan_coins(_config: Settings):
        return [
            {
                "pair": "BTC-USD",
                "lstm": 0.9,
                "transformer": 0.9,
                "rl": 0.9,
                "direction": "long",
                "entry_price": 50000.0,
            }
        ]

    monkeypatch.setattr(runtime_main, "send_message", fake_send_message)
    monkeypatch.setattr(runtime_main, "scan_coins", fake_scan_coins)
    monkeypatch.setattr(runtime_main, "log_signal", lambda signal: None)

    await runtime_main.loop_once(config, mode="live")

    assert len(runtime_main.runtime_state["rate_limiter"].signal_history) == 0
    assert emitted == []


@pytest.mark.asyncio
async def test_runtime_emits_signal_when_limits_ok(monkeypatch):
    """Runtime should emit a signal when thresholds and guardrails allow."""
    config = Settings()
    config.confidence_threshold = 0.6

    limiter = RateLimiter(max_signals_per_hour=10, max_signals_per_hour_high=10)
    runtime_main.runtime_state["kill_switch"] = False
    runtime_main.runtime_state["rate_limiter"] = limiter
    runtime_main.runtime_state["ftmo_state"] = FTMOState(account_balance=10000.0)

    emitted = []

    async def fake_send_message(message: str, mode: str = "dryrun") -> None:
        emitted.append(message)

    async def fake_scan_coins(_config: Settings):
        return [
            {
                "pair": "BTC-USD",
                "lstm": 0.9,
                "transformer": 0.9,
                "rl": 0.9,
                "direction": "long",
                "entry_price": 50000.0,
            }
        ]

    monkeypatch.setattr(runtime_main, "send_message", fake_send_message)
    monkeypatch.setattr(runtime_main, "scan_coins", fake_scan_coins)
    monkeypatch.setattr(runtime_main, "log_signal", lambda signal: None)

    await runtime_main.loop_once(config, mode="live")

    assert len(runtime_main.runtime_state["rate_limiter"].signal_history) == 1
    assert len(emitted) == 1
    assert "Runtime started" not in emitted[0]
