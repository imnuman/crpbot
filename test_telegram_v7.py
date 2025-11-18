"""
Test Telegram Bot Integration for V7

Tests:
1. Telegram connection
2. Test message sending
3. V7 signal formatting (mock signal)
"""
import sys
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any

from libs.notifications import TelegramNotifier
from libs.config.config import Settings


# Mock classes to simulate V7 signal structure
class SignalType:
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class ParsedSignal:
    signal: Any
    confidence: float
    reasoning: str
    timestamp: datetime
    is_valid: bool


@dataclass
class TheoryAnalysis:
    entropy: float
    entropy_interpretation: Dict[str, str]
    hurst_exponent: float
    hurst_interpretation: str
    current_regime: str
    denoised_price: float
    price_momentum: float
    win_rate_estimate: float
    risk_metrics: Dict[str, float]
    regime_name: str
    regime_confidence: float
    sharpe_ratio: float
    var_95: float
    kolmogorov_complexity: float
    fractal_dimension: float


@dataclass
class MockSignalType:
    value: str


@dataclass
class SignalGenerationResult:
    parsed_signal: ParsedSignal
    theory_analysis: TheoryAnalysis
    total_cost_usd: float


def create_mock_signal(signal_value: str = "BUY") -> SignalGenerationResult:
    """Create a mock V7 signal for testing"""

    mock_signal = MockSignalType(value=signal_value)

    parsed_signal = ParsedSignal(
        signal=mock_signal,
        confidence=0.72,
        reasoning="Strong bullish momentum detected with low entropy (predictable market). "
                  "Hurst exponent indicates trending behavior. Market regime classified as 'bull_volatile'. "
                  "Monte Carlo simulation shows favorable risk/reward ratio.",
        timestamp=datetime.utcnow(),
        is_valid=True
    )

    theory_analysis = TheoryAnalysis(
        entropy=0.35,
        entropy_interpretation={"predictability": "High", "randomness": "Low"},
        hurst_exponent=0.65,
        hurst_interpretation="Trending",
        current_regime="bull_volatile",
        denoised_price=67432.50,
        price_momentum=0.0023,
        win_rate_estimate=0.68,
        risk_metrics={
            "sharpe_ratio": 1.85,
            "var_95": -0.032,
            "volatility": 0.045
        },
        regime_name="bull_volatile",
        regime_confidence=0.82,
        sharpe_ratio=1.85,
        var_95=-0.032,
        kolmogorov_complexity=0.42,
        fractal_dimension=1.45
    )

    return SignalGenerationResult(
        parsed_signal=parsed_signal,
        theory_analysis=theory_analysis,
        total_cost_usd=0.000342
    )


def main():
    print("=" * 80)
    print("V7 TELEGRAM BOT INTEGRATION TEST")
    print("=" * 80)

    # Load settings
    config = Settings()

    # Initialize Telegram notifier
    print(f"\n1. Initializing Telegram notifier...")
    print(f"   Token: {config.telegram_token[:20]}..." if config.telegram_token else "   Token: (not set)")
    print(f"   Chat ID: {config.telegram_chat_id}")

    telegram = TelegramNotifier(
        token=config.telegram_token,
        chat_id=config.telegram_chat_id,
        enabled=bool(config.telegram_token and config.telegram_chat_id)
    )

    if not telegram.enabled:
        print("\n❌ Telegram not configured. Please set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID in .env")
        sys.exit(1)

    print("   ✅ Telegram notifier initialized")

    # Test 1: Send test message
    print(f"\n2. Sending test message...")
    success = telegram.send_test_message()

    if success:
        print("   ✅ Test message sent successfully")
    else:
        print("   ❌ Failed to send test message")
        sys.exit(1)

    # Test 2: Send mock BUY signal
    print(f"\n3. Sending mock BUY signal...")
    buy_signal = create_mock_signal("BUY")
    success = telegram.send_v7_signal("BTC-USD", buy_signal)

    if success:
        print("   ✅ BUY signal sent successfully")
    else:
        print("   ❌ Failed to send BUY signal")

    # Test 3: Send mock SELL signal
    print(f"\n4. Sending mock SELL signal...")
    sell_signal = create_mock_signal("SELL")
    success = telegram.send_v7_signal("ETH-USD", sell_signal)

    if success:
        print("   ✅ SELL signal sent successfully")
    else:
        print("   ❌ Failed to send SELL signal")

    # Test 4: Send mock HOLD signal
    print(f"\n5. Sending mock HOLD signal...")
    hold_signal = create_mock_signal("HOLD")
    success = telegram.send_v7_signal("SOL-USD", hold_signal)

    if success:
        print("   ✅ HOLD signal sent successfully")
    else:
        print("   ❌ Failed to send HOLD signal")

    # Test 5: Send runtime status
    print(f"\n6. Sending runtime status notification...")
    success = telegram.send_runtime_status(
        "Started",
        "Symbols: BTC-USD, ETH-USD, SOL-USD\nScan Interval: 120s\nRate Limit: 6 signals/hour"
    )

    if success:
        print("   ✅ Runtime status sent successfully")
    else:
        print("   ❌ Failed to send runtime status")

    print("\n" + "=" * 80)
    print("TELEGRAM TEST COMPLETE")
    print("=" * 80)
    print("\n✅ Check your Telegram app for 6 messages:")
    print("   1. Test message")
    print("   2. BTC-USD BUY signal")
    print("   3. ETH-USD SELL signal")
    print("   4. SOL-USD HOLD signal")
    print("   5. Runtime started status")
    print("\n")


if __name__ == "__main__":
    main()
