#!/usr/bin/env python3
"""
Test Telegram formatting with price predictions

This script tests the enhanced Telegram notification format to ensure
price targets (Entry/SL/TP) and R:R ratios display correctly.
"""

import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from libs.llm.signal_parser import ParsedSignal, SignalType

# Mock classes for testing
@dataclass
class TheoryAnalysis:
    entropy: Optional[float] = None
    hurst: Optional[float] = None
    current_regime: Optional[str] = None
    regime_probabilities: Optional[dict] = None
    risk_metrics: dict = None

    def __post_init__(self):
        if self.risk_metrics is None:
            self.risk_metrics = {}
        if self.regime_probabilities is None:
            self.regime_probabilities = {}

@dataclass
class SignalGenerationResult:
    parsed_signal: ParsedSignal
    theory_analysis: TheoryAnalysis
    total_cost_usd: float

def test_telegram_format():
    """Test Telegram formatting with different signal types"""

    print("=" * 80)
    print("Testing Telegram V7 Signal Formatting with Price Predictions")
    print("=" * 80)

    # Test 1: BUY signal with prices
    print("\n1. Testing BUY signal with price targets:")
    print("-" * 80)

    buy_signal = ParsedSignal(
        signal=SignalType.BUY,
        confidence=0.78,
        reasoning="Strong bullish momentum (Hurst 0.72 trending) + bull regime (65% confidence). Enter at current price, SL below recent support at $90,500 (0.8% risk), TP at 1.618 Fibonacci extension $92,800 (1.7% reward, R:R 1:2.1).",
        raw_response="test",
        is_valid=True,
        timestamp=datetime.utcnow(),
        parse_warnings=[],
        entry_price=91234.56,
        stop_loss=90500.00,
        take_profit=92800.00
    )

    buy_theories = TheoryAnalysis(
        entropy=0.523,
        hurst=0.72,
        current_regime="Bull Trend",
        regime_probabilities={"Bull Trend": 0.65, "Bear Trend": 0.20, "Consolidation": 0.15},
        risk_metrics={
            'sharpe_ratio': 1.2,
            'var_95': 0.046,
            'profit_probability': 0.68
        }
    )

    buy_result = SignalGenerationResult(
        parsed_signal=buy_signal,
        theory_analysis=buy_theories,
        total_cost_usd=0.000401
    )

    # Format the message (simulating TelegramNotifier.format_v7_signal)
    from libs.notifications.telegram_bot import TelegramNotifier

    # Create a dummy notifier (won't actually send)
    notifier = TelegramNotifier(token="test", chat_id="test", enabled=False)

    buy_message = notifier.format_v7_signal("BTC-USD", buy_result)
    print(buy_message)

    # Test 2: SELL signal with prices
    print("\n\n2. Testing SELL signal with price targets:")
    print("-" * 80)

    sell_signal = ParsedSignal(
        signal=SignalType.SELL,
        confidence=0.81,
        reasoning="Bear regime detected with negative momentum. Enter at current price, SL above resistance at $3,310 (2.0% risk), TP at support zone $3,120.50 (3.9% reward, R:R 1:1.9).",
        raw_response="test",
        is_valid=True,
        timestamp=datetime.utcnow(),
        parse_warnings=[],
        entry_price=3245.67,
        stop_loss=3310.00,
        take_profit=3120.50
    )

    sell_theories = TheoryAnalysis(
        entropy=0.42,
        hurst=0.35,
        current_regime="Bear Trend",
        regime_probabilities={"Bull Trend": 0.15, "Bear Trend": 0.70, "Consolidation": 0.15},
        risk_metrics={
            'sharpe_ratio': -0.8,
            'var_95': 0.052,
            'profit_probability': 0.72
        }
    )

    sell_result = SignalGenerationResult(
        parsed_signal=sell_signal,
        theory_analysis=sell_theories,
        total_cost_usd=0.000398
    )

    sell_message = notifier.format_v7_signal("ETH-USD", sell_result)
    print(sell_message)

    # Test 3: HOLD signal (no prices)
    print("\n\n3. Testing HOLD signal (no price targets):")
    print("-" * 80)

    hold_signal = ParsedSignal(
        signal=SignalType.HOLD,
        confidence=0.35,
        reasoning="High entropy (0.864) shows random conditions conflicting with trending Hurst (0.635), while Kalman momentum is bearish and Monte Carlo shows negative Sharpe (-0.65) with 24.4% profit probability. No clear edge justifies entry.",
        raw_response="test",
        is_valid=True,
        timestamp=datetime.utcnow(),
        parse_warnings=[],
        entry_price=None,
        stop_loss=None,
        take_profit=None
    )

    hold_theories = TheoryAnalysis(
        entropy=0.864,
        hurst=0.635,
        current_regime="Consolidation",
        regime_probabilities={"Bull Trend": 0.0, "Bear Trend": 0.0, "Consolidation": 1.0},
        risk_metrics={
            'sharpe_ratio': -0.65,
            'var_95': 0.046,
            'profit_probability': 0.244
        }
    )

    hold_result = SignalGenerationResult(
        parsed_signal=hold_signal,
        theory_analysis=hold_theories,
        total_cost_usd=0.000401
    )

    hold_message = notifier.format_v7_signal("BTC-USD", hold_result)
    print(hold_message)

    print("\n" + "=" * 80)
    print("✅ Telegram formatting test complete!")
    print("=" * 80)
    print("\nVerify the output above includes:")
    print("  ✓ Entry/SL/TP prices for BUY/SELL signals")
    print("  ✓ Risk % and Reward % calculations")
    print("  ✓ R:R ratio (e.g., 1:2.13)")
    print("  ✓ HOLD signal has no price targets section")
    print("  ✓ All formatting is HTML-compatible (bold tags, etc.)")
    print("=" * 80)

if __name__ == "__main__":
    try:
        test_telegram_format()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
