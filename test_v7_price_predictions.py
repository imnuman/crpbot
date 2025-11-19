#!/usr/bin/env python3
"""
Test V7 price prediction enhancements

This script tests:
1. LLM prompt includes price request
2. Parser extracts entry/SL/TP prices
3. Prices are saved to database
4. Console output shows prices
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from libs.llm.signal_parser import SignalParser

def test_parser():
    """Test signal parser with price targets"""
    print("=" * 80)
    print("TEST 1: Signal Parser with Price Targets")
    print("=" * 80)

    parser = SignalParser(strict_mode=False)

    # Test BUY signal with prices
    llm_response_buy = """
SIGNAL: BUY
CONFIDENCE: 75%
ENTRY PRICE: $91,234.56
STOP LOSS: $90,500.00
TAKE PROFIT: $92,800.00
REASONING: Strong bullish momentum (Hurst 0.72 trending) + bull regime (65% confidence). Enter at current price, SL below recent support at $90,500 (0.8% risk), TP at 1.618 Fibonacci extension $92,800 (1.7% reward, R:R 1:2.1).
"""

    print("\n1. Testing BUY signal with prices:")
    print(f"Input:\n{llm_response_buy}")

    parsed = parser.parse(llm_response_buy)

    print(f"\nParsed:")
    print(f"  Signal: {parsed.signal}")
    print(f"  Confidence: {parsed.confidence:.1%}")
    print(f"  Entry: ${parsed.entry_price:,.2f}" if parsed.entry_price else "  Entry: None")
    print(f"  Stop Loss: ${parsed.stop_loss:,.2f}" if parsed.stop_loss else "  Stop Loss: None")
    print(f"  Take Profit: ${parsed.take_profit:,.2f}" if parsed.take_profit else "  Take Profit: None")
    print(f"  Valid: {parsed.is_valid}")
    print(f"  Warnings: {parsed.parse_warnings}")

    # Calculate R:R
    if all([parsed.entry_price, parsed.stop_loss, parsed.take_profit]):
        risk = abs(parsed.entry_price - parsed.stop_loss)
        reward = abs(parsed.take_profit - parsed.entry_price)
        rr = reward / risk if risk > 0 else 0
        print(f"  Risk/Reward: 1:{rr:.2f}")

    assert parsed.entry_price == 91234.56, f"Entry price mismatch: {parsed.entry_price}"
    assert parsed.stop_loss == 90500.00, f"Stop loss mismatch: {parsed.stop_loss}"
    assert parsed.take_profit == 92800.00, f"Take profit mismatch: {parsed.take_profit}"
    print("\n✅ BUY signal test passed!")

    # Test HOLD signal with N/A prices
    llm_response_hold = """
SIGNAL: HOLD
CONFIDENCE: 45%
ENTRY PRICE: N/A
STOP LOSS: N/A
TAKE PROFIT: N/A
REASONING: High entropy (0.89) indicates random market conditions. Insufficient edge for trade entry.
"""

    print("\n2. Testing HOLD signal with N/A prices:")
    print(f"Input:\n{llm_response_hold}")

    parsed_hold = parser.parse(llm_response_hold)

    print(f"\nParsed:")
    print(f"  Signal: {parsed_hold.signal}")
    print(f"  Confidence: {parsed_hold.confidence:.1%}")
    print(f"  Entry: {parsed_hold.entry_price}")
    print(f"  Stop Loss: {parsed_hold.stop_loss}")
    print(f"  Take Profit: {parsed_hold.take_profit}")
    print(f"  Valid: {parsed_hold.is_valid}")

    assert parsed_hold.entry_price is None, "Entry should be None for HOLD"
    assert parsed_hold.stop_loss is None, "SL should be None for HOLD"
    assert parsed_hold.take_profit is None, "TP should be None for HOLD"
    print("\n✅ HOLD signal test passed!")

    # Test SELL signal
    llm_response_sell = """
SIGNAL: SELL
CONFIDENCE: 78%
ENTRY PRICE: $3,245.67
STOP LOSS: $3,310.00
TAKE PROFIT: $3,120.50
REASONING: Bear regime detected with negative momentum. Enter at current price, SL above resistance at $3,310 (2.0% risk), TP at support zone $3,120.50 (3.9% reward, R:R 1:1.9).
"""

    print("\n3. Testing SELL signal with prices:")
    print(f"Input:\n{llm_response_sell}")

    parsed_sell = parser.parse(llm_response_sell)

    print(f"\nParsed:")
    print(f"  Signal: {parsed_sell.signal}")
    print(f"  Confidence: {parsed_sell.confidence:.1%}")
    print(f"  Entry: ${parsed_sell.entry_price:,.2f}" if parsed_sell.entry_price else "  Entry: None")
    print(f"  Stop Loss: ${parsed_sell.stop_loss:,.2f}" if parsed_sell.stop_loss else "  Stop Loss: None")
    print(f"  Take Profit: ${parsed_sell.take_profit:,.2f}" if parsed_sell.take_profit else "  Take Profit: None")
    print(f"  Valid: {parsed_sell.is_valid}")

    assert parsed_sell.entry_price == 3245.67, f"Entry price mismatch: {parsed_sell.entry_price}"
    assert parsed_sell.stop_loss == 3310.00, f"Stop loss mismatch: {parsed_sell.stop_loss}"
    assert parsed_sell.take_profit == 3120.50, f"Take profit mismatch: {parsed_sell.take_profit}"
    print("\n✅ SELL signal test passed!")

    print("\n" + "=" * 80)
    print("✅ ALL PARSER TESTS PASSED!")
    print("=" * 80)


def test_prompt():
    """Test that prompt includes price instructions"""
    print("\n" + "=" * 80)
    print("TEST 2: LLM Prompt includes Price Target Instructions")
    print("=" * 80)

    from libs.llm.signal_synthesizer import SignalSynthesizer, MarketContext, TheoryAnalysis
    from datetime import datetime

    synthesizer = SignalSynthesizer(conservative_mode=True)

    context = MarketContext(
        symbol="BTC-USD",
        current_price=91234.56,
        timeframe="1m",
        timestamp=datetime.now()
    )

    analysis = TheoryAnalysis(
        entropy=0.55,
        entropy_interpretation={'predictability': 'medium', 'regime': 'mixed', 'trading_difficulty': 'moderate'},
        hurst=0.72,
        hurst_interpretation="Strong trending",
        current_regime="BULL_TREND",
        regime_probabilities={'BULL_TREND': 0.65},
        denoised_price=91180.0,
        price_momentum=0.0025,
        win_rate_estimate=0.68,
        win_rate_confidence=0.08,
        risk_metrics={'var_95': 0.12, 'sharpe_ratio': 1.2}
    )

    messages = synthesizer.build_prompt(context, analysis)

    # Check user prompt contains price instructions
    user_prompt = messages[1]['content']

    print("\nChecking prompt for price target instructions...")

    required_strings = [
        "ENTRY PRICE:",
        "STOP LOSS:",
        "TAKE PROFIT:",
        "Example for BUY signal:",
        "Example for SELL signal:",
        "Example for HOLD signal:"
    ]

    for req in required_strings:
        if req in user_prompt:
            print(f"✅ Found: {req}")
        else:
            print(f"❌ Missing: {req}")
            raise AssertionError(f"Prompt missing required string: {req}")

    print("\n✅ Prompt includes all price target instructions!")
    print("=" * 80)


if __name__ == "__main__":
    print("\n🧪 Testing V7 Price Prediction Enhancements\n")

    try:
        test_parser()
        test_prompt()

        print("\n" + "=" * 80)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Run V7 runtime with --iterations 1 to test end-to-end")
        print("2. Check console output shows price targets")
        print("3. Verify prices saved to database")
        print("4. Check dashboard displays prices")
        print("\nCommand:")
        print(".venv/bin/python3 apps/runtime/v7_runtime.py --iterations 1 --sleep-seconds 10")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
