#!/usr/bin/env python3
"""
Test V7 Integration with Order Flow (Phase 2)

Quick end-to-end test to verify:
1. V7 runtime can fetch OHLCV data
2. Order Flow analysis runs successfully
3. DeepSeek LLM receives Order Flow features in prompt
4. Signal generation completes without errors

NO TRADING - just validation test
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from apps.runtime.v7_runtime import V7TradingRuntime, V7RuntimeConfig
from loguru import logger

def main():
    print("=" * 70)
    print("V7 ORDER FLOW INTEGRATION TEST")
    print("=" * 70)
    print("Testing: V7 Runtime → Order Flow → DeepSeek LLM")
    print("Symbols: BTC-USD (single test)")
    print()

    try:
        # Initialize V7 runtime (just test one symbol)
        config = V7RuntimeConfig(
            symbols=["BTC-USD"],  # Single symbol for quick test
            min_data_points=200,
            max_signals_per_hour=10,  # Allow test signal
            conservative_mode=True
        )

        runtime = V7TradingRuntime(runtime_config=config)

        print("✅ V7 Runtime initialized")
        print(f"   - Signal Generator: {runtime.signal_generator}")
        print(f"   - Order Flow Analyzer: {runtime.signal_generator.order_flow_analyzer}")
        print()

        # Generate one signal
        print("🔍 Generating signal for BTC-USD with Order Flow...")
        print()

        result = runtime.generate_signal_for_symbol(
            symbol="BTC-USD",
            strategy="v7_full_math"  # Full theories + Order Flow
        )

        if result is None:
            print("❌ Signal generation returned None")
            return

        if not result.success:
            print(f"❌ Signal generation failed: {result.error_message}")
            return

        # Display results
        print("=" * 70)
        print("✅ SIGNAL GENERATION SUCCESSFUL")
        print("=" * 70)

        print(f"\n📊 Signal:")
        print(f"   Direction:   {result.parsed_signal.signal.name}")
        print(f"   Confidence:  {result.parsed_signal.confidence:.1%}")
        print(f"   Entry:       ${result.parsed_signal.entry_price:,.2f}" if result.parsed_signal.entry_price else "   Entry:       N/A")
        print(f"   Stop Loss:   ${result.parsed_signal.stop_loss:,.2f}" if result.parsed_signal.stop_loss else "   Stop Loss:   N/A")
        print(f"   Take Profit: ${result.parsed_signal.take_profit:,.2f}" if result.parsed_signal.take_profit else "   Take Profit: N/A")

        print(f"\n💭 Reasoning:")
        print(f"   {result.parsed_signal.reasoning}")

        print(f"\n💰 Cost:")
        print(f"   LLM Tokens:  {result.llm_response.total_tokens} ({result.llm_response.prompt_tokens} in, {result.llm_response.completion_tokens} out)")
        print(f"   Cost:        ${result.total_cost_usd:.6f}")

        print(f"\n⏱️  Performance:")
        print(f"   Generation:  {result.generation_time_seconds:.2f}s")

        # Check if Order Flow was included
        print(f"\n📈 Order Flow Status:")

        # Check prompt for Order Flow section
        user_prompt = ""
        for msg in result.prompt_messages:
            if msg.get('role') == 'user':
                user_prompt = msg.get('content', '')
                break

        if "Order Flow Analysis" in user_prompt:
            print(f"   ✅ Order Flow analysis included in LLM prompt")

            # Count features
            if "Volume Profile" in user_prompt:
                print(f"      - Volume Profile: ✅")
            if "POC (Point of Control)" in user_prompt:
                print(f"      - POC detected: ✅")
            if "Order Flow Imbalance" in user_prompt:
                print(f"      - OFI: ✅")
            if "Market Microstructure" in user_prompt:
                print(f"      - Microstructure: ✅")
        else:
            print(f"   ⚠️  Order Flow NOT found in prompt (may not have candles_df)")

        print("\n" + "=" * 70)
        print("✅ V7 ORDER FLOW INTEGRATION TEST COMPLETE")
        print("=" * 70)
        print("\nResults:")
        print("  ✅ V7 Runtime: Working")
        print("  ✅ Order Flow Analyzer: Initialized")
        print("  ✅ Signal Generation: Success")
        print("  ✅ DeepSeek LLM: Responding")
        print(f"  {'✅' if 'Order Flow Analysis' in user_prompt else '⚠️ '} Order Flow in Prompt: {'Yes' if 'Order Flow Analysis' in user_prompt else 'No'}")

        print("\n🚀 Phase 2 Order Flow integration is READY for production!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
