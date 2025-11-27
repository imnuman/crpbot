"""
Test HMAS Orchestrator End-to-End
"""
import asyncio
import sys
from libs.hmas.orchestrator.trade_orchestrator import TradeOrchestrator


async def test_hmas_signal_generation():
    """Test complete 4-agent signal generation"""

    print("\n" + "=" * 80)
    print("HMAS END-TO-END TEST")
    print("=" * 80)

    # Initialize orchestrator
    print("\n1. Initializing orchestrator...")
    orchestrator = TradeOrchestrator()
    print(f"✅ {orchestrator}")

    # Test market data (simulated mean reversion setup)
    market_data = {
        'current_price': 1.25500,
        'ma200': 1.25200,           # Price below 200-MA = downtrend
        'rsi': 72,                   # Overbought
        'bbands_upper': 1.25550,    # At upper band
        'bbands_lower': 1.25000,
        'atr': 0.00150              # Average True Range
    }

    print("\n2. Market Data:")
    for key, value in market_data.items():
        print(f"   {key}: {value}")

    # Generate signal
    print("\n3. Generating signal (all 4 agents)...")
    signal = await orchestrator.generate_signal(
        symbol='GBPUSD',
        market_data=market_data,
        spread_pips=1.5,
        fees_pips=0.5
    )

    # Display result
    print("\n" + "=" * 80)
    print("FINAL SIGNAL")
    print("=" * 80)

    print(f"\nDecision: {signal.get('decision', 'UNKNOWN')}")
    print(f"Action: {signal.get('action', 'HOLD')}")

    if signal.get('decision') == 'APPROVED':
        print(f"\nTrade Details:")
        print(f"  Entry: {signal.get('entry', 0):.5f}")
        print(f"  Stop Loss: {signal.get('stop_loss', 0):.5f}")
        print(f"  Take Profit: {signal.get('take_profit', 0):.5f}")
        print(f"  Lot Size: {signal.get('lot_size', 0)} lots")
        print(f"  Risk: {signal.get('risk_percent', 0):.1%}")
        print(f"  R:R Ratio: {signal.get('reward_risk_ratio', 0):.2f}:1")
        print(f"  Confidence: {signal.get('confidence', 0):.0%}")
        print(f"\nFTMO Compliant: {'✅' if signal.get('ftmo_compliant') else '❌'}")

    else:
        print(f"\nRejection Reason: {signal.get('rejection_reason', 'Unknown')}")

    print(f"\nGeneration Time: {signal.get('generation_time_seconds', 0):.2f}s")
    print(f"Agents Used: {', '.join(signal.get('agents_used', []))}")

    # Show rationale if available
    if signal.get('rationale'):
        print(f"\n" + "=" * 80)
        print("RATIONALE (Claude)")
        print("=" * 80)
        print(signal['rationale'][:500] + "..." if len(signal['rationale']) > 500 else signal['rationale'])

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    return signal.get('decision') in ['APPROVED', 'REJECTED']


if __name__ == "__main__":
    success = asyncio.run(test_hmas_signal_generation())
    sys.exit(0 if success else 1)
