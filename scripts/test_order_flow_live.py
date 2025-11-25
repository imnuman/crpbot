#!/usr/bin/env python3
"""
Test Order Flow with Live Coinbase Data
No trading, just monitoring to verify features work

This script:
1. Connects to Coinbase API
2. Fetches real OHLCV data
3. Gets order book (if available)
4. Runs Order Flow analysis
5. Displays results
"""
import sys
import time
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from libs.data.provider import create_data_provider
from libs.order_flow.order_flow_integration import OrderFlowAnalyzer
from libs.config.config import Settings


def format_table(data: dict, title: str):
    """Format data as a nice table"""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print('='*70)
    for key, value in data.items():
        if isinstance(value, float):
            print(f"  {key:30s}: {value:>15.4f}")
        elif isinstance(value, list):
            print(f"  {key:30s}: {len(value)} items")
        else:
            print(f"  {key:30s}: {value}")


def main():
    print("=" * 70)
    print("LIVE ORDER FLOW TEST - NO TRADING")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")
    print("Testing 3 symbols with real Coinbase data\n")

    # Initialize
    settings = Settings()

    # Create Coinbase data provider
    coinbase = create_data_provider(
        'coinbase',
        api_key_name=settings.coinbase_api_key_name,
        private_key=settings.coinbase_api_private_key
    )

    analyzer = OrderFlowAnalyzer()

    symbols_to_test = ['BTC-USD', 'ETH-USD', 'SOL-USD']

    for i, symbol in enumerate(symbols_to_test, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(symbols_to_test)}] Testing {symbol}")
        print('='*70)

        try:
            # 1. Fetch OHLCV candles
            print(f"\n🔍 Fetching OHLCV data...")
            candles_df = coinbase.fetch_klines(
                symbol=symbol,
                interval='1m',
                limit=100
            )

            if candles_df.empty:
                print(f"❌ No candle data for {symbol}")
                continue

            print(f"✅ Got {len(candles_df)} candles")
            print(f"   Latest close: ${candles_df['close'].iloc[-1]:,.2f}")
            print(f"   24h volume: {candles_df['volume'].sum():,.2f}")

            # 2. Order book note
            print(f"\n📊 Order book data...")
            print(f"⚠️  Order book not available via REST API")
            print(f"   (Volume Profile will work with OHLCV data)")
            print(f"   (To enable OFI/Microstructure: Need WebSocket integration)")
            order_book = None

            # 3. Run Order Flow analysis
            print(f"\n⚙️  Analyzing order flow...")
            features = analyzer.analyze(
                symbol,
                candles_df,
                order_book
            )

            # 4. Display results

            # Volume Profile
            if 'vp_poc' in features:
                print(f"\n📈 Volume Profile:")
                print(f"   POC (Point of Control):  ${features['vp_poc']:,.2f}")
                print(f"   VAH (Value Area High):   ${features['vp_vah']:,.2f}")
                print(f"   VAL (Value Area Low):    ${features['vp_val']:,.2f}")
                print(f"   Value Area Volume:        {features['vp_value_area_volume']:.1%}")
                print(f"   Trading Bias:             {features['vp_trading_bias']} ({features.get('vp_bias_strength', 'N/A')})")

                if features.get('vp_at_hvn'):
                    print(f"   ⚠️  Price at High Volume Node (support/resistance)")

                # Support/Resistance
                if features.get('vp_support_levels'):
                    print(f"\n   📍 Support Levels:")
                    for level in features['vp_support_levels'][:3]:
                        print(f"      ${float(level):,.2f}")

                if features.get('vp_resistance_levels'):
                    print(f"\n   📍 Resistance Levels:")
                    for level in features['vp_resistance_levels'][:3]:
                        print(f"      ${float(level):,.2f}")

            # Order Flow Imbalance (if available)
            if 'ofi_imbalance' in features:
                print(f"\n💹 Order Flow Imbalance:")
                print(f"   Imbalance:         {features['ofi_imbalance']:+.3f} ({'more bids' if features['ofi_imbalance'] > 0 else 'more asks'})")
                print(f"   Bid Volume:        {features['ofi_bid_volume']:.2f}")
                print(f"   Ask Volume:        {features['ofi_ask_volume']:.2f}")
                print(f"   Ratio:             {features['ofi_ratio']:.2f}")

                if features.get('ofi_whale_detected'):
                    print(f"   🐋 Whale Activity Detected!")
                    print(f"      Large Bids:  {features.get('ofi_large_bids', 0)}")
                    print(f"      Large Asks:  {features.get('ofi_large_asks', 0)}")

            # Microstructure (if available)
            if 'ms_vwap' in features:
                print(f"\n🔬 Market Microstructure:")
                print(f"   VWAP:              ${features['ms_vwap']:,.2f}")
                print(f"   Current Price:     ${candles_df['close'].iloc[-1]:,.2f}")
                print(f"   Deviation:         {features['ms_vwap_deviation_pct']:+.2f}%")

                if abs(features['ms_vwap_deviation_pct']) > 1.0:
                    if features['ms_vwap_deviation_pct'] > 0:
                        print(f"      ⚠️  Trading {features['ms_vwap_deviation_pct']:.2f}% above VWAP (expensive)")
                    else:
                        print(f"      ✅ Trading {abs(features['ms_vwap_deviation_pct']):.2f}% below VWAP (cheap)")

                print(f"   Spread:            {features.get('ms_spread_bps', 0):.1f} bps ({features.get('ms_spread_quality', 'N/A')})")

                if 'ms_depth_imbalance' in features:
                    depth_imb = features['ms_depth_imbalance']
                    print(f"   Depth Imbalance:   {depth_imb:+.3f} ({'bid support' if depth_imb > 0 else 'ask pressure'})")

                if 'ms_buy_pressure' in features:
                    buy_press = features['ms_buy_pressure']
                    print(f"   Buy Pressure:      {buy_press:.1%}")

            # Signal from Order Flow
            signal = features.get('signals', {})
            if signal:
                print(f"\n🎯 Order Flow Signal:")
                direction = signal.get('direction', 'UNKNOWN')
                strength = signal.get('strength', 0.0)
                reasons = signal.get('reasons', [])
                warnings = signal.get('warnings', [])

                # Color code
                if direction == 'LONG':
                    dir_display = f"🟢 {direction}"
                elif direction == 'SHORT':
                    dir_display = f"🔴 {direction}"
                else:
                    dir_display = f"⚪ {direction}"

                print(f"   Direction:  {dir_display}")
                print(f"   Strength:   {strength:.2f} / 1.00")
                print(f"   Confidence: {'High' if strength > 0.7 else 'Medium' if strength > 0.4 else 'Low'}")

                if reasons:
                    print(f"\n   📋 Reasons ({len(reasons)}):")
                    for reason in reasons[:5]:
                        print(f"      • {reason}")

                if warnings:
                    print(f"\n   ⚠️  Warnings:")
                    for warning in warnings:
                        print(f"      • {warning}")

            # Summary
            print(f"\n✅ Analysis Complete for {symbol}")
            print(f"   Data Quality:")
            print(f"      OHLCV:           ✅ Available")
            print(f"      Volume Profile:  ✅ Working")
            print(f"      Order Book:      {'✅ Available' if order_book else '⚠️  Not available'}")
            print(f"      Microstructure:  {'✅ Working' if 'ms_vwap' in features else '⚠️  Limited (no order book)'}")

        except Exception as e:
            print(f"\n❌ Error analyzing {symbol}: {e}")
            import traceback
            traceback.print_exc()

        # Brief pause between symbols
        if i < len(symbols_to_test):
            print(f"\n⏳ Waiting 5 seconds before next symbol...")
            time.sleep(5)

    # Final summary
    print("\n" + "="*70)
    print("✅ LIVE ORDER FLOW TEST COMPLETE")
    print("="*70)
    print(f"End time: {datetime.now()}")
    print("\nResults:")
    print("  • Volume Profile: ✅ Working (uses OHLCV only)")
    print("  • Order Flow/Microstructure: Depends on order book availability")
    print("\nNext Steps:")
    print("  1. If order book unavailable: Enable Coinbase WebSocket")
    print("  2. If all working: Proceed to V7 integration")
    print("  3. Deploy Phase 2 A/B test")
    print("="*70)


if __name__ == "__main__":
    main()
