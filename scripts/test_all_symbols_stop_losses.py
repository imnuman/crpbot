"""
Multi-Symbol Stop Loss Widening Test

Tests the impact of widened stop losses across ALL 10 V7 symbols:
BTC, ETH, SOL, XRP, DOGE, ADA, AVAX, LINK, POL, LTC

Compares:
- OLD: 2% SL, 4% TP (1:2 R:R)
- NEW: 4% SL, 8% TP (1:2 R:R)

This validates that the fix works across different market conditions and asset types.
"""
import sys
sys.path.insert(0, '/root/crpbot')

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

# All V7 symbols
SYMBOLS = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 'LINK', 'POL', 'LTC']

def backtest_with_stops(df: pd.DataFrame, stop_loss_pct: float, take_profit_pct: float, symbol: str) -> dict:
    """Run backtest with specified stop loss and take profit"""

    df = df.copy()

    # Generate MACD signals
    df['macd_prev'] = df['macd'].shift(1)
    df['macd_signal_prev'] = df['macd_signal'].shift(1)

    bullish_cross = (
        (df['macd'] > df['macd_signal']) &
        (df['macd_prev'] <= df['macd_signal_prev'])
    )
    bearish_cross = (
        (df['macd'] < df['macd_signal']) &
        (df['macd_prev'] >= df['macd_signal_prev'])
    )

    long_condition = bullish_cross & (df['rsi_14'] > 50)
    short_condition = bearish_cross & (df['rsi_14'] < 50)

    df['signal'] = 0
    df.loc[long_condition, 'signal'] = 1
    df.loc[short_condition, 'signal'] = -1

    # Backtest
    trades = []
    current_position = 0
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    entry_idx = 0

    for i in range(len(df)):
        row = df.iloc[i]

        # Check exits
        if current_position != 0:
            if current_position == 1:  # Long
                if row['low'] <= stop_loss:
                    pnl_pct = (stop_loss - entry_price) / entry_price * 100
                    hold_minutes = (i - entry_idx) * 60
                    trades.append({
                        'outcome': 'loss',
                        'pnl_pct': pnl_pct,
                        'hold_minutes': hold_minutes,
                        'exit_reason': 'stop_loss'
                    })
                    current_position = 0
                    continue

                if row['high'] >= take_profit:
                    pnl_pct = (take_profit - entry_price) / entry_price * 100
                    hold_minutes = (i - entry_idx) * 60
                    trades.append({
                        'outcome': 'win',
                        'pnl_pct': pnl_pct,
                        'hold_minutes': hold_minutes,
                        'exit_reason': 'take_profit'
                    })
                    current_position = 0
                    continue

            elif current_position == -1:  # Short
                if row['high'] >= stop_loss:
                    pnl_pct = (entry_price - stop_loss) / entry_price * 100
                    hold_minutes = (i - entry_idx) * 60
                    trades.append({
                        'outcome': 'loss',
                        'pnl_pct': pnl_pct,
                        'hold_minutes': hold_minutes,
                        'exit_reason': 'stop_loss'
                    })
                    current_position = 0
                    continue

                if row['low'] <= take_profit:
                    pnl_pct = (entry_price - take_profit) / entry_price * 100
                    hold_minutes = (i - entry_idx) * 60
                    trades.append({
                        'outcome': 'win',
                        'pnl_pct': pnl_pct,
                        'hold_minutes': hold_minutes,
                        'exit_reason': 'take_profit'
                    })
                    current_position = 0
                    continue

        # Check for new signal
        if current_position == 0 and row['signal'] != 0:
            entry_price = row['close']
            entry_idx = i

            if row['signal'] == 1:  # LONG
                stop_loss = entry_price * (1 - stop_loss_pct)
                take_profit = entry_price * (1 + take_profit_pct)
                current_position = 1
            elif row['signal'] == -1:  # SHORT
                stop_loss = entry_price * (1 + stop_loss_pct)
                take_profit = entry_price * (1 - take_profit_pct)
                current_position = -1

    # Calculate metrics
    if not trades:
        return {
            'symbol': symbol,
            'total_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_loss_hold_min': 0,
            'quick_loss_rate': 0
        }

    trades_df = pd.DataFrame(trades)
    wins = len(trades_df[trades_df['outcome'] == 'win'])
    losses = len(trades_df[trades_df['outcome'] == 'loss'])
    win_rate = wins / len(trades_df) * 100
    total_pnl = trades_df['pnl_pct'].sum()

    # Loss analysis
    loss_trades = trades_df[trades_df['outcome'] == 'loss']
    if len(loss_trades) > 0:
        avg_loss_hold = loss_trades['hold_minutes'].mean()
        quick_losses = len(loss_trades[loss_trades['hold_minutes'] < 60])
        quick_loss_rate = quick_losses / len(loss_trades) * 100
    else:
        avg_loss_hold = 0
        quick_loss_rate = 0

    return {
        'symbol': symbol,
        'total_trades': len(trades_df),
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_pnl_pct': trades_df['pnl_pct'].mean(),
        'avg_loss_hold_min': avg_loss_hold,
        'quick_loss_rate': quick_loss_rate
    }


def test_symbol(symbol: str) -> Dict:
    """Test both configurations for a single symbol"""

    data_path = Path(f'data/historical/{symbol}_USD_3600s_730d_features.parquet')

    if not data_path.exists():
        print(f"⚠️  Skipping {symbol}: Data not found")
        return None

    df = pd.read_parquet(data_path)
    df = df.dropna(subset=['rsi_14', 'macd', 'macd_signal', 'close', 'high', 'low'])

    if len(df) < 100:
        print(f"⚠️  Skipping {symbol}: Insufficient data ({len(df)} candles)")
        return None

    # Test both configurations
    old_result = backtest_with_stops(df, 0.02, 0.04, symbol)
    new_result = backtest_with_stops(df, 0.04, 0.08, symbol)

    return {
        'symbol': symbol,
        'candles': len(df),
        'old': old_result,
        'new': new_result
    }


def main():
    print("=" * 100)
    print("MULTI-SYMBOL STOP LOSS WIDENING TEST")
    print("=" * 100)
    print(f"\nTesting {len(SYMBOLS)} symbols: {', '.join(SYMBOLS)}")
    print("\nOLD Config: 2% SL, 4% TP (1:2 R:R)")
    print("NEW Config: 4% SL, 8% TP (1:2 R:R)")
    print("\n" + "=" * 100)

    # Test all symbols
    all_results = []
    failed_symbols = []

    for symbol in SYMBOLS:
        print(f"\n{'=' * 100}")
        print(f"Testing {symbol}")
        print("=" * 100)

        try:
            result = test_symbol(symbol)

            if result is None:
                failed_symbols.append(symbol)
                continue

            all_results.append(result)

            old = result['old']
            new = result['new']

            print(f"\nDataset: {result['candles']} candles (2 years hourly)\n")

            print(f"{'Metric':<25} {'OLD (2% SL)':<20} {'NEW (4% SL)':<20} {'Change':<15}")
            print("-" * 100)
            print(f"{'Total Trades':<25} {old['total_trades']:<20} {new['total_trades']:<20} {new['total_trades'] - old['total_trades']:<15}")
            print(f"{'Win Rate':<25} {old['win_rate']:<20.1f}% {new['win_rate']:<20.1f}% {new['win_rate'] - old['win_rate']:+.1f}%")
            print(f"{'Total P&L':<25} {old['total_pnl']:<20.2f}% {new['total_pnl']:<20.2f}% {new['total_pnl'] - old['total_pnl']:+.2f}%")
            print(f"{'Avg Loss Hold (min)':<25} {old['avg_loss_hold_min']:<20.0f} {new['avg_loss_hold_min']:<20.0f} {new['avg_loss_hold_min'] - old['avg_loss_hold_min']:+.0f}")
            print(f"{'Quick Loss Rate':<25} {old['quick_loss_rate']:<20.1f}% {new['quick_loss_rate']:<20.1f}% {new['quick_loss_rate'] - old['quick_loss_rate']:+.1f}%")

            # Verdict for this symbol
            improvements = 0
            if new['win_rate'] > old['win_rate']:
                improvements += 1
            if new['total_pnl'] > old['total_pnl']:
                improvements += 1
            if new['avg_loss_hold_min'] > old['avg_loss_hold_min']:
                improvements += 1

            if improvements >= 2:
                print(f"\n✅ {symbol}: IMPROVED ({improvements}/3 metrics better)")
            elif improvements == 1:
                print(f"\n⚠️  {symbol}: MIXED (1/3 metrics better)")
            else:
                print(f"\n❌ {symbol}: NO IMPROVEMENT")

        except Exception as e:
            print(f"\n❌ {symbol} FAILED: {e}")
            failed_symbols.append(symbol)

    # Overall summary
    print("\n" + "=" * 100)
    print("OVERALL SUMMARY")
    print("=" * 100)

    if not all_results:
        print("\n❌ No successful tests!")
        return

    # Aggregate statistics
    summary_df = pd.DataFrame([
        {
            'Symbol': r['symbol'],
            'Old_WR': r['old']['win_rate'],
            'New_WR': r['new']['win_rate'],
            'WR_Change': r['new']['win_rate'] - r['old']['win_rate'],
            'Old_PNL': r['old']['total_pnl'],
            'New_PNL': r['new']['total_pnl'],
            'PNL_Change': r['new']['total_pnl'] - r['old']['total_pnl'],
            'Old_Hold': r['old']['avg_loss_hold_min'],
            'New_Hold': r['new']['avg_loss_hold_min'],
            'Hold_Change': r['new']['avg_loss_hold_min'] - r['old']['avg_loss_hold_min']
        }
        for r in all_results
    ])

    print(f"\nTested: {len(all_results)}/{len(SYMBOLS)} symbols")
    if failed_symbols:
        print(f"Failed: {', '.join(failed_symbols)}")

    print("\n" + "-" * 100)
    print(f"{'Symbol':<10} {'Old WR':<12} {'New WR':<12} {'WR Δ':<12} {'Old P&L':<12} {'New P&L':<12} {'P&L Δ':<12}")
    print("-" * 100)

    for _, row in summary_df.iterrows():
        print(f"{row['Symbol']:<10} {row['Old_WR']:>10.1f}% {row['New_WR']:>10.1f}% {row['WR_Change']:>10.1f}% {row['Old_PNL']:>10.1f}% {row['New_PNL']:>10.1f}% {row['PNL_Change']:>+10.1f}%")

    # Averages
    print("-" * 100)
    print(f"{'AVERAGE':<10} {summary_df['Old_WR'].mean():>10.1f}% {summary_df['New_WR'].mean():>10.1f}% {summary_df['WR_Change'].mean():>10.1f}% {summary_df['Old_PNL'].mean():>10.1f}% {summary_df['New_PNL'].mean():>10.1f}% {summary_df['PNL_Change'].mean():>+10.1f}%")

    # Verdict
    print("\n" + "=" * 100)
    print("FINAL VERDICT")
    print("=" * 100)

    symbols_improved = len(summary_df[summary_df['WR_Change'] > 0])
    symbols_pnl_improved = len(summary_df[summary_df['PNL_Change'] > 0])
    avg_wr_change = summary_df['WR_Change'].mean()
    avg_pnl_change = summary_df['PNL_Change'].mean()

    print(f"\nWin Rate:")
    print(f"  - Improved: {symbols_improved}/{len(all_results)} symbols ({symbols_improved/len(all_results)*100:.1f}%)")
    print(f"  - Average change: {avg_wr_change:+.2f}%")

    print(f"\nTotal P&L:")
    print(f"  - Improved: {symbols_pnl_improved}/{len(all_results)} symbols ({symbols_pnl_improved/len(all_results)*100:.1f}%)")
    print(f"  - Average change: {avg_pnl_change:+.2f}%")

    print(f"\nAvg Loss Hold Time:")
    avg_hold_change = summary_df['Hold_Change'].mean()
    print(f"  - Average change: {avg_hold_change:+.0f} minutes")

    # Overall recommendation
    print("\n" + "=" * 100)

    if avg_wr_change > 0 and avg_pnl_change > 0 and symbols_improved >= len(all_results) * 0.6:
        print("✅ RECOMMENDATION: DEPLOY WIDENED STOP LOSSES")
        print(f"   - Consistent improvement across {symbols_improved}/{len(all_results)} symbols")
        print(f"   - Average win rate +{avg_wr_change:.2f}%")
        print(f"   - Average P&L +{avg_pnl_change:.2f}%")
    elif avg_wr_change > 0 or avg_pnl_change > 0:
        print("⚠️  RECOMMENDATION: MIXED RESULTS - DEPLOY WITH CAUTION")
        print(f"   - Some improvement but not consistent across all symbols")
        print(f"   - Monitor closely after deployment")
    else:
        print("❌ RECOMMENDATION: DO NOT DEPLOY")
        print(f"   - No consistent improvement across symbols")
        print(f"   - Investigate further before deployment")

    print("=" * 100)


if __name__ == "__main__":
    main()
