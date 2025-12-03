"""
Test Impact of Widened Stop Losses

Compares backtest results with:
- OLD: 2% SL, 4% TP (1:2 R:R)
- NEW: 4% SL, 8% TP (1:2 R:R)

This validates if wider stops improve win rate.
"""
import sys
_this_file = Path(__file__).resolve()
_project_root = _this_file.parent.parent
sys.path.insert(0, str(_project_root))

import pandas as pd
import numpy as np
from pathlib import Path

def backtest_with_stops(df: pd.DataFrame, stop_loss_pct: float, take_profit_pct: float) -> dict:
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
                    hold_minutes = (i - entry_idx) * 60  # Assuming hourly candles
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
        'total_trades': len(trades_df),
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_pnl_pct': trades_df['pnl_pct'].mean(),
        'avg_loss_hold_min': avg_loss_hold,
        'quick_loss_rate': quick_loss_rate
    }


def main():
    print("=" * 80)
    print("STOP LOSS WIDENING TEST")
    print("=" * 80)
    print("\nTesting impact of widened stop losses on backtest performance\n")

    # Load BTC data
    data_path = Path('data/historical/BTC_USD_3600s_730d_features.parquet')
    if not data_path.exists():
        print(f"❌ Data not found: {data_path}")
        return

    df = pd.read_parquet(data_path)
    df = df.dropna(subset=['rsi_14', 'macd', 'macd_signal', 'close', 'high', 'low'])

    print(f"Loaded {len(df)} BTC candles (2 years)\n")

    # Test configurations
    configs = [
        {
            'name': 'OLD (2% SL, 4% TP)',
            'sl': 0.02,
            'tp': 0.04
        },
        {
            'name': 'NEW (4% SL, 8% TP)',
            'sl': 0.04,
            'tp': 0.08
        }
    ]

    results = {}

    for config in configs:
        print(f"Testing: {config['name']}")
        print("-" * 80)

        result = backtest_with_stops(df, config['sl'], config['tp'])
        results[config['name']] = result

        print(f"  Total Trades:        {result['total_trades']}")
        print(f"  Wins:                {result.get('wins', 0)}")
        print(f"  Losses:              {result.get('losses', 0)}")
        print(f"  Win Rate:            {result['win_rate']:.1f}%")
        print(f"  Total P&L:           {result['total_pnl']:.2f}%")
        print(f"  Avg P&L per Trade:   {result['avg_pnl_pct']:.2f}%")
        print(f"  Avg Loss Hold:       {result['avg_loss_hold_min']:.0f} minutes")
        print(f"  Quick Loss Rate:     {result['quick_loss_rate']:.1f}%")
        print()

    # Comparison
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)

    old = results['OLD (2% SL, 4% TP)']
    new = results['NEW (4% SL, 8% TP)']

    print(f"\nMetric                    OLD           NEW           Change")
    print("-" * 80)
    print(f"Win Rate:                 {old['win_rate']:5.1f}%        {new['win_rate']:5.1f}%        {new['win_rate'] - old['win_rate']:+5.1f}%")
    print(f"Total P&L:                {old['total_pnl']:6.1f}%       {new['total_pnl']:6.1f}%       {new['total_pnl'] - old['total_pnl']:+6.1f}%")
    print(f"Avg Loss Hold (min):      {old['avg_loss_hold_min']:5.0f}         {new['avg_loss_hold_min']:5.0f}         {new['avg_loss_hold_min'] - old['avg_loss_hold_min']:+5.0f}")
    print(f"Quick Loss Rate:          {old['quick_loss_rate']:5.1f}%        {new['quick_loss_rate']:5.1f}%        {new['quick_loss_rate'] - old['quick_loss_rate']:+5.1f}%")

    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    if new['win_rate'] > old['win_rate']:
        print(f"✅ Win rate IMPROVED by {new['win_rate'] - old['win_rate']:.1f}%")
    else:
        print(f"❌ Win rate DEGRADED by {old['win_rate'] - new['win_rate']:.1f}%")

    if new['total_pnl'] > old['total_pnl']:
        print(f"✅ Total P&L IMPROVED by {new['total_pnl'] - old['total_pnl']:.1f}%")
    else:
        print(f"❌ Total P&L DEGRADED by {old['total_pnl'] - new['total_pnl']:.1f}%")

    if new['quick_loss_rate'] < old['quick_loss_rate']:
        print(f"✅ Quick losses REDUCED by {old['quick_loss_rate'] - new['quick_loss_rate']:.1f}%")
    else:
        print(f"⚠️  Quick losses INCREASED by {new['quick_loss_rate'] - old['quick_loss_rate']:.1f}%")

    print("\n" + "=" * 80)

    # Recommendation
    if new['win_rate'] > old['win_rate'] and new['total_pnl'] > old['total_pnl']:
        print("✅ RECOMMENDATION: Deploy widened stop losses")
    elif new['win_rate'] > old['win_rate'] or new['total_pnl'] > old['total_pnl']:
        print("⚠️  RECOMMENDATION: Widened stops show mixed results - consider further testing")
    else:
        print("❌ RECOMMENDATION: Widened stops did NOT improve performance - investigate further")

    print("=" * 80)


if __name__ == "__main__":
    main()
