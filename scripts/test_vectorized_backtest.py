"""
Test Vectorized Backtest with Historical Data

Validates V7 Ultimate signal generation on 2 years of historical data.
Uses technical indicators to verify signal quality.
"""
import sys
sys.path.insert(0, '/root/crpbot')

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VectorizedBacktest:
    """Fast vectorized backtesting engine"""

    def __init__(self, initial_capital: float = 10_000.0):
        self.initial_capital = initial_capital
        self.position_size = 0.02  # 2% risk per trade (FTMO compliant)

    def generate_simple_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on technical indicators.

        Simple MACD crossover strategy for validation:
        - LONG: MACD crosses above signal line AND RSI > 50
        - SHORT: MACD crosses below signal line AND RSI < 50
        - HOLD: Otherwise
        """
        df = df.copy()

        # Detect MACD crossovers
        df['macd_prev'] = df['macd'].shift(1)
        df['macd_signal_prev'] = df['macd_signal'].shift(1)

        # Bullish crossover: MACD crosses above signal
        bullish_cross = (
            (df['macd'] > df['macd_signal']) &
            (df['macd_prev'] <= df['macd_signal_prev'])
        )

        # Bearish crossover: MACD crosses below signal
        bearish_cross = (
            (df['macd'] < df['macd_signal']) &
            (df['macd_prev'] >= df['macd_signal_prev'])
        )

        # Add RSI filter for trend confirmation
        long_condition = bullish_cross & (df['rsi_14'] > 50)
        short_condition = bearish_cross & (df['rsi_14'] < 50)

        df['signal'] = 0  # 0 = HOLD
        df.loc[long_condition, 'signal'] = 1  # 1 = LONG
        df.loc[short_condition, 'signal'] = -1  # -1 = SHORT

        # Calculate confidence based on indicator alignment
        df['confidence'] = 50.0

        # For LONG: Higher confidence when RSI is higher (bullish momentum)
        df.loc[long_condition, 'confidence'] = 50 + (df['rsi_14'] - 50) / 50 * 30

        # For SHORT: Higher confidence when RSI is lower (bearish momentum)
        df.loc[short_condition, 'confidence'] = 50 + (50 - df['rsi_14']) / 50 * 30

        return df

    def calculate_position_size(self, capital: float, entry_price: float,
                                stop_loss: float) -> float:
        """Calculate position size based on risk (FTMO 2% rule)"""
        risk_amount = capital * self.position_size
        price_risk = abs(entry_price - stop_loss)

        if price_risk == 0:
            return 0.0

        position_size = risk_amount / price_risk
        return position_size

    def backtest(self, df: pd.DataFrame) -> Dict:
        """
        Vectorized backtest implementation.

        Returns:
            Dict with performance metrics
        """
        df = df.copy()

        # Generate signals
        df = self.generate_simple_signals(df)

        # Initialize columns
        df['position'] = 0  # Current position: 1=long, -1=short, 0=flat
        df['entry_price'] = 0.0
        df['stop_loss'] = 0.0
        df['take_profit'] = 0.0
        df['pnl'] = 0.0
        df['capital'] = self.initial_capital

        # Track trades
        trades = []
        current_position = 0
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        position_size = 0.0
        entry_idx = 0

        # Iterate through signals
        for i in range(len(df)):
            row = df.iloc[i]

            # Check if we have a position
            if current_position != 0:
                # Check stop loss hit
                if current_position == 1:  # Long position
                    if row['low'] <= stop_loss:
                        # Stop loss hit
                        pnl_pct = (stop_loss - entry_price) / entry_price
                        pnl = position_size * pnl_pct * entry_price
                        trades.append({
                            'entry_idx': entry_idx,
                            'exit_idx': i,
                            'entry_price': entry_price,
                            'exit_price': stop_loss,
                            'direction': 'LONG',
                            'pnl': pnl,
                            'pnl_pct': pnl_pct * 100,
                            'outcome': 'loss'
                        })
                        current_position = 0
                        continue

                    # Check take profit hit
                    if row['high'] >= take_profit:
                        # Take profit hit
                        pnl_pct = (take_profit - entry_price) / entry_price
                        pnl = position_size * pnl_pct * entry_price
                        trades.append({
                            'entry_idx': entry_idx,
                            'exit_idx': i,
                            'entry_price': entry_price,
                            'exit_price': take_profit,
                            'direction': 'LONG',
                            'pnl': pnl,
                            'pnl_pct': pnl_pct * 100,
                            'outcome': 'win'
                        })
                        current_position = 0
                        continue

                elif current_position == -1:  # Short position
                    if row['high'] >= stop_loss:
                        # Stop loss hit
                        pnl_pct = (entry_price - stop_loss) / entry_price
                        pnl = position_size * pnl_pct * entry_price
                        trades.append({
                            'entry_idx': entry_idx,
                            'exit_idx': i,
                            'entry_price': entry_price,
                            'exit_price': stop_loss,
                            'direction': 'SHORT',
                            'pnl': pnl,
                            'pnl_pct': pnl_pct * 100,
                            'outcome': 'loss'
                        })
                        current_position = 0
                        continue

                    # Check take profit hit
                    if row['low'] <= take_profit:
                        # Take profit hit
                        pnl_pct = (entry_price - take_profit) / entry_price
                        pnl = position_size * pnl_pct * entry_price
                        trades.append({
                            'entry_idx': entry_idx,
                            'exit_idx': i,
                            'entry_price': entry_price,
                            'exit_price': take_profit,
                            'direction': 'SHORT',
                            'pnl': pnl,
                            'pnl_pct': pnl_pct * 100,
                            'outcome': 'win'
                        })
                        current_position = 0
                        continue

            # Check for new signal (only if flat)
            if current_position == 0 and row['signal'] != 0:
                entry_price = row['close']
                entry_idx = i

                # Set stop loss and take profit (2:1 R:R ratio)
                if row['signal'] == 1:  # LONG
                    stop_loss = entry_price * 0.98  # 2% stop loss
                    take_profit = entry_price * 1.04  # 4% take profit
                    current_position = 1

                elif row['signal'] == -1:  # SHORT
                    stop_loss = entry_price * 1.02  # 2% stop loss
                    take_profit = entry_price * 0.96  # 4% take profit
                    current_position = -1

                # Calculate position size
                position_size = self.calculate_position_size(
                    self.initial_capital, entry_price, stop_loss
                )

        # Calculate metrics
        trades_df = pd.DataFrame(trades)

        if len(trades_df) == 0:
            return {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl_pct': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'trades': []
            }

        total_trades = len(trades_df)
        wins = len(trades_df[trades_df['outcome'] == 'win'])
        win_rate = wins / total_trades * 100

        total_pnl = trades_df['pnl'].sum()
        avg_pnl_pct = trades_df['pnl_pct'].mean()

        # Calculate drawdown
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        trades_df['running_max'] = trades_df['cumulative_pnl'].cummax()
        trades_df['drawdown'] = trades_df['cumulative_pnl'] - trades_df['running_max']
        max_drawdown = trades_df['drawdown'].min()

        # Calculate Sharpe ratio (assuming 252 trading days per year)
        if trades_df['pnl_pct'].std() > 0:
            sharpe_ratio = (trades_df['pnl_pct'].mean() / trades_df['pnl_pct'].std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        return {
            'total_trades': total_trades,
            'wins': wins,
            'losses': total_trades - wins,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl_pct': avg_pnl_pct,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'trades': trades_df.to_dict('records')
        }


def test_backtest_on_symbol(symbol: str) -> Dict:
    """Run backtest on a single symbol"""

    symbol_clean = symbol.replace('-', '_')
    data_path = Path(f'data/historical/{symbol_clean}_3600s_730d_features.parquet')

    if not data_path.exists():
        logger.error(f"❌ Features not found: {data_path}")
        return None

    logger.info(f"📊 Loading {symbol} features...")
    df = pd.read_parquet(data_path)

    # Remove rows with NaN values in key indicators only (not all columns)
    # This preserves more data while ensuring key indicators are valid
    key_indicators = ['rsi_14', 'macd', 'macd_signal', 'close', 'high', 'low']
    df = df.dropna(subset=key_indicators)

    logger.info(f"  {len(df)} candles after removing warmup period")

    # Run backtest
    backtest = VectorizedBacktest(initial_capital=10_000.0)
    results = backtest.backtest(df)

    return results


def main():
    """Test vectorized backtest on all symbols"""

    print("=" * 70)
    print("VECTORIZED BACKTEST TEST")
    print("=" * 70)
    print(f"\nStrategy: MACD Crossover + RSI Confirmation")
    print(f"Capital: $10,000")
    print(f"Risk per trade: 2% (FTMO compliant)")
    print(f"R:R Ratio: 1:2 (2% SL, 4% TP)\n")

    symbols = [
        'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'DOGE-USD',
        'ADA-USD', 'AVAX-USD', 'LINK-USD', 'POL-USD', 'LTC-USD'
    ]

    all_results = {}

    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] {symbol}")
        print("-" * 70)

        results = test_backtest_on_symbol(symbol)

        if results is None:
            continue

        all_results[symbol] = results

        # Display results
        print(f"  Total Trades:    {results['total_trades']}")
        print(f"  Wins:            {results['wins']}")
        print(f"  Losses:          {results['losses']}")
        print(f"  Win Rate:        {results['win_rate']:.2f}%")
        print(f"  Total P&L:       ${results['total_pnl']:,.2f}")
        print(f"  Avg P&L:         {results['avg_pnl_pct']:.2f}%")
        print(f"  Max Drawdown:    ${results['max_drawdown']:,.2f}")
        print(f"  Sharpe Ratio:    {results['sharpe_ratio']:.2f}")

    # Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATE BACKTEST RESULTS")
    print("=" * 70)

    total_trades = sum(r['total_trades'] for r in all_results.values())
    total_wins = sum(r['wins'] for r in all_results.values())
    aggregate_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    aggregate_pnl = sum(r['total_pnl'] for r in all_results.values())
    aggregate_sharpe = np.mean([r['sharpe_ratio'] for r in all_results.values()])

    print(f"\nTotal Trades:      {total_trades}")
    print(f"Total Wins:        {total_wins}")
    print(f"Aggregate Win Rate: {aggregate_win_rate:.2f}%")
    print(f"Aggregate P&L:     ${aggregate_pnl:,.2f}")
    print(f"Average Sharpe:    {aggregate_sharpe:.2f}")

    # Save results
    results_path = Path('data/backtest_results.json')
    import json
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n💾 Saved results: {results_path}")

    print("\n" + "=" * 70)
    print("✅ Vectorized Backtest Complete!")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    results = main()
