"""
Walk-Forward Optimization for V7 Ultimate

Tests strategy robustness with in-sample (IS) and out-of-sample (OOS) validation.
Calculates Walk-Forward Efficiency (WFE) metric.

WFE = OOS Performance / IS Performance
WFE > 0.6 = Good generalization (strategy is robust)
WFE < 0.6 = Overfitting (strategy is curve-fitted to IS data)
"""
import sys
_this_file = Path(__file__).resolve()
_project_root = _this_file.parent.parent
sys.path.insert(0, str(_project_root))

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WalkForwardOptimizer:
    """Walk-forward optimization with anchored windows"""

    def __init__(self, initial_capital: float = 10_000.0):
        self.initial_capital = initial_capital
        self.position_size = 0.02  # 2% risk per trade

    def generate_signals(self, df: pd.DataFrame, rsi_threshold: int = 50) -> pd.DataFrame:
        """
        Generate trading signals with configurable RSI threshold.

        Args:
            df: DataFrame with OHLCV and indicators
            rsi_threshold: RSI threshold for filtering (50 = default, optimize in IS)
        """
        df = df.copy()

        # Detect MACD crossovers
        df['macd_prev'] = df['macd'].shift(1)
        df['macd_signal_prev'] = df['macd_signal'].shift(1)

        # Bullish crossover
        bullish_cross = (
            (df['macd'] > df['macd_signal']) &
            (df['macd_prev'] <= df['macd_signal_prev'])
        )

        # Bearish crossover
        bearish_cross = (
            (df['macd'] < df['macd_signal']) &
            (df['macd_prev'] >= df['macd_signal_prev'])
        )

        # Add RSI filter (configurable threshold)
        long_condition = bullish_cross & (df['rsi_14'] > rsi_threshold)
        short_condition = bearish_cross & (df['rsi_14'] < (100 - rsi_threshold))

        df['signal'] = 0
        df.loc[long_condition, 'signal'] = 1
        df.loc[short_condition, 'signal'] = -1

        return df

    def backtest(self, df: pd.DataFrame, rsi_threshold: int = 50) -> Dict:
        """Run backtest with given parameters"""

        df = self.generate_signals(df, rsi_threshold)

        # Track trades
        trades = []
        current_position = 0
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        position_size = 0.0
        entry_idx = 0

        for i in range(len(df)):
            row = df.iloc[i]

            # Check if we have a position
            if current_position != 0:
                # Check stop loss hit
                if current_position == 1:  # Long
                    if row['low'] <= stop_loss:
                        pnl_pct = (stop_loss - entry_price) / entry_price
                        pnl = position_size * pnl_pct * entry_price
                        trades.append({
                            'exit_idx': i,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct * 100,
                            'outcome': 'loss'
                        })
                        current_position = 0
                        continue

                    if row['high'] >= take_profit:
                        pnl_pct = (take_profit - entry_price) / entry_price
                        pnl = position_size * pnl_pct * entry_price
                        trades.append({
                            'exit_idx': i,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct * 100,
                            'outcome': 'win'
                        })
                        current_position = 0
                        continue

                elif current_position == -1:  # Short
                    if row['high'] >= stop_loss:
                        pnl_pct = (entry_price - stop_loss) / entry_price
                        pnl = position_size * pnl_pct * entry_price
                        trades.append({
                            'exit_idx': i,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct * 100,
                            'outcome': 'loss'
                        })
                        current_position = 0
                        continue

                    if row['low'] <= take_profit:
                        pnl_pct = (entry_price - take_profit) / entry_price
                        pnl = position_size * pnl_pct * entry_price
                        trades.append({
                            'exit_idx': i,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct * 100,
                            'outcome': 'win'
                        })
                        current_position = 0
                        continue

            # Check for new signal
            if current_position == 0 and row['signal'] != 0:
                entry_price = row['close']
                entry_idx = i

                if row['signal'] == 1:  # LONG
                    stop_loss = entry_price * 0.98
                    take_profit = entry_price * 1.04
                    current_position = 1
                elif row['signal'] == -1:  # SHORT
                    stop_loss = entry_price * 1.02
                    take_profit = entry_price * 0.96
                    current_position = -1

                position_size = (self.initial_capital * self.position_size) / abs(entry_price - stop_loss)

        # Calculate metrics
        if not trades:
            return {
                'total_trades': 0,
                'total_pnl': 0.0,
                'sharpe_ratio': 0.0,
                'win_rate': 0.0
            }

        trades_df = pd.DataFrame(trades)
        total_pnl = trades_df['pnl'].sum()
        wins = len(trades_df[trades_df['outcome'] == 'win'])
        win_rate = (wins / len(trades_df)) * 100

        if trades_df['pnl_pct'].std() > 0:
            sharpe = (trades_df['pnl_pct'].mean() / trades_df['pnl_pct'].std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        return {
            'total_trades': len(trades_df),
            'total_pnl': total_pnl,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'trades': trades
        }

    def optimize_in_sample(self, df: pd.DataFrame) -> Tuple[int, Dict]:
        """
        Optimize RSI threshold on in-sample data.

        Returns:
            (best_rsi_threshold, best_results)
        """
        logger.info(f"  Optimizing RSI threshold on {len(df)} candles...")

        # Test RSI thresholds: 40, 45, 50, 55, 60
        rsi_thresholds = [40, 45, 50, 55, 60]

        best_sharpe = -999
        best_threshold = 50
        best_results = None

        for threshold in rsi_thresholds:
            results = self.backtest(df, rsi_threshold=threshold)

            if results['total_trades'] > 10:  # Minimum sample size
                sharpe = results['sharpe_ratio']

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_threshold = threshold
                    best_results = results

        logger.info(f"  Best RSI threshold: {best_threshold} (Sharpe: {best_sharpe:.2f})")

        return best_threshold, best_results

    def walk_forward(self, df: pd.DataFrame, n_windows: int = 4) -> Dict:
        """
        Walk-forward optimization with anchored windows.

        Args:
            df: Full dataset
            n_windows: Number of walk-forward windows (default: 4 = quarterly)

        Returns:
            WFO results with WFE metric
        """
        total_candles = len(df)
        window_size = total_candles // n_windows

        logger.info(f"  Walk-forward setup: {n_windows} windows, {window_size} candles each")

        wf_results = []

        for i in range(n_windows - 1):  # Last window is only OOS
            # In-sample window (anchored from start)
            is_start = 0
            is_end = (i + 1) * window_size

            # Out-of-sample window (next window)
            oos_start = is_end
            oos_end = min(is_end + window_size, total_candles)

            df_is = df.iloc[is_start:is_end].copy()
            df_oos = df.iloc[oos_start:oos_end].copy()

            logger.info(f"  Window {i+1}/{n_windows-1}: IS[{is_start}:{is_end}] OOS[{oos_start}:{oos_end}]")

            # Optimize on in-sample
            best_threshold, is_results = self.optimize_in_sample(df_is)

            # Test on out-of-sample with optimized parameters
            oos_results = self.backtest(df_oos, rsi_threshold=best_threshold)

            wf_results.append({
                'window': i + 1,
                'is_candles': len(df_is),
                'oos_candles': len(df_oos),
                'best_rsi_threshold': best_threshold,
                'is_trades': is_results['total_trades'],
                'is_sharpe': is_results['sharpe_ratio'],
                'is_pnl': is_results['total_pnl'],
                'oos_trades': oos_results['total_trades'],
                'oos_sharpe': oos_results['sharpe_ratio'],
                'oos_pnl': oos_results['total_pnl']
            })

        return wf_results

    def calculate_wfe(self, wf_results: List[Dict]) -> float:
        """
        Calculate Walk-Forward Efficiency (WFE).

        WFE = Total OOS P&L / Total IS P&L
        WFE > 0.6 = Good (robust strategy)
        WFE < 0.6 = Poor (overfitting)
        """
        total_is_pnl = sum(r['is_pnl'] for r in wf_results)
        total_oos_pnl = sum(r['oos_pnl'] for r in wf_results)

        if total_is_pnl > 0:
            wfe = total_oos_pnl / total_is_pnl
        else:
            wfe = 0.0

        return wfe


def run_walk_forward_on_symbol(symbol: str) -> Dict:
    """Run walk-forward optimization on a single symbol"""

    symbol_clean = symbol.replace('-', '_')
    data_path = Path(f'data/historical/{symbol_clean}_3600s_730d_features.parquet')

    if not data_path.exists():
        logger.error(f"❌ Features not found: {data_path}")
        return None

    logger.info(f"📊 Loading {symbol} features...")
    df = pd.read_parquet(data_path)

    # Remove rows with NaN values in key indicators
    key_indicators = ['rsi_14', 'macd', 'macd_signal', 'close', 'high', 'low']
    df = df.dropna(subset=key_indicators)

    logger.info(f"  {len(df)} candles after warmup")

    # Run walk-forward optimization
    optimizer = WalkForwardOptimizer()
    wf_results = optimizer.walk_forward(df, n_windows=4)

    # Calculate WFE
    wfe = optimizer.calculate_wfe(wf_results)

    return {
        'symbol': symbol,
        'wf_results': wf_results,
        'wfe': wfe
    }


def main():
    """Run walk-forward optimization on all symbols"""

    print("=" * 80)
    print("WALK-FORWARD OPTIMIZATION")
    print("=" * 80)
    print(f"\nStrategy: MACD Crossover + RSI Confirmation")
    print(f"Optimization: RSI threshold (40, 45, 50, 55, 60)")
    print(f"Windows: 4 (quarterly anchored)")
    print(f"Target WFE: > 0.6 (robust strategy)\n")

    symbols = [
        'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'DOGE-USD',
        'ADA-USD', 'AVAX-USD', 'LINK-USD', 'POL-USD', 'LTC-USD'
    ]

    all_results = {}

    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] {symbol}")
        print("-" * 80)

        result = run_walk_forward_on_symbol(symbol)

        if result is None:
            continue

        all_results[symbol] = result

        # Display WFE
        wfe = result['wfe']
        wfe_status = "✅ ROBUST" if wfe > 0.6 else "⚠️ OVERFIT"

        print(f"\n  Walk-Forward Efficiency: {wfe:.3f} {wfe_status}")

        # Display window results
        print(f"\n  Per-Window Results:")
        print(f"  {'Window':<8} {'IS Sharpe':<12} {'OOS Sharpe':<12} {'Best RSI':<10}")
        print(f"  {'-'*50}")

        for wf in result['wf_results']:
            print(f"  {wf['window']:<8} {wf['is_sharpe']:>8.2f}     {wf['oos_sharpe']:>8.2f}      {wf['best_rsi_threshold']:>5}")

    # Aggregate analysis
    print("\n" + "=" * 80)
    print("AGGREGATE WFE ANALYSIS")
    print("=" * 80)

    wfe_values = [r['wfe'] for r in all_results.values()]
    avg_wfe = np.mean(wfe_values)
    robust_count = sum(1 for wfe in wfe_values if wfe > 0.6)

    print(f"\nTotal Symbols:     {len(all_results)}")
    print(f"Average WFE:       {avg_wfe:.3f}")
    print(f"Robust Strategies: {robust_count}/{len(all_results)} ({(robust_count/len(all_results)*100):.1f}%)")

    # Top performers by WFE
    sorted_by_wfe = sorted(all_results.items(), key=lambda x: x[1]['wfe'], reverse=True)

    print(f"\nTop 3 by WFE:")
    for rank, (symbol, result) in enumerate(sorted_by_wfe[:3], 1):
        print(f"  #{rank} {symbol}: WFE = {result['wfe']:.3f}")

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if avg_wfe > 0.6:
        recommendation = "✅ Strategy is ROBUST - low overfitting risk"
    elif avg_wfe > 0.4:
        recommendation = "⚠️ Strategy is MARGINAL - monitor for overfitting"
    else:
        recommendation = "❌ Strategy is OVERFIT - requires redesign"

    print(f"\nAverage WFE: {avg_wfe:.3f}")
    print(f"Recommendation: {recommendation}")

    # Save results
    results_path = Path('data/walk_forward_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n💾 Saved results: {results_path}")

    print("\n" + "=" * 80)
    print("✅ Walk-Forward Optimization Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
