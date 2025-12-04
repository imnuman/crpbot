"""
Analyze Backtest Results and Generate Performance Report

Comprehensive analysis of backtest performance with quantitative metrics.
"""
import sys
_this_file = Path(__file__).resolve()
_project_root = _this_file.parent.parent
sys.path.insert(0, str(_project_root))

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List


def calculate_advanced_metrics(results: Dict) -> Dict:
    """Calculate advanced quantitative metrics"""

    trades = results['trades']
    if not trades:
        return {}

    trades_df = pd.DataFrame(trades)

    # Basic metrics
    total_trades = len(trades_df)
    wins = len(trades_df[trades_df['outcome'] == 'win'])
    losses = len(trades_df[trades_df['outcome'] == 'loss'])
    win_rate = wins / total_trades * 100

    # P&L metrics
    total_pnl = trades_df['pnl'].sum()
    avg_win = trades_df[trades_df['outcome'] == 'win']['pnl'].mean() if wins > 0 else 0
    avg_loss = trades_df[trades_df['outcome'] == 'loss']['pnl'].mean() if losses > 0 else 0
    profit_factor = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if losses > 0 else 0

    # Drawdown analysis
    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
    trades_df['running_max'] = trades_df['cumulative_pnl'].cummax()
    trades_df['drawdown'] = trades_df['cumulative_pnl'] - trades_df['running_max']
    max_drawdown = trades_df['drawdown'].min()
    max_drawdown_pct = (max_drawdown / 10000) * 100  # Assuming $10k capital

    # Calmar Ratio
    annual_return = (total_pnl / 10000) * 100 / 2  # 2-year backtest
    calmar_ratio = annual_return / abs(max_drawdown_pct) if max_drawdown_pct != 0 else 0

    # Sharpe Ratio
    if trades_df['pnl_pct'].std() > 0:
        sharpe_ratio = (trades_df['pnl_pct'].mean() / trades_df['pnl_pct'].std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    # Sortino Ratio (uses only downside deviation)
    downside_returns = trades_df[trades_df['pnl_pct'] < 0]['pnl_pct']
    if len(downside_returns) > 0 and downside_returns.std() > 0:
        sortino_ratio = (trades_df['pnl_pct'].mean() / downside_returns.std()) * np.sqrt(252)
    else:
        sortino_ratio = 0.0

    # Omega Ratio (simple approximation)
    threshold = 0  # 0% threshold
    gains = trades_df[trades_df['pnl_pct'] > threshold]['pnl_pct'].sum()
    losses_sum = abs(trades_df[trades_df['pnl_pct'] < threshold]['pnl_pct'].sum())
    omega_ratio = gains / losses_sum if losses_sum > 0 else 0

    # Consecutive wins/losses
    trades_df['outcome_numeric'] = (trades_df['outcome'] == 'win').astype(int)
    trades_df['streak'] = trades_df['outcome_numeric'].groupby(
        (trades_df['outcome_numeric'] != trades_df['outcome_numeric'].shift()).cumsum()
    ).cumsum()

    max_consecutive_wins = trades_df[trades_df['outcome'] == 'win']['streak'].max() if wins > 0 else 0
    max_consecutive_losses = trades_df[trades_df['outcome'] == 'loss']['streak'].max() if losses > 0 else 0

    # Expectancy
    expectancy = (win_rate / 100 * avg_win) + ((100 - win_rate) / 100 * avg_loss)

    return {
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,
        'annual_return': annual_return,
        'calmar_ratio': calmar_ratio,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'omega_ratio': omega_ratio,
        'max_consecutive_wins': int(max_consecutive_wins),
        'max_consecutive_losses': int(max_consecutive_losses),
        'expectancy': expectancy
    }


def generate_performance_report():
    """Generate comprehensive performance report"""

    print("=" * 80)
    print("BACKTEST PERFORMANCE ANALYSIS REPORT")
    print("=" * 80)
    print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Backtest Period: 2023-11-25 to 2025-11-25 (2 years)")
    print(f"Strategy: MACD Crossover + RSI Confirmation")
    print(f"Initial Capital: $10,000")
    print(f"Position Sizing: 2% risk per trade (FTMO compliant)")
    print(f"R:R Ratio: 1:2 (2% SL, 4% TP)")

    # Load results
    results_path = Path('data/backtest_results.json')
    if not results_path.exists():
        print(f"\n❌ Results file not found: {results_path}")
        return

    with open(results_path, 'r') as f:
        all_results = json.load(f)

    # Calculate advanced metrics for each symbol
    symbols_metrics = {}
    for symbol, results in all_results.items():
        if results['total_trades'] > 0:
            symbols_metrics[symbol] = calculate_advanced_metrics(results)

    # Sort by Sharpe Ratio
    sorted_symbols = sorted(
        symbols_metrics.items(),
        key=lambda x: x[1]['sharpe_ratio'],
        reverse=True
    )

    # Per-Symbol Analysis
    print("\n" + "=" * 80)
    print("PER-SYMBOL PERFORMANCE")
    print("=" * 80)

    for symbol, metrics in sorted_symbols:
        print(f"\n{symbol}")
        print("-" * 80)
        print(f"  Trades:              {metrics['total_trades']}")
        print(f"  Win Rate:            {metrics['win_rate']:.2f}%")
        print(f"  Total P&L:           ${metrics['total_pnl']:,.2f}")
        print(f"  Avg Win:             ${metrics['avg_win']:,.2f}")
        print(f"  Avg Loss:            ${metrics['avg_loss']:,.2f}")
        print(f"  Profit Factor:       {metrics['profit_factor']:.2f}")
        print(f"  Max Drawdown:        ${metrics['max_drawdown']:,.2f} ({metrics['max_drawdown_pct']:.2f}%)")
        print(f"  Annual Return:       {metrics['annual_return']:.2f}%")
        print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio:       {metrics['sortino_ratio']:.2f}")
        print(f"  Calmar Ratio:        {metrics['calmar_ratio']:.2f}")
        print(f"  Omega Ratio:         {metrics['omega_ratio']:.2f}")
        print(f"  Expectancy:          ${metrics['expectancy']:,.2f}")
        print(f"  Max Consec. Wins:    {metrics['max_consecutive_wins']}")
        print(f"  Max Consec. Losses:  {metrics['max_consecutive_losses']}")

    # Aggregate Analysis
    print("\n" + "=" * 80)
    print("AGGREGATE ANALYSIS")
    print("=" * 80)

    total_trades = sum(m['total_trades'] for m in symbols_metrics.values())
    total_wins = sum(m['wins'] for m in symbols_metrics.values())
    total_losses = sum(m['losses'] for m in symbols_metrics.values())
    aggregate_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    aggregate_pnl = sum(m['total_pnl'] for m in symbols_metrics.values())
    avg_sharpe = np.mean([m['sharpe_ratio'] for m in symbols_metrics.values()])
    avg_sortino = np.mean([m['sortino_ratio'] for m in symbols_metrics.values()])
    avg_calmar = np.mean([m['calmar_ratio'] for m in symbols_metrics.values()])
    avg_omega = np.mean([m['omega_ratio'] for m in symbols_metrics.values()])

    print(f"\nTotal Trades:          {total_trades:,}")
    print(f"Total Wins:            {total_wins:,}")
    print(f"Total Losses:          {total_losses:,}")
    print(f"Aggregate Win Rate:    {aggregate_win_rate:.2f}%")
    print(f"Aggregate P&L:         ${aggregate_pnl:,.2f}")
    print(f"Total Return:          {(aggregate_pnl / 10000) * 100:.2f}%")
    print(f"Annualized Return:     {((aggregate_pnl / 10000) * 100) / 2:.2f}%")
    print(f"Average Sharpe:        {avg_sharpe:.2f}")
    print(f"Average Sortino:       {avg_sortino:.2f}")
    print(f"Average Calmar:        {avg_calmar:.2f}")
    print(f"Average Omega:         {avg_omega:.2f}")

    # Top Performers
    print("\n" + "=" * 80)
    print("TOP 3 PERFORMERS (by Sharpe Ratio)")
    print("=" * 80)

    for i, (symbol, metrics) in enumerate(sorted_symbols[:3], 1):
        print(f"\n#{i} {symbol}")
        print(f"  Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
        print(f"  Win Rate:        {metrics['win_rate']:.2f}%")
        print(f"  Total P&L:       ${metrics['total_pnl']:,.2f}")
        print(f"  Annual Return:   {metrics['annual_return']:.2f}%")

    # Risk Assessment
    print("\n" + "=" * 80)
    print("RISK ASSESSMENT")
    print("=" * 80)

    worst_drawdown = min(m['max_drawdown'] for m in symbols_metrics.values())
    worst_drawdown_symbol = [s for s, m in symbols_metrics.items() if m['max_drawdown'] == worst_drawdown][0]
    avg_drawdown = np.mean([m['max_drawdown'] for m in symbols_metrics.values()])

    print(f"\nWorst Drawdown:        ${worst_drawdown:,.2f} ({worst_drawdown_symbol})")
    print(f"Average Drawdown:      ${avg_drawdown:,.2f}")
    print(f"FTMO Daily Limit:      $450 (4.5% on $10k)")
    print(f"FTMO Total Limit:      $900 (9% on $10k)")

    if abs(worst_drawdown) > 450:
        print(f"⚠️  WARNING: Worst drawdown exceeds FTMO daily limit!")
    else:
        print(f"✅ All drawdowns within FTMO limits")

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if avg_sharpe > 1.5:
        recommendation = "✅ EXCELLENT - Deploy to production"
    elif avg_sharpe > 1.0:
        recommendation = "✅ GOOD - Deploy with monitoring"
    elif avg_sharpe > 0.5:
        recommendation = "⚠️  MARGINAL - Consider optimization"
    else:
        recommendation = "❌ POOR - Requires significant improvement"

    print(f"\nAverage Sharpe Ratio: {avg_sharpe:.2f}")
    print(f"Recommendation: {recommendation}")

    # Save summary
    summary = {
        'generated': datetime.now().isoformat(),
        'aggregate': {
            'total_trades': total_trades,
            'win_rate': aggregate_win_rate,
            'total_pnl': aggregate_pnl,
            'total_return_pct': (aggregate_pnl / 10000) * 100,
            'annualized_return_pct': ((aggregate_pnl / 10000) * 100) / 2,
            'avg_sharpe': avg_sharpe,
            'avg_sortino': avg_sortino,
            'avg_calmar': avg_calmar,
            'avg_omega': avg_omega
        },
        'per_symbol': symbols_metrics
    }

    summary_path = Path('data/backtest_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n💾 Saved summary: {summary_path}")

    print("\n" + "=" * 80)
    print("✅ Performance Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    generate_performance_report()
