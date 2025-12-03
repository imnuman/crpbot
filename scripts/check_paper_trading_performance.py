"""
Check V7 Paper Trading Performance

Analyzes why the win rate is low and provides diagnostic insights.
"""
import sys
from pathlib import Path
_this_file = Path(__file__).resolve()
_project_root = _this_file.parent.parent
sys.path.insert(0, str(_project_root))

import sqlite3
import pandas as pd
from datetime import datetime, timedelta


def check_paper_trading():
    """Check V7 paper trading performance"""

    print("=" * 80)
    print("V7 PAPER TRADING PERFORMANCE ANALYSIS")
    print("=" * 80)
    print(f"\nAnalysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    conn = sqlite3.connect('tradingai.db')

    # Overall statistics
    print("=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)

    overall = pd.read_sql_query("""
        SELECT
            COUNT(*) as total_trades,
            SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN outcome = 'pending' THEN 1 ELSE 0 END) as pending,
            ROUND(AVG(CASE WHEN outcome = 'win' THEN 1.0 ELSE 0.0 END) * 100, 2) as win_rate,
            ROUND(AVG(pnl_percent), 2) as avg_pnl_pct,
            ROUND(SUM(pnl_percent), 2) as total_pnl_pct,
            ROUND(AVG(confidence), 2) as avg_confidence
        FROM signal_results
        WHERE outcome IN ('win', 'loss', 'pending')
    """, conn)

    print(f"\nTotal Trades:      {overall['total_trades'].iloc[0]}")
    print(f"Wins:              {overall['wins'].iloc[0]}")
    print(f"Losses:            {overall['losses'].iloc[0]}")
    print(f"Pending:           {overall['pending'].iloc[0]}")
    print(f"Win Rate:          {overall['win_rate'].iloc[0]:.2f}%")
    print(f"Avg P&L:           {overall['avg_pnl_pct'].iloc[0]:.2f}%")
    print(f"Total P&L:         {overall['total_pnl_pct'].iloc[0]:.2f}%")
    print(f"Avg Confidence:    {overall['avg_confidence'].iloc[0]:.2f}")

    # Per-symbol breakdown
    print("\n" + "=" * 80)
    print("PER-SYMBOL BREAKDOWN")
    print("=" * 80)

    per_symbol = pd.read_sql_query("""
        SELECT
            sr.symbol,
            COUNT(*) as total,
            SUM(CASE WHEN sr.outcome = 'win' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN sr.outcome = 'loss' THEN 1 ELSE 0 END) as losses,
            ROUND(AVG(CASE WHEN sr.outcome = 'win' THEN 1.0 ELSE 0.0 END) * 100, 2) as win_rate,
            ROUND(AVG(sr.pnl_percent), 2) as avg_pnl,
            ROUND(AVG(sr.confidence), 2) as avg_conf
        FROM signal_results sr
        WHERE sr.outcome IN ('win', 'loss')
        GROUP BY sr.symbol
        ORDER BY win_rate DESC
    """, conn)

    if len(per_symbol) > 0:
        print(f"\n{'Symbol':<12} {'Total':<8} {'Wins':<8} {'Losses':<10} {'Win Rate':<12} {'Avg P&L':<10} {'Avg Conf':<10}")
        print("-" * 80)
        for _, row in per_symbol.iterrows():
            print(f"{row['symbol']:<12} {int(row['total']):<8} {int(row['wins']):<8} {int(row['losses']):<10} {row['win_rate']:.2f}%{'':<7} {row['avg_pnl']:>6.2f}%  {row['avg_conf']:>7.2f}")
    else:
        print("\nNo completed trades yet.")

    # Per-direction analysis
    print("\n" + "=" * 80)
    print("PER-DIRECTION ANALYSIS")
    print("=" * 80)

    per_direction = pd.read_sql_query("""
        SELECT
            sr.direction,
            COUNT(*) as total,
            SUM(CASE WHEN sr.outcome = 'win' THEN 1 ELSE 0 END) as wins,
            ROUND(AVG(CASE WHEN sr.outcome = 'win' THEN 1.0 ELSE 0.0 END) * 100, 2) as win_rate,
            ROUND(AVG(sr.pnl_percent), 2) as avg_pnl,
            ROUND(AVG(sr.confidence), 2) as avg_conf
        FROM signal_results sr
        WHERE sr.outcome IN ('win', 'loss')
        GROUP BY sr.direction
        ORDER BY win_rate DESC
    """, conn)

    if len(per_direction) > 0:
        print(f"\n{'Direction':<12} {'Total':<8} {'Wins':<8} {'Win Rate':<12} {'Avg P&L':<10} {'Avg Conf':<10}")
        print("-" * 80)
        for _, row in per_direction.iterrows():
            print(f"{row['direction']:<12} {int(row['total']):<8} {int(row['wins']):<8} {row['win_rate']:.2f}%{'':<7} {row['avg_pnl']:>6.2f}%  {row['avg_conf']:>7.2f}")
    else:
        print("\nNo completed trades yet.")

    # Recent losses (last 10)
    print("\n" + "=" * 80)
    print("RECENT LOSSES (Last 10)")
    print("=" * 80)

    recent_losses = pd.read_sql_query("""
        SELECT
            sr.symbol,
            sr.direction,
            ROUND(sr.confidence, 2) as conf,
            ROUND(sr.entry_price, 2) as entry,
            ROUND(sr.stop_loss, 2) as sl,
            ROUND(sr.take_profit, 2) as tp,
            ROUND(sr.pnl_percent, 2) as pnl,
            datetime(sr.entry_time) as entry_time,
            datetime(sr.exit_time) as exit_time
        FROM signal_results sr
        WHERE sr.outcome = 'loss'
        ORDER BY sr.exit_time DESC
        LIMIT 10
    """, conn)

    if len(recent_losses) > 0:
        print(f"\n{'Symbol':<10} {'Dir':<6} {'Conf':<6} {'Entry':<10} {'SL':<10} {'TP':<10} {'P&L%':<8} {'Exit Time':<20}")
        print("-" * 80)
        for _, row in recent_losses.iterrows():
            print(f"{row['symbol']:<10} {row['direction']:<6} {row['conf']:<6.1f} {row['entry']:<10.2f} {row['sl']:<10.2f} {row['tp']:<10.2f} {row['pnl']:<8.2f} {row['exit_time']}")
    else:
        print("\nNo losses yet.")

    # A/B testing comparison
    print("\n" + "=" * 80)
    print("A/B TESTING: v7_deepseek_only vs v7_full_math")
    print("=" * 80)

    ab_test = pd.read_sql_query("""
        SELECT
            s.signal_variant,
            COUNT(*) as total,
            SUM(CASE WHEN sr.outcome = 'win' THEN 1 ELSE 0 END) as wins,
            ROUND(AVG(CASE WHEN sr.outcome = 'win' THEN 1.0 ELSE 0.0 END) * 100, 2) as win_rate,
            ROUND(AVG(sr.pnl_percent), 2) as avg_pnl,
            ROUND(AVG(s.confidence), 2) as avg_conf
        FROM signal_results sr
        JOIN signals s ON s.id = sr.signal_id
        WHERE sr.outcome IN ('win', 'loss')
        GROUP BY s.signal_variant
        ORDER BY win_rate DESC
    """, conn)

    if len(ab_test) > 0:
        print(f"\n{'Variant':<25} {'Total':<8} {'Wins':<8} {'Win Rate':<12} {'Avg P&L':<10} {'Avg Conf':<10}")
        print("-" * 80)
        for _, row in ab_test.iterrows():
            print(f"{row['signal_variant']:<25} {int(row['total']):<8} {int(row['wins']):<8} {row['win_rate']:.2f}%{'':<7} {row['avg_pnl']:>6.2f}%  {row['avg_conf']:>7.2f}")
    else:
        print("\nNo A/B test data available yet.")

    # Signal generation rate
    print("\n" + "=" * 80)
    print("SIGNAL GENERATION RATE (Last 7 Days)")
    print("=" * 80)

    signal_rate = pd.read_sql_query("""
        SELECT
            DATE(timestamp) as date,
            COUNT(*) as signals,
            ROUND(AVG(confidence), 2) as avg_conf
        FROM signals
        WHERE timestamp >= datetime('now', '-7 days')
        GROUP BY DATE(timestamp)
        ORDER BY date DESC
    """, conn)

    if len(signal_rate) > 0:
        print(f"\n{'Date':<12} {'Signals':<10} {'Avg Confidence':<15}")
        print("-" * 40)
        for _, row in signal_rate.iterrows():
            print(f"{row['date']:<12} {int(row['signals']):<10} {row['avg_conf']:<15.2f}")

        total_signals = signal_rate['signals'].sum()
        days = len(signal_rate)
        avg_per_day = total_signals / days if days > 0 else 0
        print(f"\nTotal signals (7d): {total_signals}")
        print(f"Average per day: {avg_per_day:.1f}")
    else:
        print("\nNo signals in the last 7 days.")

    conn.close()

    # Diagnostic insights
    print("\n" + "=" * 80)
    print("DIAGNOSTIC INSIGHTS")
    print("=" * 80)

    total_trades = overall['total_trades'].iloc[0]
    win_rate = overall['win_rate'].iloc[0]
    total_pnl = overall['total_pnl_pct'].iloc[0]

    print("\n🔍 Why is the win rate low?")
    print("-" * 40)

    if total_trades < 20:
        print(f"⚠️  SMALL SAMPLE SIZE: Only {total_trades} completed trades")
        print(f"   Statistical significance requires 20+ trades")
        print(f"   Current data may not represent true performance")

    if win_rate < 50:
        print(f"\n⚠️  WIN RATE BELOW 50%: {win_rate:.2f}%")
        print(f"   This is expected for 1:2 R:R ratio strategies")
        print(f"   Breakeven win rate for 1:2 R:R = 33.3%")
        print(f"   Current win rate {win_rate:.2f}% is {'ACCEPTABLE' if win_rate > 33 else 'CONCERNING'}")

    if total_pnl > 0:
        print(f"\n✅ POSITIVE P&L: {total_pnl:.2f}%")
        print(f"   Despite low win rate, total P&L is positive")
        print(f"   This indicates good risk management (wins > losses)")
    else:
        print(f"\n❌ NEGATIVE P&L: {total_pnl:.2f}%")
        print(f"   Low win rate + negative P&L is concerning")
        print(f"   May indicate poor entry timing or market conditions")

    print("\n📊 Recommendations:")
    print("-" * 40)
    print("1. Wait for 20+ completed trades before drawing conclusions")
    print("2. Review date: 2025-11-25 (Monday)")
    print("3. If Sharpe < 1.0, implement Phase 1 enhancements")
    print("4. Monitor A/B test results (deepseek_only vs full_math)")
    print("5. Check if losses are hitting SL too quickly")

    print("\n" + "=" * 80)
    print("✅ Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    check_paper_trading()
