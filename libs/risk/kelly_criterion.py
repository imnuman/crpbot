"""
Kelly Criterion for Optimal Position Sizing

f* = (p * b - q) / b

where:
  p = win probability
  q = loss probability (1 - p)
  b = win/loss ratio (avg_win / avg_loss)
  f* = optimal fraction of capital to risk
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class KellyCriterion:
    """Kelly Criterion position sizing"""

    def __init__(self, fractional_kelly=0.5):
        """
        Args:
            fractional_kelly: Use fractional Kelly for safety (default 0.5 = half Kelly)
                             Full Kelly can be too aggressive
        """
        self.fractional_kelly = fractional_kelly

    def calculate_kelly(self, win_rate, avg_win_pct, avg_loss_pct):
        """
        Calculate Kelly Criterion optimal position size

        Args:
            win_rate: Probability of winning (0 to 1)
            avg_win_pct: Average win as % (e.g., 0.05 = 5%)
            avg_loss_pct: Average loss as % (e.g., -0.03 = -3%)

        Returns:
            Optimal position size as fraction of capital
        """
        if avg_loss_pct >= 0:
            return 0  # Can't use Kelly if losses are positive

        p = win_rate
        q = 1 - win_rate
        b = abs(avg_win_pct / avg_loss_pct)  # Win/loss ratio

        # Kelly formula
        kelly_fraction = (p * b - q) / b

        # Apply fractional Kelly for safety
        kelly_fraction = kelly_fraction * self.fractional_kelly

        # Cap at 25% to prevent over-leverage
        kelly_fraction = max(0, min(kelly_fraction, 0.25))

        return kelly_fraction

    def analyze_historical_trades(self, trades_df):
        """
        Analyze historical trades to recommend Kelly position size

        Args:
            trades_df: DataFrame with columns ['pnl_percent', 'outcome']

        Returns:
            Dict with Kelly analysis
        """
        # Separate wins and losses
        wins = trades_df[trades_df['outcome'] == 'win']
        losses = trades_df[trades_df['outcome'] == 'loss']

        # Calculate statistics
        total_trades = len(trades_df)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0

        avg_win_pct = wins['pnl_percent'].mean() / 100 if len(wins) > 0 else 0
        avg_loss_pct = losses['pnl_percent'].mean() / 100 if len(losses) > 0 else 0

        # Calculate Kelly
        kelly_size = self.calculate_kelly(win_rate, avg_win_pct, avg_loss_pct)

        # Calculate expected value
        expected_value = (win_rate * avg_win_pct) + ((1 - win_rate) * avg_loss_pct)

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win_pct': avg_win_pct * 100,  # Convert back to %
            'avg_loss_pct': avg_loss_pct * 100,
            'kelly_fraction': kelly_size,
            'fractional_kelly': self.fractional_kelly,
            'expected_value': expected_value * 100,
            'profit_factor': abs(avg_win_pct / avg_loss_pct) if avg_loss_pct != 0 else 0
        }

    def print_analysis(self, analysis):
        """Print Kelly analysis results"""
        print("\n" + "="*70)
        print("KELLY CRITERION POSITION SIZING ANALYSIS")
        print("="*70)
        print(f"Total Trades:        {analysis['total_trades']}")
        print(f"Win Rate:            {analysis['win_rate']*100:.2f}%")
        print(f"Average Win:         {analysis['avg_win_pct']:.2f}%")
        print(f"Average Loss:        {analysis['avg_loss_pct']:.2f}%")
        print(f"Profit Factor:       {analysis['profit_factor']:.2f}")
        print(f"Expected Value:      {analysis['expected_value']:.2f}%")
        print(f"\nFractional Kelly:    {analysis['fractional_kelly']*100:.0f}%")
        print(f"RECOMMENDED SIZE:    {analysis['kelly_fraction']*100:.1f}% of capital")
        print("="*70)

        # Interpretation
        if analysis['kelly_fraction'] > 0.15:
            print("⚠️  Large position size - system has strong edge")
        elif analysis['kelly_fraction'] > 0.05:
            print("✅ Moderate position size - good risk/reward")
        elif analysis['kelly_fraction'] > 0:
            print("⚠️  Small position size - weak edge, consider improving strategy")
        else:
            print("❌ Zero position size - negative expected value, DO NOT TRADE")


# CLI usage
if __name__ == "__main__":
    # Load paper trading results from database
    import sqlite3

    conn = sqlite3.connect('tradingai.db')

    # Get historical trades
    query = """
    SELECT pnl_percent,
           outcome
    FROM signal_results
    WHERE outcome IN ('win', 'loss')
    """

    trades_df = pd.read_sql(query, conn)
    conn.close()

    if len(trades_df) > 0:
        # Analyze with Kelly Criterion
        kelly = KellyCriterion(fractional_kelly=0.5)
        analysis = kelly.analyze_historical_trades(trades_df)
        kelly.print_analysis(analysis)
    else:
        print("No historical trades found. Run paper trading first.")
