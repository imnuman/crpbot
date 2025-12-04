"""
Correlation Analysis for Position Diversification

Prevents taking multiple highly correlated positions
Calculates rolling correlation matrix for all trading symbols
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """
    Analyze correlation between crypto assets
    Prevent concentration risk from correlated positions
    """

    def __init__(
        self,
        correlation_threshold=0.7,  # Avoid positions if correlation > 0.7
        lookback_hours=168          # 7 days for correlation calculation
    ):
        """
        Initialize correlation analyzer

        Args:
            correlation_threshold: Max allowed correlation between open positions
            lookback_hours: Hours of data to use for correlation calculation
        """
        self.correlation_threshold = correlation_threshold
        self.lookback_hours = lookback_hours
        self._correlation_matrix = None
        self._last_update = None

    def calculate_correlation_matrix(self, price_data: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Calculate correlation matrix from price data

        Args:
            price_data: Dict of {symbol: price_series}

        Returns:
            Correlation matrix DataFrame
        """
        # Convert to DataFrame
        prices_df = pd.DataFrame(price_data)

        # Calculate returns
        returns_df = prices_df.pct_change().dropna()

        # Calculate correlation
        corr_matrix = returns_df.corr()

        self._correlation_matrix = corr_matrix
        self._last_update = datetime.now()

        return corr_matrix

    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """
        Get correlation between two symbols

        Returns:
            Correlation coefficient (-1 to 1)
        """
        if self._correlation_matrix is None:
            logger.warning("Correlation matrix not calculated yet")
            return 0.0

        try:
            return self._correlation_matrix.loc[symbol1, symbol2]
        except KeyError:
            logger.warning(f"Symbol {symbol1} or {symbol2} not in correlation matrix")
            return 0.0

    def check_position_correlation(
        self,
        new_symbol: str,
        open_positions: List[str]
    ) -> Tuple[bool, List[Tuple[str, float]]]:
        """
        Check if new position is highly correlated with existing positions

        Args:
            new_symbol: Symbol to check
            open_positions: List of symbols with open positions

        Returns:
            Tuple of (is_allowed, list of conflicts)
            is_allowed: False if new position would be too correlated
            conflicts: List of (symbol, correlation) for highly correlated positions
        """
        if not open_positions:
            return True, []

        conflicts = []

        for existing_symbol in open_positions:
            if existing_symbol == new_symbol:
                continue

            corr = self.get_correlation(new_symbol, existing_symbol)

            if abs(corr) >= self.correlation_threshold:
                conflicts.append((existing_symbol, corr))

        is_allowed = len(conflicts) == 0

        return is_allowed, conflicts

    def get_diversification_score(self, positions: List[str]) -> float:
        """
        Calculate diversification score for a portfolio

        Args:
            positions: List of symbols in portfolio

        Returns:
            Score from 0 (fully correlated) to 1 (uncorrelated)
        """
        if len(positions) <= 1:
            return 1.0  # Single position is maximally diversified

        # Calculate average absolute correlation
        total_corr = 0.0
        count = 0

        for i, sym1 in enumerate(positions):
            for sym2 in positions[i+1:]:
                corr = abs(self.get_correlation(sym1, sym2))
                total_corr += corr
                count += 1

        if count == 0:
            return 1.0

        avg_corr = total_corr / count

        # Convert to diversification score (inverse of correlation)
        div_score = 1.0 - avg_corr

        return div_score

    def get_cluster_assignments(self) -> Dict[str, int]:
        """
        Group symbols into correlation clusters

        Returns:
            Dict of {symbol: cluster_id}
        """
        if self._correlation_matrix is None:
            return {}

        # Simple clustering: BTC/ETH/SOL likely one cluster, others separate
        # For now, use simple thresholding
        symbols = list(self._correlation_matrix.index)
        clusters = {}
        cluster_id = 0

        assigned = set()

        for symbol in symbols:
            if symbol in assigned:
                continue

            # Start new cluster
            cluster = [symbol]
            assigned.add(symbol)

            # Find highly correlated symbols
            for other_symbol in symbols:
                if other_symbol in assigned:
                    continue

                corr = abs(self.get_correlation(symbol, other_symbol))
                if corr >= self.correlation_threshold:
                    cluster.append(other_symbol)
                    assigned.add(other_symbol)

            # Assign cluster IDs
            for sym in cluster:
                clusters[sym] = cluster_id

            cluster_id += 1

        return clusters

    def print_correlation_matrix(self):
        """Print correlation matrix in readable format"""
        if self._correlation_matrix is None:
            print("Correlation matrix not calculated")
            return

        print("\n" + "="*70)
        print("CORRELATION MATRIX")
        print("="*70)
        print(self._correlation_matrix.round(2).to_string())
        print("\nInterpretation:")
        print("  |r| > 0.8 = Very strong correlation")
        print("  |r| > 0.6 = Strong correlation")
        print("  |r| > 0.4 = Moderate correlation")
        print("  |r| < 0.4 = Weak correlation")
        print("="*70)

    def print_analysis(self, positions: List[str]):
        """Print correlation analysis for current positions"""
        print("\n" + "="*70)
        print("CORRELATION ANALYSIS")
        print("="*70)

        if not positions:
            print("No open positions")
            return

        print(f"\nOpen Positions: {', '.join(positions)}")
        print(f"Position Count: {len(positions)}")

        # Diversification score
        div_score = self.get_diversification_score(positions)
        print(f"Diversification Score: {div_score:.2f}")

        if div_score > 0.7:
            print("  ✅ Well diversified")
        elif div_score > 0.5:
            print("  ⚠️  Moderate diversification")
        else:
            print("  ❌ Poor diversification - positions highly correlated")

        # Pairwise correlations
        if len(positions) > 1:
            print("\nPairwise Correlations:")
            for i, sym1 in enumerate(positions):
                for sym2 in positions[i+1:]:
                    corr = self.get_correlation(sym1, sym2)
                    print(f"  {sym1} <-> {sym2}: {corr:+.2f}")

        # Clusters
        clusters = self.get_cluster_assignments()
        if clusters:
            print("\nCorrelation Clusters:")
            cluster_map = {}
            for sym, cid in clusters.items():
                if cid not in cluster_map:
                    cluster_map[cid] = []
                cluster_map[cid].append(sym)

            for cid, syms in cluster_map.items():
                print(f"  Cluster {cid}: {', '.join(syms)}")

        print("="*70)


# Example usage
if __name__ == "__main__":
    # Initialize
    analyzer = CorrelationAnalyzer(
        correlation_threshold=0.7,
        lookback_hours=168  # 7 days
    )

    print("Creating simulated price data for demonstration...")

    # Simulate correlated price movements
    np.random.seed(42)
    n_periods = 168

    # BTC base price
    btc_returns = np.random.randn(n_periods) * 0.02
    btc_prices = 40000 * (1 + btc_returns).cumprod()

    # ETH highly correlated with BTC (0.8 correlation)
    eth_returns = 0.8 * btc_returns + 0.2 * np.random.randn(n_periods) * 0.02
    eth_prices = 2500 * (1 + eth_returns).cumprod()

    # SOL moderately correlated (0.6)
    sol_returns = 0.6 * btc_returns + 0.4 * np.random.randn(n_periods) * 0.03
    sol_prices = 100 * (1 + sol_returns).cumprod()

    # XRP weakly correlated (0.3)
    xrp_returns = 0.3 * btc_returns + 0.7 * np.random.randn(n_periods) * 0.025
    xrp_prices = 0.5 * (1 + xrp_returns).cumprod()

    # DOGE independent (0.1 correlation)
    doge_returns = 0.1 * btc_returns + 0.9 * np.random.randn(n_periods) * 0.04
    doge_prices = 0.08 * (1 + doge_returns).cumprod()

    # Create price series
    timestamps = pd.date_range(end=datetime.now(), periods=n_periods, freq='H')

    price_data = {
        'BTC-USD': pd.Series(btc_prices, index=timestamps),
        'ETH-USD': pd.Series(eth_prices, index=timestamps),
        'SOL-USD': pd.Series(sol_prices, index=timestamps),
        'XRP-USD': pd.Series(xrp_prices, index=timestamps),
        'DOGE-USD': pd.Series(doge_prices, index=timestamps)
    }

    print(f"  ✅ Simulated {n_periods} periods for 5 symbols")

    if len(price_data) >= 2:
        # Calculate correlation matrix
        print("\nCalculating correlation matrix...")
        corr_matrix = analyzer.calculate_correlation_matrix(price_data)

        # Print correlation matrix
        analyzer.print_correlation_matrix()

        # Test position check
        print("\n" + "="*70)
        print("POSITION CORRELATION CHECK")
        print("="*70)

        open_positions = ['BTC-USD', 'ETH-USD']
        new_symbol = 'SOL-USD'

        print(f"\nOpen Positions: {', '.join(open_positions)}")
        print(f"New Signal: {new_symbol}")

        is_allowed, conflicts = analyzer.check_position_correlation(new_symbol, open_positions)

        if is_allowed:
            print(f"\n✅ ALLOWED: {new_symbol} is not highly correlated with open positions")
        else:
            print(f"\n❌ BLOCKED: {new_symbol} is highly correlated with:")
            for sym, corr in conflicts:
                print(f"   - {sym}: {corr:+.2f}")

        # Analyze current positions
        analyzer.print_analysis(open_positions + [new_symbol])
    else:
        print("\n❌ Not enough data to calculate correlations")
