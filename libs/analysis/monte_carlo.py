"""
Monte Carlo Simulation for Risk Assessment

Monte Carlo methods use repeated random sampling to model uncertainty
and simulate possible outcomes. In trading, we use them to:
1. Simulate thousands of future price paths
2. Calculate risk metrics (VaR, CVaR, drawdown)
3. Estimate probability of profit/loss scenarios
4. Stress-test strategies under various market conditions

Key Concepts:
- Geometric Brownian Motion (GBM): S(t) = S(0) × exp((μ - σ²/2)t + σW(t))
- Value at Risk (VaR): Maximum loss at confidence level (e.g., 95%)
- Conditional VaR (CVaR): Expected loss when VaR is exceeded
- Maximum Drawdown: Peak-to-trough decline
- Monte Carlo: Run simulation n times, analyze distribution

This provides probabilistic forecasts instead of single-point estimates.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Risk assessment metrics from Monte Carlo simulation"""
    var_95: float              # Value at Risk (95% confidence)
    var_99: float              # Value at Risk (99% confidence)
    cvar_95: float             # Conditional VaR (95%) / Expected Shortfall
    cvar_99: float             # Conditional VaR (99%)
    max_drawdown: float        # Maximum drawdown (%)
    sharpe_ratio: float        # Sharpe ratio (annualized)
    prob_profit: float         # Probability of profit
    expected_return: float     # Expected return
    return_std: float          # Return standard deviation
    num_simulations: int       # Number of Monte Carlo runs


class MonteCarloSimulator:
    """
    Monte Carlo simulator for trading risk assessment

    Uses Geometric Brownian Motion (GBM) to simulate price paths:
        dS = μS dt + σS dW

    Where:
        S = asset price
        μ = drift (expected return)
        σ = volatility (standard deviation)
        dW = Brownian motion increment
    """

    def __init__(
        self,
        initial_price: float,
        mu: float,
        sigma: float,
        dt: float = 1.0
    ):
        """
        Initialize Monte Carlo Simulator

        Args:
            initial_price: Starting price
            mu: Expected return (drift) per time step
            sigma: Volatility (standard deviation) per time step
            dt: Time step size (default: 1.0 for daily)
        """
        self.initial_price = initial_price
        self.mu = mu
        self.sigma = sigma
        self.dt = dt

        logger.debug(
            f"Monte Carlo initialized: S0={initial_price:.2f}, "
            f"μ={mu:.4f}, σ={sigma:.4f}, dt={dt}"
        )

    def simulate_gbm_path(
        self,
        n_steps: int,
        random_seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate single price path using Geometric Brownian Motion

        GBM formula: S(t+1) = S(t) × exp((μ - σ²/2)Δt + σ√Δt × Z)
        where Z ~ N(0, 1)

        Args:
            n_steps: Number of time steps to simulate
            random_seed: Random seed for reproducibility

        Returns:
            Array of simulated prices (length = n_steps + 1)
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Generate random shocks
        Z = np.random.standard_normal(n_steps)

        # Calculate price changes using GBM
        # log(S(t+1)/S(t)) = (μ - σ²/2)Δt + σ√Δt × Z
        log_returns = (self.mu - 0.5 * self.sigma**2) * self.dt + \
                      self.sigma * np.sqrt(self.dt) * Z

        # Convert log returns to prices
        price_path = np.zeros(n_steps + 1)
        price_path[0] = self.initial_price
        price_path[1:] = self.initial_price * np.exp(np.cumsum(log_returns))

        return price_path

    def simulate_multiple_paths(
        self,
        n_steps: int,
        n_simulations: int,
        random_seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate multiple price paths

        Args:
            n_steps: Number of time steps per path
            n_simulations: Number of paths to simulate
            random_seed: Random seed for reproducibility

        Returns:
            Array of shape (n_simulations, n_steps + 1)
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        paths = np.zeros((n_simulations, n_steps + 1))

        for i in range(n_simulations):
            paths[i] = self.simulate_gbm_path(n_steps, random_seed=None)

        logger.info(f"Simulated {n_simulations} paths with {n_steps} steps each")
        return paths

    def calculate_risk_metrics(
        self,
        paths: np.ndarray,
        initial_capital: float = 1.0
    ) -> RiskMetrics:
        """
        Calculate risk metrics from simulated paths

        Args:
            paths: Array of simulated price paths (n_simulations × n_steps)
            initial_capital: Initial portfolio value (default: 1.0)

        Returns:
            RiskMetrics with VaR, CVaR, drawdown, etc.
        """
        n_simulations, n_steps = paths.shape

        # Calculate returns for each path
        final_prices = paths[:, -1]
        returns = (final_prices - self.initial_price) / self.initial_price

        # Portfolio values
        portfolio_values = initial_capital * (1 + returns)

        # Sort returns for quantile calculations
        sorted_returns = np.sort(returns)

        # Value at Risk (VaR) - negative return at confidence level
        var_95_idx = int(0.05 * n_simulations)
        var_99_idx = int(0.01 * n_simulations)
        var_95 = -sorted_returns[var_95_idx] if var_95_idx < len(sorted_returns) else 0.0
        var_99 = -sorted_returns[var_99_idx] if var_99_idx < len(sorted_returns) else 0.0

        # Conditional VaR (CVaR) - expected loss when VaR is exceeded
        cvar_95 = -np.mean(sorted_returns[:var_95_idx]) if var_95_idx > 0 else 0.0
        cvar_99 = -np.mean(sorted_returns[:var_99_idx]) if var_99_idx > 0 else 0.0

        # Maximum Drawdown
        max_drawdowns = []
        for path in paths:
            cummax = np.maximum.accumulate(path)
            drawdown = (cummax - path) / cummax
            max_drawdowns.append(np.max(drawdown) * 100)
        max_drawdown = np.mean(max_drawdowns)

        # Expected return and volatility
        expected_return = float(np.mean(returns))
        return_std = float(np.std(returns))

        # Sharpe ratio (assuming risk-free rate = 0)
        # Annualized: multiply by sqrt(252) for daily data
        sharpe_ratio = expected_return / return_std if return_std > 0 else 0.0

        # Probability of profit
        prob_profit = float(np.sum(returns > 0) / n_simulations)

        return RiskMetrics(
            var_95=float(var_95),
            var_99=float(var_99),
            cvar_95=float(cvar_95),
            cvar_99=float(cvar_99),
            max_drawdown=float(max_drawdown),
            sharpe_ratio=float(sharpe_ratio),
            prob_profit=float(prob_profit),
            expected_return=float(expected_return),
            return_std=float(return_std),
            num_simulations=n_simulations
        )

    def stress_test(
        self,
        scenarios: Dict[str, Tuple[float, float]],
        n_steps: int = 252,
        n_simulations: int = 1000
    ) -> Dict[str, RiskMetrics]:
        """
        Stress test under different market scenarios

        Args:
            scenarios: Dict of {scenario_name: (mu, sigma)}
            n_steps: Number of time steps per simulation
            n_simulations: Number of simulations per scenario

        Returns:
            Dict of {scenario_name: RiskMetrics}
        """
        results = {}

        for scenario_name, (mu, sigma) in scenarios.items():
            # Temporarily override mu and sigma
            original_mu = self.mu
            original_sigma = self.sigma

            self.mu = mu
            self.sigma = sigma

            # Run simulation
            paths = self.simulate_multiple_paths(n_steps, n_simulations)
            metrics = self.calculate_risk_metrics(paths)

            results[scenario_name] = metrics

            # Restore original parameters
            self.mu = original_mu
            self.sigma = original_sigma

            logger.info(
                f"Stress test '{scenario_name}': "
                f"VaR95={metrics.var_95:.2%}, MaxDD={metrics.max_drawdown:.2f}%"
            )

        return results

    def estimate_portfolio_distribution(
        self,
        n_steps: int,
        n_simulations: int = 10000,
        initial_capital: float = 10000.0
    ) -> Dict:
        """
        Estimate future portfolio value distribution

        Args:
            n_steps: Number of time steps to simulate
            n_simulations: Number of Monte Carlo runs
            initial_capital: Starting portfolio value

        Returns:
            Dictionary with distribution statistics
        """
        paths = self.simulate_multiple_paths(n_steps, n_simulations)
        final_prices = paths[:, -1]

        # Calculate portfolio values
        returns = (final_prices - self.initial_price) / self.initial_price
        portfolio_values = initial_capital * (1 + returns)

        # Distribution statistics
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_values = np.percentile(portfolio_values, percentiles)

        distribution = {
            'mean': float(np.mean(portfolio_values)),
            'median': float(np.median(portfolio_values)),
            'std': float(np.std(portfolio_values)),
            'min': float(np.min(portfolio_values)),
            'max': float(np.max(portfolio_values)),
            'initial_capital': initial_capital,
            'percentiles': {
                f'p{p}': float(percentile_values[i])
                for i, p in enumerate(percentiles)
            },
            'prob_loss': float(np.sum(portfolio_values < initial_capital) / n_simulations),
            'prob_profit': float(np.sum(portfolio_values > initial_capital) / n_simulations)
        }

        logger.info(
            f"Portfolio distribution: "
            f"Mean=${distribution['mean']:.2f}, "
            f"Median=${distribution['median']:.2f}, "
            f"P(profit)={distribution['prob_profit']:.2%}"
        )

        return distribution


# Convenience function for V7 runtime
def simulate_risk_scenarios(
    current_price: float,
    historical_returns: np.ndarray,
    n_days: int = 30,
    n_simulations: int = 10000
) -> Dict:
    """
    Simulate risk scenarios based on historical data

    Args:
        current_price: Current asset price
        historical_returns: Historical return series
        n_days: Number of days to simulate ahead
        n_simulations: Number of Monte Carlo runs

    Returns:
        Dictionary with risk metrics and scenarios
    """
    # Estimate parameters from historical data
    mu = float(np.mean(historical_returns))
    sigma = float(np.std(historical_returns))

    # Base scenario (historical parameters)
    simulator = MonteCarloSimulator(
        initial_price=current_price,
        mu=mu,
        sigma=sigma,
        dt=1.0
    )

    # Run base simulation
    paths = simulator.simulate_multiple_paths(n_days, n_simulations)
    base_metrics = simulator.calculate_risk_metrics(paths)

    # Stress test scenarios
    scenarios = {
        'bull_market': (mu * 1.5, sigma * 0.8),      # Higher return, lower vol
        'bear_market': (mu * -1.0, sigma * 1.5),     # Negative return, higher vol
        'high_volatility': (mu, sigma * 2.0),        # Same return, double vol
        'market_crash': (mu * -2.0, sigma * 2.5),    # Large negative, high vol
    }

    stress_results = simulator.stress_test(scenarios, n_steps=n_days, n_simulations=n_simulations)

    return {
        'base_scenario': {
            'mu': mu,
            'sigma': sigma,
            'metrics': base_metrics
        },
        'stress_scenarios': stress_results,
        'current_price': current_price,
        'simulation_params': {
            'n_days': n_days,
            'n_simulations': n_simulations
        }
    }


if __name__ == "__main__":
    # Test Monte Carlo implementation
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-5s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    print("=" * 80)
    print("Monte Carlo Risk Simulator - Test Run")
    print("=" * 80)

    # Test parameters
    np.random.seed(42)
    initial_price = 100.0
    mu = 0.001  # 0.1% daily return
    sigma = 0.02  # 2% daily volatility
    n_days = 252  # 1 year
    n_simulations = 1000

    # Test 1: Single path simulation
    print("\n1. Testing Single Path Simulation")
    simulator = MonteCarloSimulator(initial_price, mu, sigma)
    single_path = simulator.simulate_gbm_path(n_days, random_seed=42)
    print(f"   Initial price:  ${single_path[0]:.2f}")
    print(f"   Final price:    ${single_path[-1]:.2f}")
    print(f"   Return:         {(single_path[-1]/single_path[0] - 1)*100:.2f}%")

    # Test 2: Multiple paths
    print(f"\n2. Testing Multiple Paths ({n_simulations} simulations)")
    paths = simulator.simulate_multiple_paths(n_days, n_simulations, random_seed=42)
    print(f"   Paths shape:    {paths.shape}")
    print(f"   Mean final:     ${np.mean(paths[:, -1]):.2f}")
    print(f"   Median final:   ${np.median(paths[:, -1]):.2f}")
    print(f"   Min final:      ${np.min(paths[:, -1]):.2f}")
    print(f"   Max final:      ${np.max(paths[:, -1]):.2f}")

    # Test 3: Risk metrics
    print("\n3. Testing Risk Metrics")
    metrics = simulator.calculate_risk_metrics(paths)
    print(f"   VaR (95%):      {metrics.var_95*100:.2f}%")
    print(f"   VaR (99%):      {metrics.var_99*100:.2f}%")
    print(f"   CVaR (95%):     {metrics.cvar_95*100:.2f}%")
    print(f"   CVaR (99%):     {metrics.cvar_99*100:.2f}%")
    print(f"   Max Drawdown:   {metrics.max_drawdown:.2f}%")
    print(f"   Sharpe Ratio:   {metrics.sharpe_ratio:.3f}")
    print(f"   Prob(Profit):   {metrics.prob_profit*100:.2f}%")
    print(f"   Expected Ret:   {metrics.expected_return*100:.2f}%")

    # Test 4: Stress testing
    print("\n4. Testing Stress Scenarios")
    scenarios = {
        'bull_market': (mu * 1.5, sigma * 0.8),
        'bear_market': (mu * -1.0, sigma * 1.5),
        'high_volatility': (mu, sigma * 2.0),
        'market_crash': (mu * -2.0, sigma * 2.5)
    }
    stress_results = simulator.stress_test(scenarios, n_steps=n_days, n_simulations=500)

    for scenario_name, metrics in stress_results.items():
        print(f"   {scenario_name:20s}: VaR95={metrics.var_95*100:6.2f}%, "
              f"MaxDD={metrics.max_drawdown:6.2f}%, P(profit)={metrics.prob_profit*100:5.1f}%")

    # Test 5: Portfolio distribution
    print("\n5. Testing Portfolio Distribution ($10,000 initial)")
    distribution = simulator.estimate_portfolio_distribution(
        n_steps=n_days,
        n_simulations=n_simulations,
        initial_capital=10000.0
    )
    print(f"   Mean:           ${distribution['mean']:.2f}")
    print(f"   Median:         ${distribution['median']:.2f}")
    print(f"   P5:             ${distribution['percentiles']['p5']:.2f}")
    print(f"   P95:            ${distribution['percentiles']['p95']:.2f}")
    print(f"   Prob(Loss):     {distribution['prob_loss']*100:.2f}%")
    print(f"   Prob(Profit):   {distribution['prob_profit']*100:.2f}%")

    # Visualization
    print("\n6. Creating Visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Sample paths
    sample_paths = paths[:20]  # Show first 20 paths
    for path in sample_paths:
        axes[0, 0].plot(path, alpha=0.3, linewidth=0.8)
    axes[0, 0].axhline(y=initial_price, color='red', linestyle='--',
                       label='Initial Price', linewidth=1.5)
    axes[0, 0].set_title(f"Sample Price Paths (20/{n_simulations})")
    axes[0, 0].set_xlabel("Days")
    axes[0, 0].set_ylabel("Price ($)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Final price distribution
    final_prices = paths[:, -1]
    axes[0, 1].hist(final_prices, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=initial_price, color='red', linestyle='--',
                       label='Initial Price', linewidth=2)
    axes[0, 1].axvline(x=np.mean(final_prices), color='green', linestyle='--',
                       label='Mean Final', linewidth=2)
    axes[0, 1].set_title("Final Price Distribution")
    axes[0, 1].set_xlabel("Price ($)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Return distribution
    returns = (final_prices - initial_price) / initial_price
    axes[1, 0].hist(returns * 100, bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', label='Break Even', linewidth=2)
    axes[1, 0].axvline(x=-metrics.var_95*100, color='purple', linestyle='--',
                       label='VaR 95%', linewidth=2)
    axes[1, 0].set_title("Return Distribution")
    axes[1, 0].set_xlabel("Return (%)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Stress test comparison
    scenario_names = list(stress_results.keys())
    var_95_values = [stress_results[s].var_95 * 100 for s in scenario_names]
    prob_profit_values = [stress_results[s].prob_profit * 100 for s in scenario_names]

    x = np.arange(len(scenario_names))
    width = 0.35

    ax4_1 = axes[1, 1]
    ax4_2 = ax4_1.twinx()

    bars1 = ax4_1.bar(x - width/2, var_95_values, width, label='VaR 95%',
                      color='red', alpha=0.7)
    bars2 = ax4_2.bar(x + width/2, prob_profit_values, width, label='Prob(Profit)',
                      color='green', alpha=0.7)

    ax4_1.set_xlabel('Scenario')
    ax4_1.set_ylabel('VaR 95% (%)', color='red')
    ax4_2.set_ylabel('Prob(Profit) (%)', color='green')
    ax4_1.set_title('Stress Test Results')
    ax4_1.set_xticks(x)
    ax4_1.set_xticklabels(scenario_names, rotation=45, ha='right')
    ax4_1.legend(loc='upper left')
    ax4_2.legend(loc='upper right')
    ax4_1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/monte_carlo_test.png', dpi=100, bbox_inches='tight')
    print(f"   Saved visualization to /tmp/monte_carlo_test.png")

    print("\n" + "=" * 80)
    print("Monte Carlo Test Complete!")
    print("=" * 80)
    print("\nKey Insights:")
    print("  - Monte Carlo simulates thousands of possible future scenarios")
    print("  - VaR/CVaR quantify tail risk (worst-case losses)")
    print("  - Stress testing evaluates performance under extreme conditions")
    print("  - Distribution analysis shows range of possible outcomes")
    print("  - Probabilistic forecasts beat single-point predictions")
    print("=" * 80)
