"""
Bayesian Inference for Online Learning from Trading Outcomes

Bayesian Inference updates beliefs (probabilities) as new evidence arrives.
In trading, we use it to:
1. Estimate strategy win rate with uncertainty
2. Update beliefs after each trade outcome
3. Calculate confidence intervals (credible intervals)
4. Make probabilistic predictions

Key Concepts:
- Prior: Initial belief before seeing data
- Likelihood: Probability of data given parameters
- Posterior: Updated belief after seeing data
- Bayes' Theorem: P(θ|D) ∝ P(D|θ) × P(θ)

We use Beta-Binomial conjugate prior for win rate estimation:
- Prior: Beta(α, β)
- Likelihood: Binomial(n, θ)
- Posterior: Beta(α + wins, β + losses)

This allows closed-form online updates without numerical integration.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from scipy import stats
from scipy.special import betaln

logger = logging.getLogger(__name__)


@dataclass
class BayesianEstimate:
    """Bayesian parameter estimate with uncertainty"""
    mean: float                    # Expected value (posterior mean)
    mode: float                    # Most likely value (MAP estimate)
    median: float                  # 50th percentile
    std: float                     # Standard deviation (uncertainty)
    credible_interval_95: Tuple[float, float]  # 95% credible interval
    credible_interval_68: Tuple[float, float]  # 68% credible interval (±1σ)
    num_observations: int          # Number of data points used
    timestamp: Optional[pd.Timestamp] = None


class BayesianWinRateLearner:
    """
    Bayesian learner for trading strategy win rate estimation

    Uses Beta-Binomial conjugate prior for efficient online updates:
    - Prior: Beta(α, β) represents belief about win rate
    - Update: Observe wins/losses, compute posterior Beta(α+w, β+l)
    - Predict: Use posterior to estimate future performance
    """

    def __init__(
        self,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        min_credible_samples: int = 10
    ):
        """
        Initialize Bayesian Win Rate Learner

        Args:
            alpha_prior: Prior successes (α parameter of Beta)
                        α=1, β=1 = Uniform prior (no initial belief)
                        α=β>1 = Symmetric prior centered at 0.5
                        α>β = Optimistic prior (expect wins)
            beta_prior: Prior failures (β parameter of Beta)
            min_credible_samples: Minimum samples for credible estimates
        """
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.min_credible_samples = min_credible_samples

        # Current posterior parameters
        self.alpha_posterior = alpha_prior
        self.beta_posterior = beta_prior

        # Track history
        self.wins = 0
        self.losses = 0
        self.total_trades = 0
        self.history: List[Tuple[bool, BayesianEstimate]] = []

        logger.debug(
            f"Bayesian Win Rate Learner initialized: "
            f"α={alpha_prior}, β={beta_prior}"
        )

    def update(self, win: bool) -> BayesianEstimate:
        """
        Update beliefs with new trade outcome

        Args:
            win: True if trade was profitable, False otherwise

        Returns:
            Updated BayesianEstimate with posterior statistics
        """
        # Update counts
        if win:
            self.wins += 1
            self.alpha_posterior += 1
        else:
            self.losses += 1
            self.beta_posterior += 1

        self.total_trades += 1

        # Calculate posterior statistics
        estimate = self._calculate_estimate()

        # Store in history
        self.history.append((win, estimate))

        logger.debug(
            f"Updated: win={win}, total={self.total_trades}, "
            f"win_rate={estimate.mean:.3f} ± {estimate.std:.3f}"
        )

        return estimate

    def update_batch(self, outcomes: List[bool]) -> BayesianEstimate:
        """
        Update with batch of trade outcomes

        Args:
            outcomes: List of trade outcomes (True=win, False=loss)

        Returns:
            Final BayesianEstimate after all updates
        """
        wins = sum(outcomes)
        losses = len(outcomes) - wins

        self.wins += wins
        self.losses += losses
        self.total_trades += len(outcomes)

        self.alpha_posterior += wins
        self.beta_posterior += losses

        estimate = self._calculate_estimate()

        logger.info(
            f"Batch update: {len(outcomes)} trades, "
            f"{wins} wins, {losses} losses"
        )

        return estimate

    def _calculate_estimate(self) -> BayesianEstimate:
        """
        Calculate posterior statistics

        Returns:
            BayesianEstimate with mean, mode, intervals, etc.
        """
        α = self.alpha_posterior
        β = self.beta_posterior

        # Posterior mean: E[θ] = α / (α + β)
        mean = α / (α + β)

        # Posterior mode (MAP): (α - 1) / (α + β - 2) for α,β > 1
        if α > 1 and β > 1:
            mode = (α - 1) / (α + β - 2)
        else:
            mode = mean  # Use mean if mode undefined

        # Posterior median (50th percentile)
        median = stats.beta.ppf(0.5, α, β)

        # Posterior standard deviation: sqrt(αβ / ((α+β)²(α+β+1)))
        std = np.sqrt((α * β) / ((α + β)**2 * (α + β + 1)))

        # Credible intervals (Bayesian confidence intervals)
        ci_95 = (
            stats.beta.ppf(0.025, α, β),
            stats.beta.ppf(0.975, α, β)
        )
        ci_68 = (
            stats.beta.ppf(0.16, α, β),
            stats.beta.ppf(0.84, α, β)
        )

        return BayesianEstimate(
            mean=float(mean),
            mode=float(mode),
            median=float(median),
            std=float(std),
            credible_interval_95=ci_95,
            credible_interval_68=ci_68,
            num_observations=self.total_trades,
            timestamp=pd.Timestamp.now()
        )

    def get_current_estimate(self) -> BayesianEstimate:
        """
        Get current win rate estimate without updating

        Returns:
            Current BayesianEstimate
        """
        return self._calculate_estimate()

    def predict_next_n_trades(self, n: int) -> Dict:
        """
        Predict distribution of wins in next n trades

        Args:
            n: Number of future trades

        Returns:
            Dictionary with prediction statistics
        """
        α = self.alpha_posterior
        β = self.beta_posterior

        # Posterior predictive distribution: Beta-Binomial
        # E[wins] = n × E[θ] = n × α/(α+β)
        expected_wins = n * (α / (α + β))

        # Var[wins] = n × αβ(α+β+n) / ((α+β)²(α+β+1))
        var_wins = n * α * β * (α + β + n) / ((α + β)**2 * (α + β + 1))
        std_wins = np.sqrt(var_wins)

        # Most likely number of wins (mode)
        # For Beta-Binomial, approximate mode
        mode_wins = int(np.floor(n * (α / (α + β))))

        # Probability of at least k wins for various k
        win_probabilities = {}
        for k in [0, n//4, n//2, 3*n//4, n]:
            # P(wins ≥ k) using beta-binomial PMF
            prob = 1.0 - self._beta_binomial_cdf(k-1, n, α, β)
            win_probabilities[f'p_at_least_{k}_wins'] = float(prob)

        return {
            'n_trades': n,
            'expected_wins': float(expected_wins),
            'std_wins': float(std_wins),
            'mode_wins': mode_wins,
            'expected_win_rate': float(α / (α + β)),
            **win_probabilities
        }

    def _beta_binomial_cdf(self, k: int, n: int, α: float, β: float) -> float:
        """
        Beta-Binomial cumulative distribution function

        P(X ≤ k) for X ~ Beta-Binomial(n, α, β)
        """
        if k < 0:
            return 0.0
        if k >= n:
            return 1.0

        # Sum PMF from 0 to k
        cdf = 0.0
        for i in range(int(k) + 1):
            cdf += self._beta_binomial_pmf(i, n, α, β)

        return cdf

    def _beta_binomial_pmf(self, k: int, n: int, α: float, β: float) -> float:
        """
        Beta-Binomial probability mass function

        P(X = k) for X ~ Beta-Binomial(n, α, β)
        """
        if k < 0 or k > n:
            return 0.0

        # Use log-space for numerical stability
        log_pmf = (
            stats.binom.logpmf(k, n, 0.5)  # Binomial coefficient part
            + betaln(k + α, n - k + β)
            - betaln(α, β)
        )

        return np.exp(log_pmf)

    def compare_to_threshold(
        self,
        threshold: float,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        Compare win rate to threshold with Bayesian hypothesis testing

        Args:
            threshold: Win rate threshold (e.g., 0.6 for 60%)
            confidence_level: Confidence level for credible interval

        Returns:
            Dictionary with comparison statistics
        """
        α = self.alpha_posterior
        β = self.beta_posterior

        # P(θ > threshold) = 1 - Beta_CDF(threshold)
        prob_above_threshold = 1.0 - stats.beta.cdf(threshold, α, β)

        # P(θ < threshold)
        prob_below_threshold = stats.beta.cdf(threshold, α, β)

        # Credible interval at specified level
        lower_tail = (1 - confidence_level) / 2
        upper_tail = 1 - lower_tail
        credible_interval = (
            stats.beta.ppf(lower_tail, α, β),
            stats.beta.ppf(upper_tail, α, β)
        )

        # Decision: Is win rate likely above threshold?
        # Use confidence_level as decision boundary
        is_likely_above = prob_above_threshold > confidence_level

        estimate = self._calculate_estimate()

        return {
            'threshold': threshold,
            'estimated_win_rate': estimate.mean,
            'prob_above_threshold': float(prob_above_threshold),
            'prob_below_threshold': float(prob_below_threshold),
            'is_likely_above': is_likely_above,
            'credible_interval': credible_interval,
            'confidence_level': confidence_level,
            'num_observations': self.total_trades
        }

    def is_estimate_credible(self) -> bool:
        """
        Check if we have enough data for credible estimates

        Returns:
            True if estimate is credible (enough samples)
        """
        return self.total_trades >= self.min_credible_samples

    def reset(self):
        """Reset learner to prior state"""
        self.alpha_posterior = self.alpha_prior
        self.beta_posterior = self.beta_prior
        self.wins = 0
        self.losses = 0
        self.total_trades = 0
        self.history = []
        logger.info("Bayesian learner reset to prior")


# Convenience function for V7 runtime
def update_beliefs(
    outcomes: List[bool],
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0
) -> Dict:
    """
    Update beliefs from trading outcomes

    Args:
        outcomes: List of trade outcomes (True=win, False=loss)
        alpha_prior: Prior alpha parameter
        beta_prior: Prior beta parameter

    Returns:
        Dictionary with posterior estimates and predictions
    """
    learner = BayesianWinRateLearner(
        alpha_prior=alpha_prior,
        beta_prior=beta_prior
    )

    estimate = learner.update_batch(outcomes)
    prediction = learner.predict_next_n_trades(n=10)
    comparison = learner.compare_to_threshold(threshold=0.6)

    return {
        'posterior_estimate': {
            'mean': estimate.mean,
            'mode': estimate.mode,
            'std': estimate.std,
            'ci_95': estimate.credible_interval_95,
            'num_observations': estimate.num_observations
        },
        'prediction_next_10': prediction,
        'vs_60pct_threshold': comparison
    }


if __name__ == "__main__":
    # Test Bayesian Inference implementation
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
    print("Bayesian Win Rate Learner - Test Run")
    print("=" * 80)

    # Test 1: Online learning with sequential updates
    print("\n1. Testing Online Learning (Sequential Updates)")
    learner1 = BayesianWinRateLearner(alpha_prior=1.0, beta_prior=1.0)

    # Simulate 100 trades with 65% win rate
    np.random.seed(42)
    true_win_rate = 0.65
    outcomes = np.random.random(100) < true_win_rate

    estimates = []
    for i, outcome in enumerate(outcomes, 1):
        est = learner1.update(outcome)
        if i in [5, 10, 20, 50, 100]:
            estimates.append(est)
            print(f"   After {i:3d} trades: {est.mean:.3f} ± {est.std:.3f} "
                  f"[{est.credible_interval_95[0]:.3f}, {est.credible_interval_95[1]:.3f}]")

    # Test 2: Batch update
    print("\n2. Testing Batch Update")
    learner2 = BayesianWinRateLearner(alpha_prior=1.0, beta_prior=1.0)
    outcomes2 = np.random.random(50) < 0.70
    est2 = learner2.update_batch(outcomes2.tolist())
    print(f"   Batch: {learner2.total_trades} trades")
    print(f"   Win Rate: {est2.mean:.3f} ± {est2.std:.3f}")
    print(f"   95% CI: [{est2.credible_interval_95[0]:.3f}, {est2.credible_interval_95[1]:.3f}]")

    # Test 3: Prediction
    print("\n3. Testing Prediction (Next 20 Trades)")
    prediction = learner1.predict_next_n_trades(n=20)
    print(f"   Expected wins: {prediction['expected_wins']:.1f} ± {prediction['std_wins']:.1f}")
    print(f"   Most likely:   {prediction['mode_wins']} wins")
    print(f"   P(≥10 wins):   {prediction.get('p_at_least_10_wins', 0):.3f}")

    # Test 4: Threshold comparison
    print("\n4. Testing Threshold Comparison (vs 60%)")
    comparison = learner1.compare_to_threshold(threshold=0.60, confidence_level=0.95)
    print(f"   Estimated:     {comparison['estimated_win_rate']:.3f}")
    print(f"   P(θ > 60%):    {comparison['prob_above_threshold']:.3f}")
    print(f"   Is above?      {comparison['is_likely_above']}")
    print(f"   95% CI:        {comparison['credible_interval']}")

    # Test 5: Different priors
    print("\n5. Testing Different Priors")
    outcomes_small = [True, True, False, True, False]  # 3/5 wins

    # Uninformative prior (uniform)
    learner_uniform = BayesianWinRateLearner(alpha_prior=1.0, beta_prior=1.0)
    learner_uniform.update_batch(outcomes_small)
    est_uniform = learner_uniform.get_current_estimate()

    # Informative prior (expect 50% win rate)
    learner_informed = BayesianWinRateLearner(alpha_prior=10.0, beta_prior=10.0)
    learner_informed.update_batch(outcomes_small)
    est_informed = learner_informed.get_current_estimate()

    print(f"   Uniform prior:   {est_uniform.mean:.3f} ± {est_uniform.std:.3f}")
    print(f"   Informed prior:  {est_informed.mean:.3f} ± {est_informed.std:.3f}")
    print(f"   (Informed prior pulls estimate toward 50%)")

    # Visualization
    print("\n6. Creating Visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Convergence of estimate
    n_trades = list(range(1, 101))
    means = [learner1.history[i-1][1].mean for i in n_trades]
    ci_lower = [learner1.history[i-1][1].credible_interval_95[0] for i in n_trades]
    ci_upper = [learner1.history[i-1][1].credible_interval_95[1] for i in n_trades]

    axes[0, 0].plot(n_trades, means, label='Posterior Mean', color='blue', linewidth=2)
    axes[0, 0].fill_between(n_trades, ci_lower, ci_upper, alpha=0.3, label='95% CI')
    axes[0, 0].axhline(y=true_win_rate, color='green', linestyle='--',
                       label=f'True Rate ({true_win_rate})', linewidth=1.5)
    axes[0, 0].set_title("Convergence to True Win Rate")
    axes[0, 0].set_xlabel("Number of Trades")
    axes[0, 0].set_ylabel("Estimated Win Rate")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Posterior distribution evolution
    x = np.linspace(0, 1, 1000)
    for n in [5, 20, 100]:
        α = 1 + sum(outcomes[:n])
        β = 1 + n - sum(outcomes[:n])
        pdf = stats.beta.pdf(x, α, β)
        axes[0, 1].plot(x, pdf, label=f'After {n} trades', linewidth=2)

    axes[0, 1].axvline(x=true_win_rate, color='green', linestyle='--',
                       label='True Rate', linewidth=1.5)
    axes[0, 1].set_title("Posterior Distribution Evolution")
    axes[0, 1].set_xlabel("Win Rate (θ)")
    axes[0, 1].set_ylabel("Probability Density")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Uncertainty reduction
    stds = [learner1.history[i-1][1].std for i in n_trades]
    axes[1, 0].plot(n_trades, stds, color='red', linewidth=2)
    axes[1, 0].set_title("Uncertainty Reduction Over Time")
    axes[1, 0].set_xlabel("Number of Trades")
    axes[1, 0].set_ylabel("Standard Deviation")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Prior vs Posterior comparison
    x = np.linspace(0, 1, 1000)
    prior_pdf = stats.beta.pdf(x, 1, 1)  # Uniform
    posterior_pdf = stats.beta.pdf(x, learner1.alpha_posterior, learner1.beta_posterior)

    axes[1, 1].plot(x, prior_pdf, label='Prior (Uniform)', linestyle='--',
                    color='gray', linewidth=2)
    axes[1, 1].plot(x, posterior_pdf, label=f'Posterior ({learner1.total_trades} trades)',
                    color='blue', linewidth=2)
    axes[1, 1].axvline(x=true_win_rate, color='green', linestyle='--',
                       label='True Rate', linewidth=1.5)
    axes[1, 1].set_title("Prior vs Posterior Distribution")
    axes[1, 1].set_xlabel("Win Rate (θ)")
    axes[1, 1].set_ylabel("Probability Density")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/bayesian_inference_test.png', dpi=100, bbox_inches='tight')
    print(f"   Saved visualization to /tmp/bayesian_inference_test.png")

    print("\n" + "=" * 80)
    print("Bayesian Inference Test Complete!")
    print("=" * 80)
    print("\nKey Insights:")
    print("  - Bayesian updates beliefs incrementally with each trade")
    print("  - Uncertainty decreases as more data is observed")
    print("  - Credible intervals quantify estimate uncertainty")
    print("  - Prior knowledge can be incorporated (informed priors)")
    print("  - Predictions include uncertainty (probability distributions)")
    print("=" * 80)
