"""
HYDRA 3.0 - Monte Carlo Validator

Fast Monte Carlo simulation for strategy validation (< 1 second).

Uses bootstrap resampling to:
1. Shuffle trade returns many times (10,000 simulations)
2. Calculate statistics across all simulations
3. Provide confidence intervals for key metrics
4. Determine if results are due to skill vs luck

Key Outputs:
- P-value for strategy edge (is it statistically significant?)
- Confidence intervals for Sharpe, win rate, max drawdown
- Probability of ruin (chance of hitting stop-loss)
- Expected range of outcomes

Performance: Vectorized numpy operations for < 1 second execution.

Phase 2, Week 2 - Step 18
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger
import json
import time
import math

# Use numpy for fast vectorized operations
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.warning("[MonteCarloValidator] numpy not available, using fallback")


@dataclass
class MonteCarloResult:
    """Result of Monte Carlo simulation."""
    timestamp: datetime
    engine: str
    execution_time_ms: float

    # Input stats
    num_trades: int
    actual_sharpe: float
    actual_win_rate: float
    actual_total_pnl: float
    actual_max_drawdown: float

    # Simulation stats
    num_simulations: int

    # Sharpe ratio analysis
    sharpe_mean: float
    sharpe_std: float
    sharpe_5th_percentile: float
    sharpe_95th_percentile: float
    sharpe_p_value: float  # Probability of getting this Sharpe by chance

    # Win rate analysis
    win_rate_mean: float
    win_rate_std: float
    win_rate_5th_percentile: float
    win_rate_95th_percentile: float

    # Drawdown analysis
    max_dd_mean: float
    max_dd_std: float
    max_dd_5th_percentile: float
    max_dd_95th_percentile: float
    probability_of_ruin: float  # Prob of hitting -20% drawdown

    # P&L analysis
    pnl_mean: float
    pnl_std: float
    pnl_5th_percentile: float
    pnl_95th_percentile: float

    # Statistical significance
    is_statistically_significant: bool
    confidence_level: float  # 0-1 confidence in edge
    luck_score: float  # 0-1, higher = more likely due to luck

    # Recommendations
    recommendations: List[str]


class MonteCarloValidator:
    """
    Fast Monte Carlo Validator for HYDRA 3.0.

    Uses bootstrap resampling with vectorized numpy operations
    to run 10,000 simulations in under 1 second.
    """

    # Configuration
    DEFAULT_SIMULATIONS = 10_000  # Number of Monte Carlo runs
    SIGNIFICANCE_THRESHOLD = 0.05  # p-value threshold for significance
    RUIN_THRESHOLD = -0.20  # -20% drawdown = ruin

    def __init__(self, data_dir: Optional[Path] = None):
        # Auto-detect data directory based on environment
        if data_dir is None:
            from ..config import HYDRA_DATA_DIR
            data_dir = HYDRA_DATA_DIR

        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Results history
        self.results_history: List[MonteCarloResult] = []

        # Persistence
        self.results_file = self.data_dir / "monte_carlo_results.jsonl"

        # Random seed for reproducibility (optional)
        self.rng = np.random.default_rng() if HAS_NUMPY else None

        logger.info("[MonteCarloValidator] Initialized")

    def validate(
        self,
        returns: List[float],
        engine: str,
        num_simulations: int = None
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation on trade returns.

        Args:
            returns: List of trade P&L percentages
            engine: Engine name being validated
            num_simulations: Number of simulations (default: 10,000)

        Returns:
            MonteCarloResult with full analysis
        """
        start_time = time.perf_counter()
        n_sims = num_simulations or self.DEFAULT_SIMULATIONS

        if not HAS_NUMPY:
            return self._fallback_validate(returns, engine, n_sims)

        # Convert to numpy array
        returns_arr = np.array(returns, dtype=np.float64)
        n_trades = len(returns_arr)

        if n_trades < 5:
            return self._insufficient_data_result(engine, n_trades)

        # Calculate actual metrics
        actual_sharpe = self._calc_sharpe_np(returns_arr)
        actual_win_rate = np.mean(returns_arr > 0)
        actual_total_pnl = np.sum(returns_arr)
        actual_max_dd = self._calc_max_drawdown_np(returns_arr)

        # Run Monte Carlo simulation (vectorized)
        sim_results = self._run_simulations_np(returns_arr, n_sims)

        # Extract statistics
        sharpe_sims = sim_results['sharpe']
        wr_sims = sim_results['win_rate']
        dd_sims = sim_results['max_drawdown']
        pnl_sims = sim_results['total_pnl']

        # Calculate percentiles and stats
        sharpe_stats = self._calc_stats_np(sharpe_sims)
        wr_stats = self._calc_stats_np(wr_sims)
        dd_stats = self._calc_stats_np(dd_sims)
        pnl_stats = self._calc_stats_np(pnl_sims)

        # Calculate p-value for Sharpe (one-tailed test)
        # What fraction of random shuffles produced Sharpe >= actual?
        sharpe_p_value = np.mean(sharpe_sims >= actual_sharpe)

        # Probability of ruin
        prob_ruin = np.mean(dd_sims >= abs(self.RUIN_THRESHOLD))

        # Statistical significance
        is_significant = sharpe_p_value < self.SIGNIFICANCE_THRESHOLD

        # Confidence level (inverse of p-value, capped)
        confidence = min(1.0 - sharpe_p_value, 0.99)

        # Luck score (higher p-value = more likely luck)
        luck_score = min(sharpe_p_value * 2, 1.0)  # Scale for readability

        execution_time = (time.perf_counter() - start_time) * 1000

        # Generate recommendations
        recommendations = self._generate_recommendations(
            actual_sharpe, actual_win_rate, sharpe_p_value,
            prob_ruin, is_significant, luck_score
        )

        result = MonteCarloResult(
            timestamp=datetime.now(timezone.utc),
            engine=engine,
            execution_time_ms=execution_time,
            num_trades=n_trades,
            actual_sharpe=actual_sharpe,
            actual_win_rate=actual_win_rate,
            actual_total_pnl=actual_total_pnl,
            actual_max_drawdown=actual_max_dd,
            num_simulations=n_sims,
            sharpe_mean=sharpe_stats['mean'],
            sharpe_std=sharpe_stats['std'],
            sharpe_5th_percentile=sharpe_stats['p5'],
            sharpe_95th_percentile=sharpe_stats['p95'],
            sharpe_p_value=sharpe_p_value,
            win_rate_mean=wr_stats['mean'],
            win_rate_std=wr_stats['std'],
            win_rate_5th_percentile=wr_stats['p5'],
            win_rate_95th_percentile=wr_stats['p95'],
            max_dd_mean=dd_stats['mean'],
            max_dd_std=dd_stats['std'],
            max_dd_5th_percentile=dd_stats['p5'],
            max_dd_95th_percentile=dd_stats['p95'],
            probability_of_ruin=prob_ruin,
            pnl_mean=pnl_stats['mean'],
            pnl_std=pnl_stats['std'],
            pnl_5th_percentile=pnl_stats['p5'],
            pnl_95th_percentile=pnl_stats['p95'],
            is_statistically_significant=is_significant,
            confidence_level=confidence,
            luck_score=luck_score,
            recommendations=recommendations
        )

        # Save and log
        self._save_result(result)
        self._log_result(result)

        self.results_history.append(result)

        return result

    def _run_simulations_np(
        self,
        returns: np.ndarray,
        n_sims: int
    ) -> Dict[str, np.ndarray]:
        """
        Run vectorized Monte Carlo simulations.

        Uses numpy broadcasting for speed.
        """
        n_trades = len(returns)

        # Generate all random indices at once (n_sims x n_trades)
        indices = self.rng.integers(0, n_trades, size=(n_sims, n_trades))

        # Bootstrap resample all simulations at once
        sim_returns = returns[indices]  # Shape: (n_sims, n_trades)

        # Calculate metrics for all simulations (vectorized)
        sharpe_arr = self._calc_sharpe_batch_np(sim_returns)
        win_rate_arr = np.mean(sim_returns > 0, axis=1)
        total_pnl_arr = np.sum(sim_returns, axis=1)
        max_dd_arr = self._calc_max_drawdown_batch_np(sim_returns)

        return {
            'sharpe': sharpe_arr,
            'win_rate': win_rate_arr,
            'total_pnl': total_pnl_arr,
            'max_drawdown': max_dd_arr
        }

    def _calc_sharpe_np(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio for single return series."""
        if len(returns) < 2:
            return 0.0

        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)

        if std_ret == 0:
            return 0.0

        # Annualized
        return float(mean_ret / std_ret * np.sqrt(252))

    def _calc_sharpe_batch_np(self, returns_batch: np.ndarray) -> np.ndarray:
        """Calculate Sharpe ratio for batch of simulations."""
        mean_ret = np.mean(returns_batch, axis=1)
        std_ret = np.std(returns_batch, axis=1, ddof=1)

        # Avoid division by zero
        std_ret = np.where(std_ret == 0, 1e-10, std_ret)

        return mean_ret / std_ret * np.sqrt(252)

    def _calc_max_drawdown_np(self, returns: np.ndarray) -> float:
        """Calculate max drawdown for single return series."""
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative

        return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    def _calc_max_drawdown_batch_np(self, returns_batch: np.ndarray) -> np.ndarray:
        """Calculate max drawdown for batch of simulations."""
        cumulative = np.cumsum(returns_batch, axis=1)
        running_max = np.maximum.accumulate(cumulative, axis=1)
        drawdowns = running_max - cumulative

        return np.max(drawdowns, axis=1)

    def _calc_stats_np(self, arr: np.ndarray) -> Dict[str, float]:
        """Calculate summary statistics for array."""
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'p5': float(np.percentile(arr, 5)),
            'p95': float(np.percentile(arr, 95)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr))
        }

    def _fallback_validate(
        self,
        returns: List[float],
        engine: str,
        n_sims: int
    ) -> MonteCarloResult:
        """Fallback validation without numpy (slower)."""
        import random
        start_time = time.perf_counter()

        n_trades = len(returns)
        if n_trades < 5:
            return self._insufficient_data_result(engine, n_trades)

        # Actual metrics
        actual_sharpe = self._calc_sharpe_simple(returns)
        actual_win_rate = sum(1 for r in returns if r > 0) / n_trades
        actual_total_pnl = sum(returns)
        actual_max_dd = self._calc_max_drawdown_simple(returns)

        # Run simulations (slower without numpy)
        sharpe_sims = []
        wr_sims = []
        dd_sims = []
        pnl_sims = []

        # Reduce simulations for fallback
        n_sims = min(n_sims, 1000)

        for _ in range(n_sims):
            shuffled = random.choices(returns, k=n_trades)
            sharpe_sims.append(self._calc_sharpe_simple(shuffled))
            wr_sims.append(sum(1 for r in shuffled if r > 0) / n_trades)
            dd_sims.append(self._calc_max_drawdown_simple(shuffled))
            pnl_sims.append(sum(shuffled))

        # Calculate stats
        def stats(arr):
            arr = sorted(arr)
            n = len(arr)
            return {
                'mean': sum(arr) / n,
                'std': (sum((x - sum(arr)/n)**2 for x in arr) / n) ** 0.5,
                'p5': arr[int(n * 0.05)],
                'p95': arr[int(n * 0.95)]
            }

        sharpe_stats = stats(sharpe_sims)
        wr_stats = stats(wr_sims)
        dd_stats = stats(dd_sims)
        pnl_stats = stats(pnl_sims)

        sharpe_p_value = sum(1 for s in sharpe_sims if s >= actual_sharpe) / n_sims
        prob_ruin = sum(1 for d in dd_sims if d >= abs(self.RUIN_THRESHOLD)) / n_sims

        is_significant = sharpe_p_value < self.SIGNIFICANCE_THRESHOLD
        confidence = min(1.0 - sharpe_p_value, 0.99)
        luck_score = min(sharpe_p_value * 2, 1.0)

        execution_time = (time.perf_counter() - start_time) * 1000

        recommendations = self._generate_recommendations(
            actual_sharpe, actual_win_rate, sharpe_p_value,
            prob_ruin, is_significant, luck_score
        )

        return MonteCarloResult(
            timestamp=datetime.now(timezone.utc),
            engine=engine,
            execution_time_ms=execution_time,
            num_trades=n_trades,
            actual_sharpe=actual_sharpe,
            actual_win_rate=actual_win_rate,
            actual_total_pnl=actual_total_pnl,
            actual_max_drawdown=actual_max_dd,
            num_simulations=n_sims,
            sharpe_mean=sharpe_stats['mean'],
            sharpe_std=sharpe_stats['std'],
            sharpe_5th_percentile=sharpe_stats['p5'],
            sharpe_95th_percentile=sharpe_stats['p95'],
            sharpe_p_value=sharpe_p_value,
            win_rate_mean=wr_stats['mean'],
            win_rate_std=wr_stats['std'],
            win_rate_5th_percentile=wr_stats['p5'],
            win_rate_95th_percentile=wr_stats['p95'],
            max_dd_mean=dd_stats['mean'],
            max_dd_std=dd_stats['std'],
            max_dd_5th_percentile=dd_stats['p5'],
            max_dd_95th_percentile=dd_stats['p95'],
            probability_of_ruin=prob_ruin,
            pnl_mean=pnl_stats['mean'],
            pnl_std=pnl_stats['std'],
            pnl_5th_percentile=pnl_stats['p5'],
            pnl_95th_percentile=pnl_stats['p95'],
            is_statistically_significant=is_significant,
            confidence_level=confidence,
            luck_score=luck_score,
            recommendations=recommendations
        )

    def _calc_sharpe_simple(self, returns: List[float]) -> float:
        """Simple Sharpe calculation without numpy."""
        if len(returns) < 2:
            return 0.0

        n = len(returns)
        mean_ret = sum(returns) / n
        variance = sum((r - mean_ret) ** 2 for r in returns) / (n - 1)
        std_ret = variance ** 0.5

        if std_ret == 0:
            return 0.0

        return mean_ret / std_ret * (252 ** 0.5)

    def _calc_max_drawdown_simple(self, returns: List[float]) -> float:
        """Simple max drawdown calculation."""
        cumulative = 0
        peak = 0
        max_dd = 0

        for r in returns:
            cumulative += r
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def _insufficient_data_result(self, engine: str, n_trades: int) -> MonteCarloResult:
        """Return result for insufficient data."""
        return MonteCarloResult(
            timestamp=datetime.now(timezone.utc),
            engine=engine,
            execution_time_ms=0,
            num_trades=n_trades,
            actual_sharpe=0,
            actual_win_rate=0,
            actual_total_pnl=0,
            actual_max_drawdown=0,
            num_simulations=0,
            sharpe_mean=0, sharpe_std=0,
            sharpe_5th_percentile=0, sharpe_95th_percentile=0,
            sharpe_p_value=1.0,
            win_rate_mean=0, win_rate_std=0,
            win_rate_5th_percentile=0, win_rate_95th_percentile=0,
            max_dd_mean=0, max_dd_std=0,
            max_dd_5th_percentile=0, max_dd_95th_percentile=0,
            probability_of_ruin=0,
            pnl_mean=0, pnl_std=0,
            pnl_5th_percentile=0, pnl_95th_percentile=0,
            is_statistically_significant=False,
            confidence_level=0,
            luck_score=1.0,
            recommendations=[f"Insufficient data: {n_trades} trades (need at least 5)"]
        )

    def _generate_recommendations(
        self,
        sharpe: float,
        win_rate: float,
        p_value: float,
        prob_ruin: float,
        is_significant: bool,
        luck_score: float
    ) -> List[str]:
        """Generate recommendations based on Monte Carlo results."""
        recs = []

        if is_significant:
            recs.append(
                f"SIGNIFICANT EDGE: p-value={p_value:.3f} indicates "
                f"results unlikely due to chance (confidence: {(1-p_value)*100:.1f}%)"
            )
        else:
            recs.append(
                f"NOT SIGNIFICANT: p-value={p_value:.3f} suggests results "
                f"could be due to luck. Need more trades or larger edge."
            )

        if luck_score > 0.5:
            recs.append(
                f"HIGH LUCK SCORE ({luck_score:.2f}): Results may not be repeatable. "
                "Consider gathering more data."
            )

        if prob_ruin > 0.1:
            recs.append(
                f"RUIN RISK: {prob_ruin:.1%} probability of hitting -20% drawdown. "
                "Consider reducing position sizes."
            )

        if sharpe < 0.5 and is_significant:
            recs.append(
                f"Low Sharpe ({sharpe:.2f}) despite significance. "
                "Edge exists but risk-adjusted returns are poor."
            )

        if win_rate < 0.4:
            recs.append(
                f"Low win rate ({win_rate:.1%}). Strategy relies heavily on "
                "few large wins. Vulnerable to variance."
            )
        elif win_rate > 0.7:
            recs.append(
                f"High win rate ({win_rate:.1%}) is promising. "
                "Verify this holds in different market conditions."
            )

        return recs

    def _save_result(self, result: MonteCarloResult):
        """Save result to file."""
        try:
            entry = {
                "timestamp": result.timestamp.isoformat(),
                "engine": result.engine,
                "execution_time_ms": float(result.execution_time_ms),
                "num_trades": int(result.num_trades),
                "num_simulations": int(result.num_simulations),
                "actual_sharpe": float(result.actual_sharpe),
                "actual_win_rate": float(result.actual_win_rate),
                "sharpe_p_value": float(result.sharpe_p_value),
                "is_significant": bool(result.is_statistically_significant),
                "confidence_level": float(result.confidence_level),
                "luck_score": float(result.luck_score),
                "probability_of_ruin": float(result.probability_of_ruin)
            }

            with open(self.results_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')

        except Exception as e:
            logger.warning(f"[MonteCarloValidator] Failed to save: {e}")

    def _log_result(self, result: MonteCarloResult):
        """Log Monte Carlo result."""
        sig_icon = "✅" if result.is_statistically_significant else "⚠️"

        logger.info("=" * 60)
        logger.info(f"MONTE CARLO VALIDATION - Engine {result.engine}")
        logger.info(f"Execution Time: {result.execution_time_ms:.1f}ms")
        logger.info("=" * 60)
        logger.info(f"Trades: {result.num_trades} | Simulations: {result.num_simulations}")
        logger.info(f"")
        logger.info(f"ACTUAL PERFORMANCE:")
        logger.info(f"  Sharpe: {result.actual_sharpe:.2f}")
        logger.info(f"  Win Rate: {result.actual_win_rate:.1%}")
        logger.info(f"  Total P&L: {result.actual_total_pnl:.2f}%")
        logger.info(f"  Max Drawdown: {result.actual_max_drawdown:.2f}%")
        logger.info(f"")
        logger.info(f"STATISTICAL ANALYSIS:")
        logger.info(f"  {sig_icon} Significant: {result.is_statistically_significant}")
        logger.info(f"  p-value: {result.sharpe_p_value:.4f}")
        logger.info(f"  Confidence: {result.confidence_level:.1%}")
        logger.info(f"  Luck Score: {result.luck_score:.2f}")
        logger.info(f"")
        logger.info(f"CONFIDENCE INTERVALS (5th-95th percentile):")
        logger.info(f"  Sharpe: [{result.sharpe_5th_percentile:.2f}, {result.sharpe_95th_percentile:.2f}]")
        logger.info(f"  Win Rate: [{result.win_rate_5th_percentile:.1%}, {result.win_rate_95th_percentile:.1%}]")
        logger.info(f"  P&L: [{result.pnl_5th_percentile:.2f}%, {result.pnl_95th_percentile:.2f}%]")
        logger.info(f"")
        logger.info(f"RISK:")
        logger.info(f"  Probability of Ruin: {result.probability_of_ruin:.1%}")
        logger.info(f"")
        logger.info(f"RECOMMENDATIONS:")
        for rec in result.recommendations:
            logger.info(f"  • {rec}")
        logger.info("=" * 60)

    def validate_quick(self, returns: List[float], engine: str) -> Dict[str, Any]:
        """
        Quick validation returning just essential metrics.

        For use in real-time decision making where full report isn't needed.
        """
        result = self.validate(returns, engine, num_simulations=1000)

        return {
            "engine": engine,
            "is_significant": result.is_statistically_significant,
            "p_value": result.sharpe_p_value,
            "confidence": result.confidence_level,
            "luck_score": result.luck_score,
            "prob_ruin": result.probability_of_ruin,
            "execution_ms": result.execution_time_ms
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get validator summary."""
        return {
            "validations_run": len(self.results_history),
            "has_numpy": HAS_NUMPY,
            "default_simulations": self.DEFAULT_SIMULATIONS,
            "significance_threshold": self.SIGNIFICANCE_THRESHOLD,
            "ruin_threshold": self.RUIN_THRESHOLD
        }


# ==================== SINGLETON PATTERN ====================

_monte_carlo_validator: Optional[MonteCarloValidator] = None

def get_monte_carlo_validator() -> MonteCarloValidator:
    """Get singleton instance of MonteCarloValidator."""
    global _monte_carlo_validator
    if _monte_carlo_validator is None:
        _monte_carlo_validator = MonteCarloValidator()
    return _monte_carlo_validator
