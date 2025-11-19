"""
V7 Ultimate - Bayesian Learning Module

Continuously improves V7's win rate estimates and confidence calibration
by learning from actual trade outcomes using Bayesian inference.

Uses Beta distribution to model win rate uncertainty and updates beliefs
as new trade outcomes (wins/losses) are recorded.
"""

from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import math

from loguru import logger
from sqlalchemy import and_

from libs.db.models import Signal, get_session


@dataclass
class BayesianEstimate:
    """Bayesian win rate estimate with uncertainty"""
    mean: float  # Expected win rate
    std: float   # Standard deviation (uncertainty)
    alpha: float # Beta distribution alpha parameter (wins + 1)
    beta: float  # Beta distribution beta parameter (losses + 1)
    confidence: float  # Confidence in estimate (0-1)
    sample_size: int   # Number of trades observed


class BayesianLearner:
    """
    Bayesian learning system for V7 Ultimate

    Learns from actual trade outcomes to improve:
    1. Win rate estimates per signal type (BUY/SELL)
    2. Win rate estimates per symbol
    3. Overall confidence calibration
    4. Market regime adaptability
    """

    def __init__(self, db_url: str):
        """
        Initialize Bayesian learner

        Args:
            db_url: Database connection string
        """
        self.db_url = db_url

        # Prior beliefs (weak priors - start neutral)
        # Alpha=1, Beta=1 = uniform prior (no bias)
        self.prior_alpha = 1.0
        self.prior_beta = 1.0

    def get_win_rate_estimate(
        self,
        signal_type: Optional[str] = None,
        symbol: Optional[str] = None,
        days: int = 30
    ) -> BayesianEstimate:
        """
        Get Bayesian win rate estimate for specific conditions

        Args:
            signal_type: Filter by 'long' or 'short' (None = all)
            symbol: Filter by symbol (None = all)
            days: Number of days to analyze

        Returns:
            BayesianEstimate with mean, std, and confidence
        """
        try:
            session = get_session(self.db_url)
            try:
                # Get closed trades from last N days
                since = datetime.utcnow() - timedelta(days=days)

                filters = [
                    Signal.model_version == 'v7_ultimate',
                    Signal.executed == 1,
                    Signal.execution_time >= since,
                    Signal.result.in_(['win', 'loss'])
                ]

                if signal_type:
                    filters.append(Signal.direction == signal_type)
                if symbol:
                    filters.append(Signal.symbol == symbol)

                trades = session.query(Signal).filter(and_(*filters)).all()

                # Count wins and losses
                wins = sum(1 for t in trades if t.result == 'win')
                losses = sum(1 for t in trades if t.result == 'loss')
                total = wins + losses

                if total == 0:
                    # No data - return prior
                    return BayesianEstimate(
                        mean=0.5,  # Neutral prior
                        std=0.289,  # High uncertainty (Beta(1,1) std)
                        alpha=self.prior_alpha,
                        beta=self.prior_beta,
                        confidence=0.0,  # No confidence without data
                        sample_size=0
                    )

                # Update Beta distribution parameters
                # Posterior: Beta(alpha + wins, beta + losses)
                alpha = self.prior_alpha + wins
                beta = self.prior_beta + losses

                # Calculate posterior mean and std
                mean = alpha / (alpha + beta)
                variance = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
                std = math.sqrt(variance)

                # Calculate confidence (inverse of uncertainty)
                # More samples = higher confidence
                # Confidence approaches 1.0 as sample size increases
                confidence = 1.0 - (1.0 / (1.0 + math.log(total + 1)))

                return BayesianEstimate(
                    mean=mean,
                    std=std,
                    alpha=alpha,
                    beta=beta,
                    confidence=confidence,
                    sample_size=total
                )

            finally:
                session.close()

        except Exception as e:
            logger.error(f"Error calculating Bayesian estimate: {e}")
            # Return prior on error
            return BayesianEstimate(
                mean=0.5,
                std=0.289,
                alpha=self.prior_alpha,
                beta=self.prior_beta,
                confidence=0.0,
                sample_size=0
            )

    def get_confidence_calibration(self, days: int = 30) -> Dict[str, float]:
        """
        Analyze confidence calibration (are high-confidence signals actually better?)

        Args:
            days: Number of days to analyze

        Returns:
            Dict with calibration metrics per confidence bucket
        """
        try:
            session = get_session(self.db_url)
            try:
                since = datetime.utcnow() - timedelta(days=days)

                trades = session.query(Signal).filter(
                    and_(
                        Signal.model_version == 'v7_ultimate',
                        Signal.executed == 1,
                        Signal.execution_time >= since,
                        Signal.result.in_(['win', 'loss'])
                    )
                ).all()

                if not trades:
                    return {}

                # Group by confidence buckets
                buckets = {
                    'low': (0.0, 0.6),      # <60%
                    'medium': (0.6, 0.75),  # 60-75%
                    'high': (0.75, 1.0)     # >75%
                }

                calibration = {}
                for bucket_name, (min_conf, max_conf) in buckets.items():
                    bucket_trades = [
                        t for t in trades
                        if min_conf <= t.confidence < max_conf
                    ]

                    if not bucket_trades:
                        continue

                    wins = sum(1 for t in bucket_trades if t.result == 'win')
                    total = len(bucket_trades)
                    actual_win_rate = wins / total if total > 0 else 0.0
                    avg_predicted_conf = sum(t.confidence for t in bucket_trades) / total

                    calibration[bucket_name] = {
                        'predicted_win_rate': avg_predicted_conf,
                        'actual_win_rate': actual_win_rate,
                        'calibration_error': abs(avg_predicted_conf - actual_win_rate),
                        'sample_size': total
                    }

                return calibration

            finally:
                session.close()

        except Exception as e:
            logger.error(f"Error analyzing confidence calibration: {e}")
            return {}

    def get_adaptive_confidence_adjustment(
        self,
        base_confidence: float,
        signal_type: str,
        symbol: str
    ) -> float:
        """
        Adjust model's base confidence using Bayesian learning

        Args:
            base_confidence: Model's raw confidence prediction
            signal_type: 'long' or 'short'
            symbol: Trading symbol

        Returns:
            Adjusted confidence incorporating historical performance
        """
        try:
            # Get Bayesian estimates for this specific context
            signal_estimate = self.get_win_rate_estimate(
                signal_type=signal_type,
                symbol=None,  # All symbols for signal type
                days=30
            )

            symbol_estimate = self.get_win_rate_estimate(
                signal_type=None,  # All signal types
                symbol=symbol,
                days=30
            )

            # If no data, return base confidence
            if signal_estimate.sample_size == 0 and symbol_estimate.sample_size == 0:
                return base_confidence

            # Weight adjustments by sample size and confidence
            adjustments = []
            weights = []

            if signal_estimate.sample_size > 0:
                adjustments.append(signal_estimate.mean)
                weights.append(signal_estimate.confidence * signal_estimate.sample_size)

            if symbol_estimate.sample_size > 0:
                adjustments.append(symbol_estimate.mean)
                weights.append(symbol_estimate.confidence * symbol_estimate.sample_size)

            # Add base confidence with lower weight
            adjustments.append(base_confidence)
            weights.append(1.0)  # Low weight for base model

            # Weighted average
            total_weight = sum(weights)
            adjusted_confidence = sum(
                adj * w for adj, w in zip(adjustments, weights)
            ) / total_weight

            # Ensure within bounds [0, 1]
            adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))

            logger.debug(
                f"Confidence adjustment: {base_confidence:.3f} → {adjusted_confidence:.3f} "
                f"(signal_wr={signal_estimate.mean:.3f}, symbol_wr={symbol_estimate.mean:.3f})"
            )

            return adjusted_confidence

        except Exception as e:
            logger.error(f"Error adjusting confidence: {e}")
            return base_confidence

    def get_learning_summary(self, days: int = 30) -> Dict:
        """
        Get comprehensive Bayesian learning summary

        Args:
            days: Number of days to analyze

        Returns:
            Dict with learning metrics and insights
        """
        try:
            # Overall estimate
            overall = self.get_win_rate_estimate(days=days)

            # By signal type
            long_estimate = self.get_win_rate_estimate(signal_type='long', days=days)
            short_estimate = self.get_win_rate_estimate(signal_type='short', days=days)

            # By symbol
            btc_estimate = self.get_win_rate_estimate(symbol='BTC-USD', days=days)
            eth_estimate = self.get_win_rate_estimate(symbol='ETH-USD', days=days)
            sol_estimate = self.get_win_rate_estimate(symbol='SOL-USD', days=days)

            # Confidence calibration
            calibration = self.get_confidence_calibration(days=days)

            return {
                'period_days': days,
                'overall': {
                    'win_rate': overall.mean,
                    'uncertainty': overall.std,
                    'confidence': overall.confidence,
                    'sample_size': overall.sample_size
                },
                'by_signal_type': {
                    'long': {
                        'win_rate': long_estimate.mean,
                        'uncertainty': long_estimate.std,
                        'sample_size': long_estimate.sample_size
                    },
                    'short': {
                        'win_rate': short_estimate.mean,
                        'uncertainty': short_estimate.std,
                        'sample_size': short_estimate.sample_size
                    }
                },
                'by_symbol': {
                    'BTC-USD': {
                        'win_rate': btc_estimate.mean,
                        'uncertainty': btc_estimate.std,
                        'sample_size': btc_estimate.sample_size
                    },
                    'ETH-USD': {
                        'win_rate': eth_estimate.mean,
                        'uncertainty': eth_estimate.std,
                        'sample_size': eth_estimate.sample_size
                    },
                    'SOL-USD': {
                        'win_rate': sol_estimate.mean,
                        'uncertainty': sol_estimate.std,
                        'sample_size': sol_estimate.sample_size
                    }
                },
                'calibration': calibration
            }

        except Exception as e:
            logger.error(f"Error generating learning summary: {e}")
            return {}
