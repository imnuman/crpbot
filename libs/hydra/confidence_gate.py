"""
HYDRA 4.0 - Confidence Threshold Gate

Calculates composite confidence score and gates live execution.
Threshold: 80% confidence required for live trading.

Components:
- Historical WR (30% weight)
- Paper WR (30% weight)
- Multi-regime validation (20% weight)
- Specialty match (20% weight)
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceResult:
    """Result of confidence calculation."""
    strategy_id: str
    total_confidence: float
    should_execute: bool

    # Component scores (0-1)
    historical_score: float
    paper_score: float
    regime_score: float
    specialty_score: float

    # Details
    reason: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "total_confidence": self.total_confidence,
            "should_execute": self.should_execute,
            "historical_score": self.historical_score,
            "paper_score": self.paper_score,
            "regime_score": self.regime_score,
            "specialty_score": self.specialty_score,
            "reason": self.reason,
            "details": self.details,
        }


class ConfidenceGate:
    """
    Calculates composite confidence and gates live execution.

    Confidence Components:
    - Historical WR: 30% weight (backtest performance)
    - Paper WR: 30% weight (paper trading performance)
    - Multi-regime: 20% weight (performs in multiple regimes)
    - Specialty match: 20% weight (current market matches specialty)

    Threshold: 80% for live execution
    """

    # Weights for confidence components
    WEIGHTS = {
        "historical": 0.30,
        "paper": 0.30,
        "regime": 0.20,
        "specialty": 0.20,
    }

    # Threshold for live execution
    CONFIDENCE_THRESHOLD = 0.80  # 80%

    # Scoring parameters
    HISTORICAL_WR_IDEAL = 0.60  # 60% WR = perfect score
    PAPER_WR_IDEAL = 0.65  # 65% WR = perfect score
    MIN_REGIME_COUNT = 2  # Perform in at least 2 regimes for full score

    def __init__(self):
        """Initialize the confidence gate."""
        logger.info("[ConfidenceGate] Initialized")
        logger.info(f"[ConfidenceGate] Threshold: {self.CONFIDENCE_THRESHOLD*100}%")
        logger.info(f"[ConfidenceGate] Weights: {self.WEIGHTS}")

    def calculate_confidence(
        self,
        strategy_id: str,
        historical_wr: float,
        paper_wr: float,
        regime_performance: Dict[str, Dict],
        current_regime: str,
        strategy_specialty: str,
        specialty_triggered: bool,
        specialty_strength: float = 0.0
    ) -> ConfidenceResult:
        """
        Calculate composite confidence score.

        Args:
            strategy_id: Strategy identifier
            historical_wr: Backtest win rate (0-1)
            paper_wr: Paper trading win rate (0-1)
            regime_performance: Dict of regime -> {trades, wins}
            current_regime: Current market regime
            strategy_specialty: Strategy's specialty type
            specialty_triggered: Whether specialty is currently triggered
            specialty_strength: Strength of specialty trigger (0-1)

        Returns:
            ConfidenceResult with scores and execution decision
        """
        # Calculate component scores
        historical_score = self._score_historical(historical_wr)
        paper_score = self._score_paper(paper_wr)
        regime_score = self._score_regime(regime_performance, current_regime)
        specialty_score = self._score_specialty(specialty_triggered, specialty_strength)

        # Weighted sum
        total = (
            self.WEIGHTS["historical"] * historical_score +
            self.WEIGHTS["paper"] * paper_score +
            self.WEIGHTS["regime"] * regime_score +
            self.WEIGHTS["specialty"] * specialty_score
        )

        # Determine if should execute
        should_execute = total >= self.CONFIDENCE_THRESHOLD

        # Build reason
        if should_execute:
            reason = f"Confidence {total*100:.1f}% >= {self.CONFIDENCE_THRESHOLD*100}% threshold"
        else:
            # Find weakest component
            scores = {
                "historical": historical_score,
                "paper": paper_score,
                "regime": regime_score,
                "specialty": specialty_score,
            }
            weakest = min(scores, key=scores.get)
            reason = f"Confidence {total*100:.1f}% < {self.CONFIDENCE_THRESHOLD*100}% (weakest: {weakest}={scores[weakest]*100:.1f}%)"

        return ConfidenceResult(
            strategy_id=strategy_id,
            total_confidence=round(total, 4),
            should_execute=should_execute,
            historical_score=round(historical_score, 4),
            paper_score=round(paper_score, 4),
            regime_score=round(regime_score, 4),
            specialty_score=round(specialty_score, 4),
            reason=reason,
            details={
                "historical_wr": historical_wr,
                "paper_wr": paper_wr,
                "current_regime": current_regime,
                "specialty_triggered": specialty_triggered,
                "specialty_strength": specialty_strength,
            }
        )

    def _score_historical(self, win_rate: float) -> float:
        """
        Score historical backtest performance.

        Score reaches 1.0 at HISTORICAL_WR_IDEAL (60%), capped at 1.0.
        """
        if win_rate <= 0:
            return 0.0

        # Linear scaling up to ideal, then capped
        score = min(win_rate / self.HISTORICAL_WR_IDEAL, 1.0)
        return score

    def _score_paper(self, win_rate: float) -> float:
        """
        Score paper trading performance.

        Score reaches 1.0 at PAPER_WR_IDEAL (65%), capped at 1.0.
        """
        if win_rate <= 0:
            return 0.0

        score = min(win_rate / self.PAPER_WR_IDEAL, 1.0)
        return score

    def _score_regime(
        self,
        regime_performance: Dict[str, Dict],
        current_regime: str
    ) -> float:
        """
        Score multi-regime validation.

        Higher score for:
        - Performing in multiple regimes (diversity)
        - Good performance in current regime
        """
        if not regime_performance:
            return 0.0

        # Count regimes with trades
        regimes_with_trades = sum(
            1 for r, p in regime_performance.items()
            if p.get("trades", 0) >= 3
        )

        # Diversity score (0-0.5)
        diversity_score = min(regimes_with_trades / self.MIN_REGIME_COUNT, 1.0) * 0.5

        # Current regime score (0-0.5)
        current_perf = regime_performance.get(current_regime, {})
        current_trades = current_perf.get("trades", 0)
        current_wins = current_perf.get("wins", 0)

        if current_trades >= 3:
            current_wr = current_wins / current_trades
            current_score = min(current_wr / 0.55, 1.0) * 0.5  # 55% WR = full score
        else:
            current_score = 0.25  # Partial score if untested in current regime

        return diversity_score + current_score

    def _score_specialty(
        self,
        triggered: bool,
        strength: float
    ) -> float:
        """
        Score specialty match.

        - 0.0 if specialty not triggered
        - 0.5-1.0 based on trigger strength
        """
        if not triggered:
            return 0.0

        # Base score for trigger + strength bonus
        base = 0.5
        strength_bonus = strength * 0.5

        return min(base + strength_bonus, 1.0)

    def should_execute(
        self,
        strategy_id: str,
        historical_wr: float,
        paper_wr: float,
        regime_performance: Dict[str, Dict],
        current_regime: str,
        strategy_specialty: str,
        specialty_triggered: bool,
        specialty_strength: float = 0.0
    ) -> bool:
        """
        Quick check if strategy should execute.

        Returns True if confidence >= threshold.
        """
        result = self.calculate_confidence(
            strategy_id=strategy_id,
            historical_wr=historical_wr,
            paper_wr=paper_wr,
            regime_performance=regime_performance,
            current_regime=current_regime,
            strategy_specialty=strategy_specialty,
            specialty_triggered=specialty_triggered,
            specialty_strength=specialty_strength
        )
        return result.should_execute

    def check_strategy_from_data(
        self,
        strategy_data: Dict[str, Any],
        current_regime: str,
        specialty_triggered: bool,
        specialty_strength: float = 0.0
    ) -> ConfidenceResult:
        """
        Check strategy using data from strategy memory.

        Args:
            strategy_data: Strategy dict containing performance data
            current_regime: Current market regime
            specialty_triggered: Whether specialty is triggered
            specialty_strength: Trigger strength

        Returns:
            ConfidenceResult
        """
        return self.calculate_confidence(
            strategy_id=strategy_data.get("strategy_id", "unknown"),
            historical_wr=strategy_data.get("backtest_wr", strategy_data.get("win_rate", 0)),
            paper_wr=strategy_data.get("paper_wr", 0),
            regime_performance=strategy_data.get("regime_performance", {}),
            current_regime=current_regime,
            strategy_specialty=strategy_data.get("specialty", ""),
            specialty_triggered=specialty_triggered,
            specialty_strength=specialty_strength
        )


# Singleton instance
_gate_instance: Optional[ConfidenceGate] = None


def get_confidence_gate() -> ConfidenceGate:
    """Get or create the confidence gate singleton."""
    global _gate_instance
    if _gate_instance is None:
        _gate_instance = ConfidenceGate()
    return _gate_instance
