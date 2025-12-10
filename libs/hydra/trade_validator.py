"""
HYDRA 3.0 - Trade Validator

Centralized validation for all trade signals:
- 70% minimum confidence threshold (MOD 4)
- Correlation check between engines (MOD 5)
- Integration with engine specialization

No trade executes without passing ALL validators.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Configuration
MIN_CONFIDENCE = 0.70  # 70% minimum confidence
MAX_CORRELATION = 0.80  # Engines must not be too similar (>80% = reject)
MIN_HISTORICAL_TRADES = 5  # Need 5 trades before correlation check kicks in

# LESSON LEARNED: BUY=100% WR, SELL=27% WR across all engines
# Disable SELL until short detection improves
ALLOW_SHORT_TRADES = False  # Set to True when short detection is fixed


@dataclass
class TradeProposal:
    """A proposed trade from an engine."""
    engine_id: str
    symbol: str
    direction: str  # LONG, SHORT
    trigger_type: str  # liquidation_cascade, funding_extreme, etc.
    trigger_value: float
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    reasoning: str
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ValidationResult:
    """Result of trade validation."""
    approved: bool
    proposal: TradeProposal
    checks: dict  # Individual check results
    rejection_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "approved": self.approved,
            "engine_id": self.proposal.engine_id,
            "symbol": self.proposal.symbol,
            "direction": self.proposal.direction,
            "confidence": self.proposal.confidence,
            "checks": self.checks,
            "rejection_reason": self.rejection_reason,
        }


class TradeValidator:
    """
    Validates all trade proposals before execution.

    Checks:
    1. Minimum 70% confidence
    2. Engine specialty match (via SpecialtyValidator)
    3. Correlation check (engines shouldn't agree too often)
    4. Engine D special rules (14-day cooldown)
    """

    _instance: Optional["TradeValidator"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Track recent proposals for correlation check
        self._recent_proposals: list[TradeProposal] = []
        self._proposal_window_hours = 24

        # Stats tracking
        self._total_validated = 0
        self._total_rejected = 0
        self._rejection_reasons: dict[str, int] = {}

        self._initialized = True
        logger.info("TradeValidator initialized")

    def validate(self, proposal: TradeProposal) -> ValidationResult:
        """
        Validate a trade proposal.

        Args:
            proposal: The trade proposal to validate

        Returns:
            ValidationResult with approval status
        """
        checks = {}

        # Check 0: Direction filter (BUY only until shorts improve)
        checks["direction"] = self._check_direction(proposal)

        # Check 1: Minimum confidence (70%)
        checks["confidence"] = self._check_confidence(proposal)

        # Check 2: Engine specialty match
        checks["specialty"] = self._check_specialty(proposal)

        # Check 3: Engine D special rules (if applicable)
        if proposal.engine_id.upper() == "D":
            checks["engine_d_rules"] = self._check_engine_d(proposal)

        # Check 4: Correlation check
        checks["correlation"] = self._check_correlation(proposal)

        # Determine overall approval
        approved = all(check["passed"] for check in checks.values())

        # Find rejection reason
        rejection_reason = None
        if not approved:
            for check_name, check_result in checks.items():
                if not check_result["passed"]:
                    rejection_reason = f"{check_name}: {check_result['reason']}"
                    break

        # Store proposal for correlation tracking
        self._add_proposal(proposal)

        result = ValidationResult(
            approved=approved,
            proposal=proposal,
            checks=checks,
            rejection_reason=rejection_reason,
        )

        # Track stats
        if approved:
            self._total_validated += 1
            logger.info(f"Trade APPROVED: Engine {proposal.engine_id} {proposal.direction} {proposal.symbol}")
        else:
            self._total_rejected += 1
            # Track rejection reason
            reason_key = rejection_reason.split(":")[0] if rejection_reason else "unknown"
            self._rejection_reasons[reason_key] = self._rejection_reasons.get(reason_key, 0) + 1
            logger.warning(f"Trade REJECTED: {rejection_reason}")

        return result

    def _check_direction(self, proposal: TradeProposal) -> dict:
        """
        Check if trade direction is allowed.

        LESSON LEARNED from 44 trades:
        - BUY trades: 100% win rate (13/13)
        - SELL trades: 27% win rate (8/31)

        Until short detection improves, only allow BUY/LONG.
        """
        direction = proposal.direction.upper()
        is_long = direction in ("LONG", "BUY")

        if ALLOW_SHORT_TRADES:
            # Shorts enabled - allow all directions
            return {
                "passed": True,
                "direction": direction,
                "reason": f"Direction {direction} allowed (shorts enabled)",
            }

        if is_long:
            return {
                "passed": True,
                "direction": direction,
                "reason": f"Direction {direction} allowed (BUY only mode)",
            }
        else:
            return {
                "passed": False,
                "direction": direction,
                "reason": f"SHORT/SELL disabled (historical WR: 27%). Set ALLOW_SHORT_TRADES=True to enable.",
            }

    def _check_confidence(self, proposal: TradeProposal) -> dict:
        """Check minimum 70% confidence."""
        passed = proposal.confidence >= MIN_CONFIDENCE
        return {
            "passed": passed,
            "value": proposal.confidence,
            "threshold": MIN_CONFIDENCE,
            "reason": f"Confidence {proposal.confidence:.0%}" if passed else f"Confidence {proposal.confidence:.0%} below {MIN_CONFIDENCE:.0%}",
        }

    def _check_specialty(self, proposal: TradeProposal) -> dict:
        """Check engine specialty match."""
        try:
            from .engine_specialization import get_specialty_validator

            validator = get_specialty_validator()
            result = validator.validate_trigger(
                proposal.engine_id,
                proposal.trigger_type,
                proposal.trigger_value,
                proposal.confidence,
            )
            return {
                "passed": result["valid"],
                "expected": validator.get_specialty(proposal.engine_id).specialty.value if validator.get_specialty(proposal.engine_id) else None,
                "actual": proposal.trigger_type,
                "reason": result["reason"],
            }
        except ImportError:
            # Specialty validator not available
            return {"passed": True, "reason": "Specialty check skipped (not available)"}

    def _check_engine_d(self, proposal: TradeProposal) -> dict:
        """Check Engine D special rules (14-day cooldown, expectancy)."""
        try:
            from .engine_d_rules import check_engine_d_activation

            # Extract ATR multiplier from trigger value for regime transitions
            atr_multiplier = proposal.trigger_value
            result = check_engine_d_activation(atr_multiplier, proposal.confidence)
            return {
                "passed": result["allowed"],
                "can_activate": result["allowed"],
                "state": result.get("state", {}),
                "reason": result["reason"],
            }
        except ImportError:
            return {"passed": True, "reason": "Engine D check skipped (not available)"}

    def _check_correlation(self, proposal: TradeProposal) -> dict:
        """
        Check if engines are too correlated (making same calls).

        If engines agree on >80% of trades in the same direction,
        they're not providing independent signals. This defeats the
        purpose of the multi-engine system.
        """
        # Get recent proposals from OTHER engines
        now = datetime.now()
        cutoff = now - timedelta(hours=self._proposal_window_hours)

        other_proposals = [
            p for p in self._recent_proposals
            if p.engine_id != proposal.engine_id
            and p.timestamp > cutoff
            and p.symbol == proposal.symbol
        ]

        if len(other_proposals) < MIN_HISTORICAL_TRADES:
            return {
                "passed": True,
                "correlation": 0.0,
                "sample_size": len(other_proposals),
                "reason": f"Insufficient data ({len(other_proposals)}/{MIN_HISTORICAL_TRADES} trades)",
            }

        # Check direction agreement
        same_direction = sum(1 for p in other_proposals if p.direction == proposal.direction)
        correlation = same_direction / len(other_proposals)

        passed = correlation <= MAX_CORRELATION

        return {
            "passed": passed,
            "correlation": correlation,
            "same_direction": same_direction,
            "total": len(other_proposals),
            "threshold": MAX_CORRELATION,
            "reason": f"Engine correlation {correlation:.0%}" if passed else f"Too correlated ({correlation:.0%} > {MAX_CORRELATION:.0%}) - engines agreeing too often",
        }

    def _add_proposal(self, proposal: TradeProposal):
        """Add proposal to history for correlation tracking."""
        self._recent_proposals.append(proposal)

        # Cleanup old proposals
        cutoff = datetime.now() - timedelta(hours=self._proposal_window_hours)
        self._recent_proposals = [
            p for p in self._recent_proposals
            if p.timestamp > cutoff
        ]

    def get_correlation_matrix(self) -> dict:
        """
        Get correlation matrix between engines.

        Shows how often each engine agrees with others.
        """
        engines = ["A", "B", "C", "D"]
        matrix = {}

        now = datetime.now()
        cutoff = now - timedelta(hours=self._proposal_window_hours)
        recent = [p for p in self._recent_proposals if p.timestamp > cutoff]

        for e1 in engines:
            e1_proposals = [p for p in recent if p.engine_id == e1]
            if not e1_proposals:
                continue

            matrix[e1] = {}
            for e2 in engines:
                if e1 == e2:
                    matrix[e1][e2] = 1.0  # Self-correlation is 100%
                    continue

                # Find matching proposals (same symbol, same time window)
                agreements = 0
                comparisons = 0

                for p1 in e1_proposals:
                    # Find e2 proposals for same symbol within 1 hour
                    e2_matches = [
                        p for p in recent
                        if p.engine_id == e2
                        and p.symbol == p1.symbol
                        and abs((p.timestamp - p1.timestamp).total_seconds()) < 3600
                    ]
                    for p2 in e2_matches:
                        comparisons += 1
                        if p1.direction == p2.direction:
                            agreements += 1

                matrix[e1][e2] = agreements / comparisons if comparisons > 0 else 0.0

        return matrix

    def get_stats(self) -> dict:
        """Get validation statistics."""
        now = datetime.now()
        cutoff = now - timedelta(hours=24)
        recent = [p for p in self._recent_proposals if p.timestamp > cutoff]

        by_engine = {}
        for p in recent:
            if p.engine_id not in by_engine:
                by_engine[p.engine_id] = {"total": 0, "long": 0, "short": 0}
            by_engine[p.engine_id]["total"] += 1
            if p.direction == "LONG":
                by_engine[p.engine_id]["long"] += 1
            else:
                by_engine[p.engine_id]["short"] += 1

        return {
            "proposals_24h": len(recent),
            "by_engine": by_engine,
            "correlation_matrix": self.get_correlation_matrix(),
            "total_validated": self._total_validated,
            "total_rejected": self._total_rejected,
            "rejection_reasons": self._rejection_reasons,
        }


# Singleton accessor
_validator_instance: Optional[TradeValidator] = None


def get_trade_validator() -> TradeValidator:
    """Get or create the trade validator singleton."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = TradeValidator()
    return _validator_instance


# Convenience function
def validate_trade(proposal: TradeProposal) -> ValidationResult:
    """Quick validate a trade proposal."""
    return get_trade_validator().validate(proposal)
