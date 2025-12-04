"""
HYDRA 3.0 - Consensus System (Layer 6)

Multi-agent voting system that combines gladiator decisions.

Voting rules:
- 4/4 agree → 100% position size
- 3/4 agree → 75% position size
- 2/4 agree → 50% position size
- <2/4 → NO TRADE

This reduces risk while allowing trades with strong consensus.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from loguru import logger


class ConsensusEngine:
    """
    Aggregates votes from all 4 engines into final decision.

    Each engine votes: BUY, SELL, or HOLD
    Consensus determines: trade or not + position size modifier

    Position sizing now incorporates engine weights:
    - Base modifier from vote count (unanimous/strong/weak)
    - Weighted by total weight of agreeing engines
    """

    # Consensus thresholds
    UNANIMOUS_MODIFIER = 1.0  # 4/4 agree = 100% position
    STRONG_MODIFIER = 0.75    # 3/4 agree = 75% position
    WEAK_MODIFIER = 0.5       # 2/4 agree = 50% position
    MIN_VOTES_REQUIRED = 2    # Need at least 2 votes for same direction

    # Default engine weights (will be updated from tournament manager)
    DEFAULT_WEIGHTS = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}

    def __init__(self):
        self.vote_history = []
        self.engine_weights = self.DEFAULT_WEIGHTS.copy()
        logger.info("Consensus Engine initialized")

    def update_weights(self, weights: Dict[str, float]) -> None:
        """
        Update engine weights from tournament manager.

        Args:
            weights: Dict mapping engine name to weight (0.0-1.0)
        """
        for engine, weight in weights.items():
            if engine in self.engine_weights:
                self.engine_weights[engine] = weight
        logger.debug(f"Consensus weights updated: {self.engine_weights}")

    def get_consensus(
        self,
        votes: List[Dict],
        require_unanimous: bool = False
    ) -> Dict:
        """
        Calculate consensus from engine votes.

        Args:
            votes: List of vote dicts from each engine
                [
                    {"gladiator": "A", "vote": "BUY", "confidence": 0.7, "reasoning": "..."},
                    {"gladiator": "B", "vote": "BUY", "confidence": 0.6, "reasoning": "..."},
                    {"gladiator": "C", "vote": "HOLD", "confidence": 0.5, "reasoning": "..."},
                    {"gladiator": "D", "vote": "BUY", "confidence": 0.8, "reasoning": "..."}
                ]
            require_unanimous: If True, require 4/4 agreement (for high-risk assets)

        Returns:
            {
                "action": "BUY" | "SELL" | "HOLD",
                "consensus_level": "UNANIMOUS" | "STRONG" | "WEAK" | "NO_CONSENSUS",
                "position_size_modifier": 1.0 | 0.75 | 0.5 | 0.0,
                "votes_for": 3,
                "votes_against": 0,
                "votes_hold": 1,
                "avg_confidence": 0.7,
                "dissenting_gladiators": ["C"],
                "dissenting_reasons": ["Gladiator C: Backtest shows lower win rate"],
                "summary": "3/4 gladiators agree on BUY (STRONG consensus)"
            }
        """
        if not votes or len(votes) != 4:
            logger.error(f"Invalid votes: expected 4, got {len(votes)}")
            return self._no_consensus()

        # Count votes by direction
        buy_votes = [v for v in votes if v.get("vote") == "BUY"]
        sell_votes = [v for v in votes if v.get("vote") == "SELL"]
        hold_votes = [v for v in votes if v.get("vote") == "HOLD"]

        # Determine primary direction
        if len(buy_votes) > len(sell_votes):
            primary_direction = "BUY"
            votes_for = len(buy_votes)
            votes_against = len(sell_votes)
        elif len(sell_votes) > len(buy_votes):
            primary_direction = "SELL"
            votes_for = len(sell_votes)
            votes_against = len(buy_votes)
        else:
            # Tie between BUY and SELL, or all HOLD
            return self._no_consensus()

        # Check if we meet minimum threshold
        if votes_for < self.MIN_VOTES_REQUIRED:
            return self._no_consensus()

        # Check unanimous requirement
        if require_unanimous and votes_for != 4:
            logger.warning(f"Unanimous consensus required but only {votes_for}/4 agree")
            return self._no_consensus()

        # Determine consensus level
        if votes_for == 4:
            consensus_level = "UNANIMOUS"
            base_modifier = self.UNANIMOUS_MODIFIER
        elif votes_for == 3:
            consensus_level = "STRONG"
            base_modifier = self.STRONG_MODIFIER
        elif votes_for == 2:
            consensus_level = "WEAK"
            base_modifier = self.WEAK_MODIFIER
        else:
            return self._no_consensus()

        # Calculate weighted position modifier based on agreeing engines
        # Get engines that voted for the primary direction
        if primary_direction == "BUY":
            agreeing_votes = buy_votes
        else:
            agreeing_votes = sell_votes

        # Sum weights of agreeing engines
        agreeing_weight = sum(
            self.engine_weights.get(v.get("gladiator", ""), 0.25)
            for v in agreeing_votes
        )

        # Position modifier = base_modifier * (agreeing_weight / 0.5)
        # If agreeing engines have >50% weight, boost position
        # If agreeing engines have <50% weight, reduce position
        # Capped between 0.5 and 1.2
        weight_multiplier = min(1.2, max(0.8, agreeing_weight / 0.5))
        position_modifier = base_modifier * weight_multiplier

        # Ensure position modifier stays within reasonable bounds
        position_modifier = min(1.2, max(0.4, position_modifier))

        # Get dissenting gladiators
        if primary_direction == "BUY":
            dissenters = sell_votes + hold_votes
        else:
            dissenters = buy_votes + hold_votes

        dissenting_gladiators = [d.get("gladiator", "?") for d in dissenters]
        dissenting_reasons = [
            f"Gladiator {d.get('gladiator', '?')}: {d.get('reasoning', 'No reason')}"
            for d in dissenters
        ]

        # Calculate average confidence
        avg_confidence = sum(v.get("confidence", 0) for v in votes) / len(votes)

        result = {
            "action": primary_direction,
            "consensus_level": consensus_level,
            "position_size_modifier": position_modifier,
            "base_modifier": base_modifier,
            "agreeing_weight": agreeing_weight,
            "weight_multiplier": weight_multiplier,
            "votes_for": votes_for,
            "votes_against": votes_against,
            "votes_hold": len(hold_votes),
            "avg_confidence": avg_confidence,
            "dissenting_gladiators": dissenting_gladiators,
            "dissenting_reasons": dissenting_reasons,
            "summary": f"{votes_for}/4 gladiators agree on {primary_direction} ({consensus_level} consensus, {agreeing_weight:.0%} weight)",
            "all_votes": votes
        }

        # Log consensus
        self._log_consensus(result)

        logger.info(
            f"Consensus: {primary_direction} ({consensus_level}) - "
            f"{votes_for}/4 votes, weight={agreeing_weight:.0%}, size={position_modifier:.0%}"
        )

        return result

    def get_tie_breaker_vote(
        self,
        votes: List[Dict],
        tie_breaker_gladiator: str = "D"
    ) -> Dict:
        """
        In case of 2-2 tie, use specific gladiator as tie-breaker.

        Default: Gladiator D (Synthesizer) breaks ties.
        """
        buy_count = sum(1 for v in votes if v.get("vote") == "BUY")
        sell_count = sum(1 for v in votes if v.get("vote") == "SELL")

        if buy_count != sell_count:
            # Not a tie, use normal consensus
            return self.get_consensus(votes)

        # It's a tie - find tie-breaker vote
        tie_breaker = next((v for v in votes if v.get("gladiator") == tie_breaker_gladiator), None)

        if not tie_breaker:
            logger.error(f"Tie-breaker gladiator {tie_breaker_gladiator} not found")
            return self._no_consensus()

        tie_direction = tie_breaker.get("vote")

        if tie_direction == "HOLD":
            logger.info(f"Tie-breaker (Gladiator {tie_breaker_gladiator}) voted HOLD - no consensus")
            return self._no_consensus()

        # Count votes in tie-breaker direction
        votes_with_tie_breaker = sum(1 for v in votes if v.get("vote") == tie_direction)

        result = {
            "action": tie_direction,
            "consensus_level": "WEAK",
            "position_size_modifier": self.WEAK_MODIFIER,
            "votes_for": votes_with_tie_breaker,
            "votes_against": 4 - votes_with_tie_breaker - sum(1 for v in votes if v.get("vote") == "HOLD"),
            "votes_hold": sum(1 for v in votes if v.get("vote") == "HOLD"),
            "avg_confidence": sum(v.get("confidence", 0) for v in votes) / len(votes),
            "tie_broken_by": tie_breaker_gladiator,
            "summary": f"2-2 tie broken by Gladiator {tie_breaker_gladiator}: {tie_direction}",
            "all_votes": votes
        }

        logger.info(f"Tie broken by Gladiator {tie_breaker_gladiator}: {tie_direction}")

        return result

    def _no_consensus(self) -> Dict:
        """Return result when no consensus reached."""
        return {
            "action": "HOLD",
            "consensus_level": "NO_CONSENSUS",
            "position_size_modifier": 0.0,
            "votes_for": 0,
            "votes_against": 0,
            "votes_hold": 0,
            "avg_confidence": 0.0,
            "dissenting_gladiators": [],
            "dissenting_reasons": [],
            "summary": "No consensus reached - holding position",
            "all_votes": []
        }

    def _log_consensus(self, result: Dict):
        """Log consensus to history."""
        self.vote_history.append({
            "timestamp": datetime.now(timezone.utc),
            **result
        })

        # Keep only last 1000 consensus results
        if len(self.vote_history) > 1000:
            self.vote_history = self.vote_history[-1000:]

    # ==================== ANALYSIS METHODS ====================

    def get_consensus_stats(self, hours: int = 24) -> Dict:
        """
        Get consensus statistics over last N hours.

        Returns:
            {
                "total_votes": 50,
                "unanimous": 10,
                "strong": 25,
                "weak": 10,
                "no_consensus": 5,
                "avg_confidence": 0.68,
                "most_active_gladiator": "A"
            }
        """
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent = [v for v in self.vote_history if v["timestamp"] > cutoff]

        if not recent:
            return {
                "total_votes": 0,
                "unanimous": 0,
                "strong": 0,
                "weak": 0,
                "no_consensus": 0,
                "avg_confidence": 0.0
            }

        unanimous = sum(1 for v in recent if v.get("consensus_level") == "UNANIMOUS")
        strong = sum(1 for v in recent if v.get("consensus_level") == "STRONG")
        weak = sum(1 for v in recent if v.get("consensus_level") == "WEAK")
        no_consensus = sum(1 for v in recent if v.get("consensus_level") == "NO_CONSENSUS")

        return {
            "total_votes": len(recent),
            "unanimous": unanimous,
            "strong": strong,
            "weak": weak,
            "no_consensus": no_consensus,
            "avg_confidence": sum(v.get("avg_confidence", 0) for v in recent) / len(recent)
        }

    def get_engine_agreement_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate how often each pair of gladiators agree.

        Returns:
            {
                "A-B": 0.85,  # A and B agree 85% of time
                "A-C": 0.72,
                ...
            }
        """
        agreement_counts = {}
        total_votes = {}

        gladiators = ["A", "B", "C", "D"]

        for i in range(len(gladiators)):
            for j in range(i+1, len(gladiators)):
                pair = f"{gladiators[i]}-{gladiators[j]}"
                agreement_counts[pair] = 0
                total_votes[pair] = 0

        for vote_result in self.vote_history:
            all_votes = vote_result.get("all_votes", [])
            if len(all_votes) != 4:
                continue

            for i in range(len(gladiators)):
                for j in range(i+1, len(gladiators)):
                    pair = f"{gladiators[i]}-{gladiators[j]}"

                    vote_i = next((v for v in all_votes if v.get("gladiator") == gladiators[i]), None)
                    vote_j = next((v for v in all_votes if v.get("gladiator") == gladiators[j]), None)

                    if vote_i and vote_j:
                        total_votes[pair] += 1
                        if vote_i.get("vote") == vote_j.get("vote"):
                            agreement_counts[pair] += 1

        agreement_rates = {}
        for pair in agreement_counts:
            if total_votes[pair] > 0:
                agreement_rates[pair] = agreement_counts[pair] / total_votes[pair]
            else:
                agreement_rates[pair] = 0.0

        return agreement_rates


# Global singleton instance
_consensus_engine = None

def get_consensus_engine() -> ConsensusEngine:
    """Get global ConsensusEngine singleton."""
    global _consensus_engine
    if _consensus_engine is None:
        _consensus_engine = ConsensusEngine()
    return _consensus_engine
