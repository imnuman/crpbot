"""
HYDRA 3.0 - Breeding Engine (Layer 5)

Genetic algorithm for strategy evolution.

Breeding = Crossover + Mutation

Crossover:
- Take entry logic from Parent A
- Take exit logic from Parent B
- Combine filters from both
- Average risk parameters

Mutation:
- 10% chance to mutate each component
- Small random changes to preserve parent DNA
- Mutations are conservative (no wild changes)

This is how HYDRA evolves better strategies over time.
"""

from typing import Dict, List, Optional, Tuple
from loguru import logger
import random
import copy


class BreedingEngine:
    """
    Creates offspring strategies from parent strategies using genetic algorithms.

    Process:
    1. Select two parent strategies (winners from tournament)
    2. Crossover: Mix DNA (entry/exit logic, filters, parameters)
    3. Mutation: Small random changes (10% per gene)
    4. Validate: Ensure offspring is coherent
    5. Return: New strategy ready for tournament
    """

    # Mutation probabilities
    MUTATION_RATE = 0.10  # 10% chance per component
    MUTATION_STRENGTH = 0.15  # ±15% max change

    # Crossover strategies
    CROSSOVER_TYPES = [
        "half_half",  # 50% from each parent
        "best_of_both",  # Take best components from each
        "weighted_fitness"  # Weight by parent fitness scores
    ]

    def __init__(self):
        self.offspring_count = 0
        self.breeding_history = []  # Trimmed to last 1000 after each breed
        logger.info("Breeding Engine initialized")

    def breed(
        self,
        parent1: Dict,
        parent2: Dict,
        parent1_fitness: float,
        parent2_fitness: float,
        crossover_type: str = "weighted_fitness"
    ) -> Dict:
        """
        Breed two parent strategies to create offspring.

        Args:
            parent1: First parent strategy dict (from engine)
            parent2: Second parent strategy dict (from engine)
            parent1_fitness: Fitness score of parent 1 (0-1)
            parent2_fitness: Fitness score of parent 2 (0-1)
            crossover_type: "half_half" | "best_of_both" | "weighted_fitness"

        Returns:
            Offspring strategy dict with:
            - strategy_id: "OFFSPRING_0001"
            - parent_ids: [parent1_id, parent2_id]
            - All strategy components (entry, exit, filters, etc.)
            - mutation_log: What was mutated
        """
        logger.info(
            f"Breeding {parent1.get('strategy_id', '?')} (fitness {parent1_fitness:.2f}) "
            f"x {parent2.get('strategy_id', '?')} (fitness {parent2_fitness:.2f})"
        )

        # Deep copy to avoid mutating parents
        p1 = copy.deepcopy(parent1)
        p2 = copy.deepcopy(parent2)

        # Crossover
        if crossover_type == "half_half":
            offspring = self._crossover_half_half(p1, p2)
        elif crossover_type == "best_of_both":
            offspring = self._crossover_best_of_both(p1, p2, parent1_fitness, parent2_fitness)
        elif crossover_type == "weighted_fitness":
            offspring = self._crossover_weighted(p1, p2, parent1_fitness, parent2_fitness)
        else:
            logger.warning(f"Unknown crossover type {crossover_type}, defaulting to weighted")
            offspring = self._crossover_weighted(p1, p2, parent1_fitness, parent2_fitness)

        # Mutation
        offspring, mutation_log = self._mutate(offspring)

        # Finalize offspring
        self.offspring_count += 1
        offspring["strategy_id"] = f"OFFSPRING_{self.offspring_count:04d}"
        offspring["parent_ids"] = [
            p1.get("strategy_id", "unknown"),
            p2.get("strategy_id", "unknown")
        ]
        offspring["generation"] = max(
            p1.get("generation", 0),
            p2.get("generation", 0)
        ) + 1
        offspring["mutation_log"] = mutation_log
        offspring["crossover_type"] = crossover_type

        # Log breeding
        self.breeding_history.append({
            "offspring_id": offspring["strategy_id"],
            "parent1_id": p1.get("strategy_id"),
            "parent2_id": p2.get("strategy_id"),
            "parent1_fitness": parent1_fitness,
            "parent2_fitness": parent2_fitness,
            "crossover_type": crossover_type,
            "mutations": mutation_log
        })
        # Trim to prevent memory leak
        if len(self.breeding_history) > 1000:
            self.breeding_history = self.breeding_history[-1000:]

        logger.success(
            f"Offspring {offspring['strategy_id']} created (generation {offspring['generation']})"
        )

        return offspring

    # ==================== CROSSOVER STRATEGIES ====================

    def _crossover_half_half(self, parent1: Dict, parent2: Dict) -> Dict:
        """
        50/50 crossover: Flip coin for each component.

        - Entry logic: Random choice
        - Exit logic: Random choice
        - Filters: Union of both
        - Risk params: Average
        - Edge: From whoever provided entry
        """
        offspring = {}

        # Flip coin for entry logic
        if random.random() < 0.5:
            offspring["entry_rules"] = parent1.get("entry_rules", "")
            offspring["structural_edge"] = parent1.get("structural_edge", "")
            offspring["entry_parent"] = "parent1"
        else:
            offspring["entry_rules"] = parent2.get("entry_rules", "")
            offspring["structural_edge"] = parent2.get("structural_edge", "")
            offspring["entry_parent"] = "parent2"

        # Flip coin for exit logic
        if random.random() < 0.5:
            offspring["exit_rules"] = parent1.get("exit_rules", "")
            offspring["exit_parent"] = "parent1"
        else:
            offspring["exit_rules"] = parent2.get("exit_rules", "")
            offspring["exit_parent"] = "parent2"

        # Combine filters (union)
        filters1 = set(parent1.get("filters", []))
        filters2 = set(parent2.get("filters", []))
        offspring["filters"] = list(filters1.union(filters2))

        # Average risk parameters
        offspring["risk_per_trade"] = (
            parent1.get("risk_per_trade", 0.01) +
            parent2.get("risk_per_trade", 0.01)
        ) / 2

        # Average expected performance (this will be recalculated)
        offspring["expected_wr"] = (
            parent1.get("expected_wr", 0.5) +
            parent2.get("expected_wr", 0.5)
        ) / 2

        offspring["expected_rr"] = (
            parent1.get("expected_rr", 1.0) +
            parent2.get("expected_rr", 1.0)
        ) / 2

        # Strategy name
        offspring["strategy_name"] = f"Hybrid: {parent1.get('strategy_name', 'P1')[:20]} x {parent2.get('strategy_name', 'P2')[:20]}"

        # Combine why_it_works
        offspring["why_it_works"] = (
            f"Entry: {offspring.get('entry_rules', 'N/A')[:100]}. "
            f"Exit: {offspring.get('exit_rules', 'N/A')[:100]}"
        )

        # Combine weaknesses
        weaknesses1 = parent1.get("weaknesses", [])
        weaknesses2 = parent2.get("weaknesses", [])
        offspring["weaknesses"] = list(set(weaknesses1 + weaknesses2))

        return offspring

    def _crossover_best_of_both(
        self,
        parent1: Dict,
        parent2: Dict,
        fitness1: float,
        fitness2: float
    ) -> Dict:
        """
        Best-of-both crossover: Take components from better parent.

        - Entry: From fitter parent
        - Exit: From fitter parent
        - Filters: Union
        - Risk: From more conservative parent
        """
        offspring = {}

        # Determine fitter parent
        fitter = parent1 if fitness1 >= fitness2 else parent2

        # Take entry/exit from fitter parent
        offspring["entry_rules"] = fitter.get("entry_rules", "")
        offspring["exit_rules"] = fitter.get("exit_rules", "")
        offspring["structural_edge"] = fitter.get("structural_edge", "")

        # Combine filters
        filters1 = set(parent1.get("filters", []))
        filters2 = set(parent2.get("filters", []))
        offspring["filters"] = list(filters1.union(filters2))

        # More conservative risk (lower value)
        offspring["risk_per_trade"] = min(
            parent1.get("risk_per_trade", 0.01),
            parent2.get("risk_per_trade", 0.01)
        )

        # Use fitter parent's expectations
        offspring["expected_wr"] = fitter.get("expected_wr", 0.5)
        offspring["expected_rr"] = fitter.get("expected_rr", 1.0)

        # Strategy name
        offspring["strategy_name"] = f"Elite: {fitter.get('strategy_name', 'Parent')[:30]}"

        offspring["why_it_works"] = fitter.get("why_it_works", "Inherited from fitter parent")
        offspring["weaknesses"] = list(set(
            parent1.get("weaknesses", []) + parent2.get("weaknesses", [])
        ))

        return offspring

    def _crossover_weighted(
        self,
        parent1: Dict,
        parent2: Dict,
        fitness1: float,
        fitness2: float
    ) -> Dict:
        """
        Weighted crossover: Probability of selection based on fitness.

        Higher fitness = higher chance of contributing each component.
        """
        offspring = {}

        # Calculate selection probabilities
        total_fitness = fitness1 + fitness2
        if total_fitness > 0:
            p1_prob = fitness1 / total_fitness
        else:
            p1_prob = 0.5

        # Entry logic: Weighted random
        if random.random() < p1_prob:
            offspring["entry_rules"] = parent1.get("entry_rules", "")
            offspring["structural_edge"] = parent1.get("structural_edge", "")
            offspring["entry_parent"] = "parent1"
        else:
            offspring["entry_rules"] = parent2.get("entry_rules", "")
            offspring["structural_edge"] = parent2.get("structural_edge", "")
            offspring["entry_parent"] = "parent2"

        # Exit logic: Weighted random
        if random.random() < p1_prob:
            offspring["exit_rules"] = parent1.get("exit_rules", "")
            offspring["exit_parent"] = "parent1"
        else:
            offspring["exit_rules"] = parent2.get("exit_rules", "")
            offspring["exit_parent"] = "parent2"

        # Combine filters
        filters1 = set(parent1.get("filters", []))
        filters2 = set(parent2.get("filters", []))
        offspring["filters"] = list(filters1.union(filters2))

        # Weighted average for risk parameters
        offspring["risk_per_trade"] = (
            parent1.get("risk_per_trade", 0.01) * p1_prob +
            parent2.get("risk_per_trade", 0.01) * (1 - p1_prob)
        )

        offspring["expected_wr"] = (
            parent1.get("expected_wr", 0.5) * p1_prob +
            parent2.get("expected_wr", 0.5) * (1 - p1_prob)
        )

        offspring["expected_rr"] = (
            parent1.get("expected_rr", 1.0) * p1_prob +
            parent2.get("expected_rr", 1.0) * (1 - p1_prob)
        )

        # Strategy name
        offspring["strategy_name"] = (
            f"Evolved: {parent1.get('strategy_name', 'P1')[:15]} + "
            f"{parent2.get('strategy_name', 'P2')[:15]}"
        )

        # Weighted combination of why_it_works
        if random.random() < p1_prob:
            offspring["why_it_works"] = parent1.get("why_it_works", "")
        else:
            offspring["why_it_works"] = parent2.get("why_it_works", "")

        # Combine weaknesses
        offspring["weaknesses"] = list(set(
            parent1.get("weaknesses", []) + parent2.get("weaknesses", [])
        ))

        return offspring

    # ==================== MUTATION ====================

    def _mutate(self, offspring: Dict) -> Tuple[Dict, List[str]]:
        """
        Apply random mutations to offspring.

        Mutations:
        - Risk per trade: ±15%
        - Expected WR: ±5 percentage points
        - Expected R:R: ±0.2
        - Filters: 10% chance to add/remove one

        Returns:
            (mutated_offspring, mutation_log)
        """
        mutation_log = []

        # Mutate risk per trade
        if random.random() < self.MUTATION_RATE:
            old_risk = offspring.get("risk_per_trade", 0.01)
            change = random.uniform(-self.MUTATION_STRENGTH, self.MUTATION_STRENGTH)
            new_risk = old_risk * (1 + change)
            new_risk = max(0.005, min(0.02, new_risk))  # Clamp to [0.5%, 2%]
            offspring["risk_per_trade"] = new_risk
            mutation_log.append(f"Risk: {old_risk:.3f} → {new_risk:.3f}")

        # Mutate expected win rate
        if random.random() < self.MUTATION_RATE:
            old_wr = offspring.get("expected_wr", 0.5)
            change = random.uniform(-0.05, 0.05)  # ±5 percentage points
            new_wr = old_wr + change
            new_wr = max(0.40, min(0.70, new_wr))  # Clamp to [40%, 70%]
            offspring["expected_wr"] = new_wr
            mutation_log.append(f"Win Rate: {old_wr:.1%} → {new_wr:.1%}")

        # Mutate expected R:R
        if random.random() < self.MUTATION_RATE:
            old_rr = offspring.get("expected_rr", 1.0)
            change = random.uniform(-0.2, 0.2)
            new_rr = old_rr + change
            new_rr = max(0.8, min(2.5, new_rr))  # Clamp to [0.8, 2.5]
            offspring["expected_rr"] = new_rr
            mutation_log.append(f"R:R: {old_rr:.2f} → {new_rr:.2f}")

        # Mutate filters (add or remove one)
        if random.random() < self.MUTATION_RATE:
            filters = offspring.get("filters", [])

            # Possible filters to add
            available_filters = [
                "spread_normal",
                "volume_confirmation",
                "no_major_news_1hr",
                "no_cb_meeting_24hrs",
                "regime_stable",
                "correlation_check",
                "time_of_day_filter",
                "max_position_limit"
            ]

            # 50% chance to add, 50% to remove
            if random.random() < 0.5 and len(filters) < 8:
                # Add a filter
                candidates = [f for f in available_filters if f not in filters]
                if candidates:
                    new_filter = random.choice(candidates)
                    filters.append(new_filter)
                    mutation_log.append(f"Added filter: {new_filter}")
            else:
                # Remove a filter
                if len(filters) > 2:  # Keep at least 2 filters
                    removed = random.choice(filters)
                    filters.remove(removed)
                    mutation_log.append(f"Removed filter: {removed}")

            offspring["filters"] = filters

        # Mutate entry/exit logic (small text modification)
        if random.random() < self.MUTATION_RATE * 0.5:  # Lower chance (5%)
            # Add a constraint to entry logic
            constraints = [
                " (only during high volume)",
                " (wait for confirmation candle)",
                " (check order book imbalance)",
                " (verify with higher timeframe)",
                " (ensure no divergence)"
            ]
            entry = offspring.get("entry_rules", "")
            if len(entry) < 300:  # Don't bloat too much
                mutation = random.choice(constraints)
                offspring["entry_rules"] = entry + mutation
                mutation_log.append(f"Modified entry: added '{mutation}'")

        if not mutation_log:
            mutation_log.append("No mutations applied")

        return offspring, mutation_log

    # ==================== VALIDATION ====================

    def validate_offspring(self, offspring: Dict) -> Tuple[bool, str]:
        """
        Validate that offspring is coherent and ready for tournament.

        Checks:
        - Has entry and exit logic
        - Has at least 2 filters
        - Risk parameters are reasonable
        - Has structural edge
        """
        # Check required fields
        required = ["entry_rules", "exit_rules", "structural_edge", "filters"]
        for field in required:
            if field not in offspring or not offspring[field]:
                return False, f"Missing {field}"

        # Check filters
        if len(offspring.get("filters", [])) < 2:
            return False, "Need at least 2 filters"

        # Check risk per trade
        risk = offspring.get("risk_per_trade", 0)
        if not (0.003 <= risk <= 0.025):
            return False, f"Risk per trade out of range: {risk:.3f}"

        # Check win rate
        wr = offspring.get("expected_wr", 0)
        if not (0.35 <= wr <= 0.75):
            return False, f"Expected win rate unrealistic: {wr:.1%}"

        # Check R:R
        rr = offspring.get("expected_rr", 0)
        if not (0.5 <= rr <= 3.0):
            return False, f"Expected R:R unrealistic: {rr:.2f}"

        return True, "Offspring valid"

    # ==================== STATISTICS ====================

    def get_breeding_stats(self) -> Dict:
        """Get breeding statistics."""
        if not self.breeding_history:
            return {
                "total_offspring": 0,
                "total_breedings": 0
            }

        return {
            "total_offspring": self.offspring_count,
            "total_breedings": len(self.breeding_history),
            "avg_parent1_fitness": sum(b["parent1_fitness"] for b in self.breeding_history) / len(self.breeding_history),
            "avg_parent2_fitness": sum(b["parent2_fitness"] for b in self.breeding_history) / len(self.breeding_history),
            "crossover_types": {
                ct: sum(1 for b in self.breeding_history if b["crossover_type"] == ct)
                for ct in self.CROSSOVER_TYPES
            }
        }


# Global singleton instance
_breeding_engine = None

def get_breeding_engine() -> BreedingEngine:
    """Get global BreedingEngine singleton."""
    global _breeding_engine
    if _breeding_engine is None:
        _breeding_engine = BreedingEngine()
    return _breeding_engine
