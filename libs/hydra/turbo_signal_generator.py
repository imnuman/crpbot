"""
HYDRA 4.0 - Turbo Signal Generator Bridge

This module bridges TurboGenerator and TurboTournament to the main runtime.
Orchestrates batch strategy generation and ranking for signal production.

Flow:
1. Check which engine specialties are triggered
2. Generate 250 strategies per triggered specialty
3. Quick rank all strategies via TurboTournament
4. Select diverse top 4 for voting
5. Return for engine consensus
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from loguru import logger

from libs.hydra.turbo_generator import get_turbo_generator, GeneratedStrategy, StrategyType
from libs.hydra.turbo_tournament import get_turbo_tournament, BacktestResult


# Map engine names to their specialties
ENGINE_SPECIALTY_MAP = {
    "A": StrategyType.LIQUIDATION_CASCADE,
    "B": StrategyType.FUNDING_EXTREME,
    "C": StrategyType.ORDERBOOK_IMBALANCE,
    "D": StrategyType.REGIME_TRANSITION,
}


@dataclass
class TurboSignalResult:
    """Result from turbo batch generation."""
    strategies: List[Dict[str, Any]] = field(default_factory=list)
    total_generated: int = 0
    total_survivors: int = 0
    generation_cost_usd: float = 0.0
    ranking_time_seconds: float = 0.0
    specialty_breakdown: Dict[str, int] = field(default_factory=dict)


class TurboSignalGenerator:
    """
    Bridge between TurboGenerator/TurboTournament and runtime.

    Orchestrates:
    - Batch strategy generation (250 per specialty)
    - Quick ranking via tournament
    - Diverse top-4 selection for voting
    """

    STRATEGIES_PER_SPECIALTY = 250
    MIN_RANK_SCORE = 30.0

    def __init__(self, use_mock: bool = True):
        """Initialize the turbo signal generator.

        Args:
            use_mock: If True, use mock generation instead of real API
        """
        self.generator = get_turbo_generator()
        self.tournament = get_turbo_tournament()
        self.use_mock = use_mock

        logger.info(f"[TurboSignalGenerator] Initialized (mock={use_mock})")

    def generate_and_rank(
        self,
        asset: str,
        regime: str,
        market_data: Dict[str, Any],
        specialty_triggers: Dict[str, bool]
    ) -> TurboSignalResult:
        """
        Generate and rank strategies for all triggered specialties.

        Args:
            asset: Trading pair (e.g., "BTC-USD")
            regime: Market regime (e.g., "trending_up")
            market_data: Current market data dict
            specialty_triggers: Dict of engine_name -> is_triggered

        Returns:
            TurboSignalResult with top strategies for voting
        """
        import time
        start_time = time.time()

        # Determine asset class from asset name
        asset_class = self._get_asset_class(asset)

        # Generate batch for each TRIGGERED specialty
        all_strategies = []
        specialty_breakdown = {}
        total_cost = 0.0

        for engine_name, specialty in ENGINE_SPECIALTY_MAP.items():
            if specialty_triggers.get(engine_name, False):
                logger.info(f"[TurboSignalGenerator] Generating {self.STRATEGIES_PER_SPECIALTY} {specialty.value} strategies")

                batch = self.generator.generate_batch(
                    specialty=specialty,
                    regime=regime,
                    asset_class=asset_class,
                    count=self.STRATEGIES_PER_SPECIALTY,
                    use_mock=self.use_mock
                )

                # Tag strategies with source engine
                for strat in batch:
                    strat.gladiator = engine_name  # type: ignore

                all_strategies.extend(batch)
                specialty_breakdown[specialty.value] = len(batch)

                # Estimate cost (if not mock)
                if not self.use_mock:
                    total_cost += 1.50  # ~$1.50 per 250 strategies
            else:
                specialty_breakdown[specialty.value] = 0
                logger.debug(f"[TurboSignalGenerator] Engine {engine_name} specialty not triggered, skipping")

        if not all_strategies:
            logger.info("[TurboSignalGenerator] No specialties triggered, returning empty result")
            return TurboSignalResult(
                strategies=[],
                total_generated=0,
                total_survivors=0,
                generation_cost_usd=0.0,
                ranking_time_seconds=0.0,
                specialty_breakdown=specialty_breakdown
            )

        logger.info(f"[TurboSignalGenerator] Generated {len(all_strategies)} total strategies, now ranking...")

        # Rank all strategies via tournament
        ranked_results = self.tournament.rank_batch(
            strategies=all_strategies,
            max_workers=4
        )

        # Filter by minimum rank score
        survivors = [
            (strat, result) for strat, result in ranked_results
            if result.rank_score >= self.MIN_RANK_SCORE
        ]

        logger.info(f"[TurboSignalGenerator] {len(survivors)}/{len(all_strategies)} strategies survived ranking (min score: {self.MIN_RANK_SCORE})")

        # Select top 4 (one per specialty if possible)
        top_4 = self._select_diverse_top_4(survivors)

        # Convert to dict format for voting
        strategies_for_voting = []
        for strat, result in top_4:
            strategies_for_voting.append({
                "strategy_id": strat.strategy_id,
                "name": strat.name,
                "gladiator": getattr(strat, 'gladiator', 'X'),
                "specialty": strat.specialty.value,
                "confidence": min(0.95, result.rank_score / 100),  # Convert to 0-1 scale
                "entry_rules": strat.entry_rules,
                "exit_rules": strat.exit_rules,
                "stop_loss_pct": strat.stop_loss_atr_mult * 0.01,  # Approximate
                "take_profit_pct": strat.take_profit_atr_mult * 0.01,
                "risk_per_trade": strat.risk_per_trade,
                "backtest_wr": result.win_rate,
                "backtest_sharpe": result.sharpe_ratio,
                "backtest_trades": result.total_trades,
                "rank_score": result.rank_score,
                "source": "turbo_batch"
            })

        elapsed = time.time() - start_time

        return TurboSignalResult(
            strategies=strategies_for_voting,
            total_generated=len(all_strategies),
            total_survivors=len(survivors),
            generation_cost_usd=total_cost,
            ranking_time_seconds=elapsed,
            specialty_breakdown=specialty_breakdown
        )

    def _get_asset_class(self, asset: str) -> str:
        """Determine asset class from asset name."""
        asset_upper = asset.upper()
        if any(x in asset_upper for x in ["BTC", "ETH", "SOL"]):
            return "major_crypto"
        elif any(x in asset_upper for x in ["DOGE", "SHIB", "PEPE"]):
            return "meme"
        elif any(x in asset_upper for x in ["LINK", "UNI", "AAVE"]):
            return "defi"
        else:
            return "altcoin"

    def _select_diverse_top_4(
        self,
        ranked: List[tuple]
    ) -> List[tuple]:
        """
        Select top 4 strategies with diversity.

        Tries to get one from each specialty, falls back to pure ranking.
        """
        if len(ranked) <= 4:
            return ranked

        selected = []
        specialties_covered = set()

        # First pass: get top from each specialty
        for strat, result in ranked:
            specialty = strat.specialty.value
            if specialty not in specialties_covered and len(selected) < 4:
                selected.append((strat, result))
                specialties_covered.add(specialty)

        # Second pass: fill remaining slots with top ranked
        for strat, result in ranked:
            if len(selected) >= 4:
                break
            if (strat, result) not in selected:
                selected.append((strat, result))

        return selected[:4]


# Singleton instance
_turbo_signal_generator: Optional[TurboSignalGenerator] = None


def get_turbo_signal_generator(use_mock: bool = True) -> TurboSignalGenerator:
    """Get the singleton turbo signal generator instance."""
    global _turbo_signal_generator
    if _turbo_signal_generator is None:
        _turbo_signal_generator = TurboSignalGenerator(use_mock=use_mock)
    return _turbo_signal_generator
