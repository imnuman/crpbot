"""
HYDRA 3.0 - Evolution Cycles

Implements tournament evolution mechanics:
- Kill Cycle (24hr): Eliminate weakest, force new strategy
- Breeding Cycle (4 days): Combine winners' DNA into offspring
- Knowledge Transfer: Winner insights → all losers (every cycle)
- Stats Injection: Tournament standings in every prompt
- Weight Adjustment: Dynamic weight allocation (24hr)
"""

from .kill_cycle import KillCycle, get_kill_cycle
from .breeding_cycle import BreedingCycle, get_breeding_cycle
from .knowledge_transfer import KnowledgeTransfer, get_knowledge_transfer
from .stats_injector import StatsInjector, get_stats_injector
from .weight_adjuster import WeightAdjuster, get_weight_adjuster, WeightStrategy

__all__ = [
    "KillCycle",
    "get_kill_cycle",
    "BreedingCycle",
    "get_breeding_cycle",
    "KnowledgeTransfer",
    "get_knowledge_transfer",
    "StatsInjector",
    "get_stats_injector",
    "WeightAdjuster",
    "get_weight_adjuster",
    "WeightStrategy",
]
