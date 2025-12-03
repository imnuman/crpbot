"""
HYDRA 3.0 - Evolution Cycles

Implements tournament evolution mechanics:
- Kill Cycle (24hr): Eliminate weakest, force new strategy
- Breeding Cycle (4 days): Combine winners' DNA into offspring
- Knowledge Transfer: Winner insights → all losers
"""

from .kill_cycle import KillCycle, get_kill_cycle
from .breeding_cycle import BreedingCycle, get_breeding_cycle

__all__ = [
    "KillCycle",
    "get_kill_cycle",
    "BreedingCycle",
    "get_breeding_cycle",
]
