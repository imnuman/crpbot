"""
HYDRA 3.0 - Evolution Cycles

Implements tournament evolution mechanics:
- Kill Cycle (24hr): Eliminate weakest, force new strategy
- Breeding Cycle (4 days): Winner teaches losers
- Knowledge Transfer: Winner insights → all losers
"""

from .kill_cycle import KillCycle, get_kill_cycle

__all__ = [
    "KillCycle",
    "get_kill_cycle",
]
