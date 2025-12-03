# HYDRA 3.0 - Hierarchical Multi-Agent Evolution System

# Lazy imports to avoid circular dependencies and missing optional deps
__all__ = [
    "MotherAI",
    "get_mother_ai",
    "EnginePortfolio",
    "TournamentManager",
    "get_tournament_manager",
    "KillCycle",
    "get_kill_cycle",
]


def __getattr__(name):
    """Lazy import to avoid loading heavy dependencies until needed."""
    if name in ("MotherAI", "get_mother_ai"):
        from .mother_ai import MotherAI, get_mother_ai
        return MotherAI if name == "MotherAI" else get_mother_ai

    if name in ("EnginePortfolio", "TournamentManager", "get_tournament_manager"):
        from .engine_portfolio import EnginePortfolio, TournamentManager, get_tournament_manager
        if name == "EnginePortfolio":
            return EnginePortfolio
        elif name == "TournamentManager":
            return TournamentManager
        else:
            return get_tournament_manager

    if name in ("KillCycle", "get_kill_cycle"):
        from .cycles.kill_cycle import KillCycle, get_kill_cycle
        return KillCycle if name == "KillCycle" else get_kill_cycle

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
