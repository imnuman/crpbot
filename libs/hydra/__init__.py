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
    "EdgeGraveyard",
    "get_edge_graveyard",
    "bury_failed_edge",
    "DeathCause",
    "APICache",
    "get_api_cache",
    "CacheType",
    "cached",
    "cached_async",
    # Engine Specialization
    "Specialty",
    "SpecialtyConfig",
    "SpecialtyValidator",
    "get_specialty_validator",
    "validate_engine_trade",
    "get_engine_specialty",
    "ENGINE_SPECIALTIES",
    # Engine D Special Rules
    "EngineDController",
    "get_engine_d_controller",
    "check_engine_d_activation",
    "record_engine_d_activation",
    "record_engine_d_trade",
    # Trade Validator (confidence + correlation)
    "TradeValidator",
    "get_trade_validator",
    "TradeProposal",
    "ValidationResult",
    "validate_trade",
    # Improvement Tracker (MOD 11)
    "ImprovementTracker",
    "get_improvement_tracker",
    "record_engine_daily_stats",
    "get_engine_improvement",
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

    if name in ("EdgeGraveyard", "get_edge_graveyard", "bury_failed_edge", "DeathCause"):
        from .edge_graveyard import EdgeGraveyard, get_edge_graveyard, bury_failed_edge, DeathCause
        return {"EdgeGraveyard": EdgeGraveyard, "get_edge_graveyard": get_edge_graveyard,
                "bury_failed_edge": bury_failed_edge, "DeathCause": DeathCause}[name]

    if name in ("APICache", "get_api_cache", "CacheType", "cached", "cached_async"):
        from .api_cache import APICache, get_api_cache, CacheType, cached, cached_async
        return {"APICache": APICache, "get_api_cache": get_api_cache, "CacheType": CacheType,
                "cached": cached, "cached_async": cached_async}[name]

    if name in ("Specialty", "SpecialtyConfig", "SpecialtyValidator", "get_specialty_validator",
                "validate_engine_trade", "get_engine_specialty", "ENGINE_SPECIALTIES"):
        from .engine_specialization import (
            Specialty, SpecialtyConfig, SpecialtyValidator, get_specialty_validator,
            validate_engine_trade, get_engine_specialty, ENGINE_SPECIALTIES
        )
        return {"Specialty": Specialty, "SpecialtyConfig": SpecialtyConfig,
                "SpecialtyValidator": SpecialtyValidator, "get_specialty_validator": get_specialty_validator,
                "validate_engine_trade": validate_engine_trade, "get_engine_specialty": get_engine_specialty,
                "ENGINE_SPECIALTIES": ENGINE_SPECIALTIES}[name]

    if name in ("EngineDController", "get_engine_d_controller", "check_engine_d_activation",
                "record_engine_d_activation", "record_engine_d_trade"):
        from .engine_d_rules import (
            EngineDController, get_engine_d_controller, check_engine_d_activation,
            record_engine_d_activation, record_engine_d_trade
        )
        return {"EngineDController": EngineDController, "get_engine_d_controller": get_engine_d_controller,
                "check_engine_d_activation": check_engine_d_activation,
                "record_engine_d_activation": record_engine_d_activation,
                "record_engine_d_trade": record_engine_d_trade}[name]

    if name in ("TradeValidator", "get_trade_validator", "TradeProposal", "ValidationResult", "validate_trade"):
        from .trade_validator import (
            TradeValidator, get_trade_validator, TradeProposal, ValidationResult, validate_trade
        )
        return {"TradeValidator": TradeValidator, "get_trade_validator": get_trade_validator,
                "TradeProposal": TradeProposal, "ValidationResult": ValidationResult,
                "validate_trade": validate_trade}[name]

    if name in ("ImprovementTracker", "get_improvement_tracker", "record_engine_daily_stats", "get_engine_improvement"):
        from .improvement_tracker import (
            ImprovementTracker, get_improvement_tracker, record_engine_daily_stats, get_engine_improvement
        )
        return {"ImprovementTracker": ImprovementTracker, "get_improvement_tracker": get_improvement_tracker,
                "record_engine_daily_stats": record_engine_daily_stats,
                "get_engine_improvement": get_engine_improvement}[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
