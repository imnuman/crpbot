"""
HYDRA 3.0 - Edge Graveyard

Archives failed/dead trading edges so they can be:
- Remembered (don't repeat mistakes)
- Analyzed (understand why they failed)
- Potentially resurrected if market conditions change

Tracks:
- When edge was born and died
- Why it failed (market regime change, overfitting, etc.)
- Performance metrics at time of death
- Resurrection candidates based on current conditions
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DeathCause(Enum):
    """Why an edge died."""
    OVERFITTING = "overfitting"           # Worked in-sample, failed out-of-sample
    REGIME_CHANGE = "regime_change"       # Market conditions changed
    ARBITRAGED_AWAY = "arbitraged_away"   # Too many players found same edge
    VALIDATOR_FAILED = "validator_failed" # Failed walk-forward or Monte Carlo
    LOSING_STREAK = "losing_streak"       # Consecutive losses exceeded threshold
    DRAWDOWN = "drawdown"                 # Max drawdown exceeded
    LOW_WIN_RATE = "low_win_rate"         # Win rate dropped too low
    ENGINE_KILLED = "engine_killed"       # Engine using this edge was killed
    MANUAL = "manual"                     # Manually retired
    UNKNOWN = "unknown"


class ResurrectionStatus(Enum):
    """Current resurrection potential."""
    BURIED = "buried"           # Not eligible for resurrection
    DORMANT = "dormant"         # Waiting for conditions to change
    CANDIDATE = "candidate"     # Conditions might be favorable
    RESURRECTED = "resurrected" # Brought back to life


@dataclass
class EdgeMetrics:
    """Performance metrics at time of death."""
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EdgeMetrics":
        return cls(**data)


@dataclass
class BuriedEdge:
    """A dead trading edge in the graveyard."""
    edge_id: str                              # Unique identifier
    engine_id: str                            # Which engine used it
    strategy_type: str                        # trend_follow, mean_revert, etc.
    description: str                          # Human-readable description

    # Lifecycle
    born_at: datetime                         # When edge was discovered
    died_at: datetime                         # When edge was buried
    lifespan_days: float                      # How long it lived

    # Death details
    death_cause: DeathCause                   # Why it died
    death_notes: str                          # Additional context

    # Performance at death
    metrics: EdgeMetrics                      # Performance when buried

    # Market conditions
    market_regime: str = "unknown"            # trending, ranging, volatile
    volatility_regime: str = "medium"         # low, medium, high
    symbols: list = field(default_factory=list)  # Which symbols it traded

    # Resurrection tracking
    resurrection_status: ResurrectionStatus = ResurrectionStatus.DORMANT
    resurrection_conditions: str = ""         # What would need to change
    resurrection_attempts: int = 0            # Times we tried to bring it back
    last_resurrection_check: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "edge_id": self.edge_id,
            "engine_id": self.engine_id,
            "strategy_type": self.strategy_type,
            "description": self.description,
            "born_at": self.born_at.isoformat(),
            "died_at": self.died_at.isoformat(),
            "lifespan_days": self.lifespan_days,
            "death_cause": self.death_cause.value,
            "death_notes": self.death_notes,
            "metrics": self.metrics.to_dict(),
            "market_regime": self.market_regime,
            "volatility_regime": self.volatility_regime,
            "symbols": self.symbols,
            "resurrection_status": self.resurrection_status.value,
            "resurrection_conditions": self.resurrection_conditions,
            "resurrection_attempts": self.resurrection_attempts,
            "last_resurrection_check": self.last_resurrection_check.isoformat() if self.last_resurrection_check else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BuriedEdge":
        """Create from dict."""
        return cls(
            edge_id=data["edge_id"],
            engine_id=data["engine_id"],
            strategy_type=data["strategy_type"],
            description=data["description"],
            born_at=datetime.fromisoformat(data["born_at"]),
            died_at=datetime.fromisoformat(data["died_at"]),
            lifespan_days=data["lifespan_days"],
            death_cause=DeathCause(data["death_cause"]),
            death_notes=data["death_notes"],
            metrics=EdgeMetrics.from_dict(data["metrics"]),
            market_regime=data.get("market_regime", "unknown"),
            volatility_regime=data.get("volatility_regime", "medium"),
            symbols=data.get("symbols", []),
            resurrection_status=ResurrectionStatus(data.get("resurrection_status", "dormant")),
            resurrection_conditions=data.get("resurrection_conditions", ""),
            resurrection_attempts=data.get("resurrection_attempts", 0),
            last_resurrection_check=datetime.fromisoformat(data["last_resurrection_check"]) if data.get("last_resurrection_check") else None,
        )


# Resurrection configuration
MIN_BURIAL_DAYS = 7           # Minimum days before resurrection check
RESURRECTION_COOLDOWN_DAYS = 3  # Days between resurrection attempts
MAX_RESURRECTION_ATTEMPTS = 3   # Max times to try resurrecting


class EdgeGraveyard:
    """
    Archives and manages dead trading edges.

    Features:
    - Bury edges with full context
    - Track resurrection candidates
    - Analyze failure patterns
    - Prevent repeating mistakes
    """

    _instance: Optional["EdgeGraveyard"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, data_dir: Optional[Path] = None):
        if self._initialized:
            return

        # Auto-detect data directory
        if data_dir is None:
            from .config import HYDRA_DATA_DIR
            data_dir = HYDRA_DATA_DIR

        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.graveyard_file = self.data_dir / "edge_graveyard.json"

        # In-memory graveyard
        self.buried_edges: dict[str, BuriedEdge] = {}

        # Load existing graveyard
        self._load_graveyard()

        self._initialized = True
        logger.info(f"EdgeGraveyard initialized with {len(self.buried_edges)} buried edges")

    def _load_graveyard(self):
        """Load graveyard from disk."""
        if self.graveyard_file.exists():
            try:
                with open(self.graveyard_file) as f:
                    data = json.load(f)
                    for edge_data in data.get("edges", []):
                        edge = BuriedEdge.from_dict(edge_data)
                        self.buried_edges[edge.edge_id] = edge
                logger.info(f"Loaded {len(self.buried_edges)} edges from graveyard")
            except Exception as e:
                logger.error(f"Failed to load graveyard: {e}")

    def _save_graveyard(self):
        """Save graveyard to disk."""
        try:
            data = {
                "last_updated": datetime.now().isoformat(),
                "total_edges": len(self.buried_edges),
                "edges": [edge.to_dict() for edge in self.buried_edges.values()]
            }
            with open(self.graveyard_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save graveyard: {e}")

    def bury_edge(
        self,
        engine_id: str,
        strategy_type: str,
        description: str,
        death_cause: DeathCause,
        metrics: EdgeMetrics,
        born_at: Optional[datetime] = None,
        death_notes: str = "",
        market_regime: str = "unknown",
        volatility_regime: str = "medium",
        symbols: Optional[list] = None,
        resurrection_conditions: str = ""
    ) -> BuriedEdge:
        """
        Bury a dead edge in the graveyard.

        Args:
            engine_id: Which engine used this edge
            strategy_type: Type of strategy
            description: Human-readable description
            death_cause: Why the edge died
            metrics: Performance at time of death
            born_at: When edge was discovered
            death_notes: Additional context
            market_regime: Market conditions
            volatility_regime: Volatility level
            symbols: Which symbols it traded
            resurrection_conditions: What would bring it back

        Returns:
            BuriedEdge object
        """
        now = datetime.now()
        if born_at is None:
            born_at = now - timedelta(days=7)  # Default 1 week lifespan

        edge_id = f"{engine_id}_{strategy_type}_{now.strftime('%Y%m%d_%H%M%S')}"
        lifespan = (now - born_at).total_seconds() / 86400  # Days

        edge = BuriedEdge(
            edge_id=edge_id,
            engine_id=engine_id,
            strategy_type=strategy_type,
            description=description,
            born_at=born_at,
            died_at=now,
            lifespan_days=lifespan,
            death_cause=death_cause,
            death_notes=death_notes,
            metrics=metrics,
            market_regime=market_regime,
            volatility_regime=volatility_regime,
            symbols=symbols or [],
            resurrection_status=ResurrectionStatus.DORMANT,
            resurrection_conditions=resurrection_conditions,
        )

        self.buried_edges[edge_id] = edge
        self._save_graveyard()

        logger.info(f"Buried edge: {edge_id} (cause: {death_cause.value})")
        return edge

    def check_resurrection_candidates(
        self,
        current_regime: str = "unknown",
        current_volatility: str = "medium"
    ) -> list[BuriedEdge]:
        """
        Check for edges that might be worth resurrecting.

        Args:
            current_regime: Current market regime
            current_volatility: Current volatility level

        Returns:
            List of resurrection candidates
        """
        candidates = []
        now = datetime.now()

        for edge in self.buried_edges.values():
            # Skip already resurrected or buried forever
            if edge.resurrection_status in [ResurrectionStatus.BURIED, ResurrectionStatus.RESURRECTED]:
                continue

            # Check minimum burial time
            days_buried = (now - edge.died_at).total_seconds() / 86400
            if days_buried < MIN_BURIAL_DAYS:
                continue

            # Check cooldown
            if edge.last_resurrection_check:
                days_since_check = (now - edge.last_resurrection_check).total_seconds() / 86400
                if days_since_check < RESURRECTION_COOLDOWN_DAYS:
                    continue

            # Check max attempts
            if edge.resurrection_attempts >= MAX_RESURRECTION_ATTEMPTS:
                edge.resurrection_status = ResurrectionStatus.BURIED
                continue

            # Check if conditions match
            is_candidate = False

            # Regime change edges might work again if regime matches original
            if edge.death_cause == DeathCause.REGIME_CHANGE:
                if current_regime == edge.market_regime:
                    is_candidate = True

            # Volatility-killed edges might work in different volatility
            elif edge.death_cause in [DeathCause.OVERFITTING, DeathCause.LOW_WIN_RATE]:
                if current_volatility != edge.volatility_regime:
                    is_candidate = True

            # General check - if conditions seem favorable
            if edge.metrics.win_rate > 0.45 and edge.metrics.profit_factor > 0.9:
                is_candidate = True

            if is_candidate:
                edge.resurrection_status = ResurrectionStatus.CANDIDATE
                edge.last_resurrection_check = now
                candidates.append(edge)

        if candidates:
            self._save_graveyard()
            logger.info(f"Found {len(candidates)} resurrection candidates")

        return candidates

    def resurrect_edge(self, edge_id: str) -> Optional[BuriedEdge]:
        """
        Mark an edge as resurrected and return it.

        Args:
            edge_id: ID of edge to resurrect

        Returns:
            Resurrected edge or None
        """
        if edge_id not in self.buried_edges:
            logger.warning(f"Edge not found: {edge_id}")
            return None

        edge = self.buried_edges[edge_id]
        edge.resurrection_status = ResurrectionStatus.RESURRECTED
        edge.resurrection_attempts += 1
        edge.last_resurrection_check = datetime.now()

        self._save_graveyard()
        logger.info(f"Resurrected edge: {edge_id} (attempt {edge.resurrection_attempts})")

        return edge

    def fail_resurrection(self, edge_id: str, notes: str = "") -> bool:
        """
        Mark a resurrection attempt as failed.

        Args:
            edge_id: ID of edge
            notes: Why it failed

        Returns:
            True if edge was updated
        """
        if edge_id not in self.buried_edges:
            return False

        edge = self.buried_edges[edge_id]
        edge.resurrection_attempts += 1
        edge.death_notes += f"\nResurrection failed: {notes}"

        if edge.resurrection_attempts >= MAX_RESURRECTION_ATTEMPTS:
            edge.resurrection_status = ResurrectionStatus.BURIED
            logger.info(f"Edge permanently buried after {MAX_RESURRECTION_ATTEMPTS} failed resurrections: {edge_id}")
        else:
            edge.resurrection_status = ResurrectionStatus.DORMANT

        self._save_graveyard()
        return True

    def get_failure_patterns(self) -> dict:
        """
        Analyze failure patterns across all buried edges.

        Returns:
            Analysis of common failure patterns
        """
        if not self.buried_edges:
            return {"total_edges": 0}

        # Count by death cause
        cause_counts = {}
        for edge in self.buried_edges.values():
            cause = edge.death_cause.value
            cause_counts[cause] = cause_counts.get(cause, 0) + 1

        # Count by strategy type
        strategy_counts = {}
        for edge in self.buried_edges.values():
            strategy_counts[edge.strategy_type] = strategy_counts.get(edge.strategy_type, 0) + 1

        # Count by engine
        engine_counts = {}
        for edge in self.buried_edges.values():
            engine_counts[edge.engine_id] = engine_counts.get(edge.engine_id, 0) + 1

        # Average lifespan
        lifespans = [e.lifespan_days for e in self.buried_edges.values()]
        avg_lifespan = sum(lifespans) / len(lifespans) if lifespans else 0

        # Most common failure cause
        most_common_cause = max(cause_counts.items(), key=lambda x: x[1])[0] if cause_counts else "none"

        # Resurrection stats
        resurrected = sum(1 for e in self.buried_edges.values()
                         if e.resurrection_status == ResurrectionStatus.RESURRECTED)

        return {
            "total_edges": len(self.buried_edges),
            "by_death_cause": cause_counts,
            "by_strategy": strategy_counts,
            "by_engine": engine_counts,
            "avg_lifespan_days": round(avg_lifespan, 2),
            "most_common_cause": most_common_cause,
            "total_resurrected": resurrected,
            "resurrection_rate": round(resurrected / len(self.buried_edges) * 100, 1) if self.buried_edges else 0,
        }

    def is_similar_to_dead_edge(
        self,
        strategy_type: str,
        market_regime: str,
        engine_id: Optional[str] = None
    ) -> Optional[BuriedEdge]:
        """
        Check if a new edge looks similar to a dead one.

        Args:
            strategy_type: Type of strategy
            market_regime: Current market regime
            engine_id: Optional engine filter

        Returns:
            Similar dead edge if found, None otherwise
        """
        for edge in self.buried_edges.values():
            # Skip resurrected edges
            if edge.resurrection_status == ResurrectionStatus.RESURRECTED:
                continue

            # Check strategy type match
            if edge.strategy_type != strategy_type:
                continue

            # Check regime match (avoid same mistakes)
            if edge.market_regime == market_regime:
                # This strategy failed in similar conditions before
                if engine_id is None or edge.engine_id == engine_id:
                    logger.warning(f"New edge similar to dead edge {edge.edge_id}")
                    return edge

        return None

    def get_recent_deaths(self, days: int = 7) -> list[BuriedEdge]:
        """Get edges that died recently."""
        cutoff = datetime.now() - timedelta(days=days)
        return [
            e for e in self.buried_edges.values()
            if e.died_at > cutoff
        ]

    def get_graveyard_summary(self) -> str:
        """Get a human-readable summary of the graveyard."""
        if not self.buried_edges:
            return "🪦 Edge Graveyard: Empty (no dead edges)"

        patterns = self.get_failure_patterns()
        recent = self.get_recent_deaths(7)
        candidates = [e for e in self.buried_edges.values()
                     if e.resurrection_status == ResurrectionStatus.CANDIDATE]

        lines = [
            f"🪦 Edge Graveyard Summary",
            f"   Total buried: {patterns['total_edges']}",
            f"   Died this week: {len(recent)}",
            f"   Resurrection candidates: {len(candidates)}",
            f"   Avg lifespan: {patterns['avg_lifespan_days']:.1f} days",
            f"   Most common cause: {patterns['most_common_cause']}",
        ]

        return "\n".join(lines)


# Singleton accessor
_graveyard_instance: Optional[EdgeGraveyard] = None


def get_edge_graveyard(data_dir: Optional[Path] = None) -> EdgeGraveyard:
    """Get or create the edge graveyard singleton."""
    global _graveyard_instance
    if _graveyard_instance is None:
        _graveyard_instance = EdgeGraveyard(data_dir)
    return _graveyard_instance


# Convenience function for quick burial
def bury_failed_edge(
    engine_id: str,
    strategy_type: str,
    description: str,
    death_cause: DeathCause,
    win_rate: float = 0.0,
    pnl: float = 0.0,
    trades: int = 0,
    death_notes: str = ""
) -> BuriedEdge:
    """
    Quick burial of a failed edge.

    Args:
        engine_id: Which engine
        strategy_type: Strategy type
        description: What was the edge
        death_cause: Why it died
        win_rate: Win rate at death
        pnl: P&L at death
        trades: Total trades
        death_notes: Additional notes

    Returns:
        BuriedEdge object
    """
    graveyard = get_edge_graveyard()

    metrics = EdgeMetrics(
        win_rate=win_rate,
        total_pnl=pnl,
        total_trades=trades,
        winning_trades=int(trades * win_rate),
        losing_trades=int(trades * (1 - win_rate)),
    )

    return graveyard.bury_edge(
        engine_id=engine_id,
        strategy_type=strategy_type,
        description=description,
        death_cause=death_cause,
        metrics=metrics,
        death_notes=death_notes,
    )
