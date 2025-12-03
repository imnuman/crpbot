"""
HYDRA 3.0 - No Edge Today Detector

Detects when market conditions are unfavorable for trading.

"No edge today" is triggered when:
1. Recent win rate drops below threshold
2. Market regime is highly uncertain
3. Consecutive losses exceed threshold
4. Validator shows no statistical edge
5. Volatility is too high or too low
6. All engines agree market is untradeable

The system prevents forcing trades in bad conditions,
protecting capital and waiting for better opportunities.

Phase 2, Week 2 - Step 20
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from loguru import logger
from enum import Enum
import json


class NoEdgeReason(Enum):
    """Reasons for declaring no edge."""
    LOW_WIN_RATE = "low_win_rate"           # Recent WR below threshold
    LOSING_STREAK = "losing_streak"          # Too many consecutive losses
    HIGH_VOLATILITY = "high_volatility"      # Vol too high for strategy
    LOW_VOLATILITY = "low_volatility"        # Vol too low, no movement
    UNCERTAIN_REGIME = "uncertain_regime"    # Can't determine market state
    VALIDATOR_FAILED = "validator_failed"    # MC/WF validator shows no edge
    ENGINE_CONSENSUS = "engine_consensus"    # All engines say no trade
    RECENT_DRAWDOWN = "recent_drawdown"      # Significant recent losses
    CHOPPY_MARKET = "choppy_market"          # No clear direction
    NEWS_EVENT = "news_event"                # Major news causing uncertainty


@dataclass
class NoEdgeEvent:
    """Record of a no-edge declaration."""
    timestamp: datetime
    engine: str  # "ALL" for system-wide, or specific engine
    reasons: List[NoEdgeReason]
    duration_hours: float  # How long to stay out
    metrics: Dict[str, float]  # Supporting metrics
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class EdgeStatus:
    """Current edge status for an engine or system."""
    has_edge: bool
    confidence: float  # 0-1 confidence in edge
    reasons_against: List[NoEdgeReason]
    recent_win_rate: float
    recent_pnl: float
    losing_streak: int
    recommendation: str


class NoEdgeDetector:
    """
    No Edge Today Detector for HYDRA 3.0.

    Monitors market conditions and engine performance to detect
    when trading edge is absent or diminished.
    """

    # Thresholds
    MIN_WIN_RATE = 0.40          # Below this = no edge
    MAX_LOSING_STREAK = 5        # 5 losses in a row = stop
    MIN_TRADES_FOR_ANALYSIS = 10 # Need 10 trades to analyze
    MAX_DRAWDOWN_PERCENT = 10    # -10% drawdown = pause

    # Volatility bounds (standard deviations)
    MIN_VOLATILITY = 0.5         # Too quiet
    MAX_VOLATILITY = 3.0         # Too crazy

    # Regime confidence threshold
    MIN_REGIME_CONFIDENCE = 0.5  # Below = uncertain

    # Default no-edge duration
    DEFAULT_PAUSE_HOURS = 4      # Stay out for 4 hours

    def __init__(self, data_dir: Optional[Path] = None):
        # Auto-detect data directory based on environment
        if data_dir is None:
            import os
            if os.path.exists("/root/crpbot"):
                data_dir = Path("/root/crpbot/data/hydra")
            else:
                data_dir = Path.home() / "crpbot" / "data" / "hydra"

        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Active no-edge events
        self.active_events: Dict[str, NoEdgeEvent] = {}  # engine -> event

        # Event history
        self.event_history: List[NoEdgeEvent] = []

        # Performance tracking (last N trades per engine)
        self.recent_trades: Dict[str, List[Dict]] = {
            "A": [], "B": [], "C": [], "D": []
        }

        # Persistence
        self.events_file = self.data_dir / "no_edge_events.jsonl"
        self.state_file = self.data_dir / "no_edge_state.json"

        # Load state
        self._load_state()

        logger.info("[NoEdgeDetector] Initialized")

    def check_edge(
        self,
        engine: str,
        recent_trades: List[Dict],
        regime: str,
        regime_confidence: float,
        volatility: Optional[float] = None,
        validator_result: Optional[Dict] = None
    ) -> EdgeStatus:
        """
        Check if an engine has tradeable edge.

        Args:
            engine: Engine name (A, B, C, D)
            recent_trades: List of recent trade results
            regime: Current market regime
            regime_confidence: Confidence in regime detection
            volatility: Current volatility measure
            validator_result: Result from Monte Carlo/Walk-Forward validator

        Returns:
            EdgeStatus with edge assessment
        """
        reasons = []
        metrics = {}

        # Update recent trades
        self.recent_trades[engine] = recent_trades[-20:]  # Keep last 20

        # Check if currently in no-edge state
        if self._is_in_no_edge_state(engine):
            event = self.active_events.get(engine)
            if event:
                return EdgeStatus(
                    has_edge=False,
                    confidence=0,
                    reasons_against=event.reasons,
                    recent_win_rate=0,
                    recent_pnl=0,
                    losing_streak=0,
                    recommendation=f"In no-edge state until {event.timestamp + timedelta(hours=event.duration_hours)}"
                )

        # Analyze recent performance
        if len(recent_trades) >= self.MIN_TRADES_FOR_ANALYSIS:
            # Win rate check
            wins = sum(1 for t in recent_trades if t.get("outcome") == "win")
            win_rate = wins / len(recent_trades)
            metrics["win_rate"] = win_rate

            if win_rate < self.MIN_WIN_RATE:
                reasons.append(NoEdgeReason.LOW_WIN_RATE)

            # P&L check
            total_pnl = sum(t.get("pnl_percent", 0) for t in recent_trades)
            metrics["total_pnl"] = total_pnl

            if total_pnl < -self.MAX_DRAWDOWN_PERCENT:
                reasons.append(NoEdgeReason.RECENT_DRAWDOWN)

            # Losing streak check
            losing_streak = self._count_losing_streak(recent_trades)
            metrics["losing_streak"] = losing_streak

            if losing_streak >= self.MAX_LOSING_STREAK:
                reasons.append(NoEdgeReason.LOSING_STREAK)
        else:
            win_rate = 0.5
            total_pnl = 0
            losing_streak = 0

        # Regime confidence check
        if regime_confidence < self.MIN_REGIME_CONFIDENCE:
            reasons.append(NoEdgeReason.UNCERTAIN_REGIME)
            metrics["regime_confidence"] = regime_confidence

        # Volatility check
        if volatility is not None:
            metrics["volatility"] = volatility
            if volatility < self.MIN_VOLATILITY:
                reasons.append(NoEdgeReason.LOW_VOLATILITY)
            elif volatility > self.MAX_VOLATILITY:
                reasons.append(NoEdgeReason.HIGH_VOLATILITY)

        # Validator result check
        if validator_result:
            if not validator_result.get("is_significant", True):
                reasons.append(NoEdgeReason.VALIDATOR_FAILED)
                metrics["p_value"] = validator_result.get("p_value", 1.0)

        # Choppy market detection
        if regime in ["CHOPPY", "RANGING", "SIDEWAYS"]:
            if regime_confidence > 0.7:  # High confidence in choppiness
                reasons.append(NoEdgeReason.CHOPPY_MARKET)

        # Determine edge status
        has_edge = len(reasons) == 0
        confidence = max(0, 1 - (len(reasons) * 0.2))  # Each reason reduces confidence

        # Generate recommendation
        if has_edge:
            recommendation = "Edge detected - proceed with trading"
        else:
            recommendation = self._generate_recommendation(reasons, metrics)

            # Record no-edge event if significant
            if len(reasons) >= 2 or NoEdgeReason.LOSING_STREAK in reasons:
                self._declare_no_edge(engine, reasons, metrics)

        return EdgeStatus(
            has_edge=has_edge,
            confidence=confidence,
            reasons_against=reasons,
            recent_win_rate=win_rate,
            recent_pnl=total_pnl,
            losing_streak=losing_streak,
            recommendation=recommendation
        )

    def check_system_edge(
        self,
        engine_statuses: Dict[str, EdgeStatus]
    ) -> EdgeStatus:
        """
        Check if the entire system has edge (consensus across engines).

        Args:
            engine_statuses: EdgeStatus for each engine

        Returns:
            System-wide EdgeStatus
        """
        if not engine_statuses:
            return EdgeStatus(
                has_edge=True,
                confidence=0.5,
                reasons_against=[],
                recent_win_rate=0.5,
                recent_pnl=0,
                losing_streak=0,
                recommendation="No engine data available"
            )

        # Count engines with/without edge
        engines_with_edge = sum(1 for s in engine_statuses.values() if s.has_edge)
        total_engines = len(engine_statuses)

        # Aggregate metrics
        avg_win_rate = sum(s.recent_win_rate for s in engine_statuses.values()) / total_engines
        total_pnl = sum(s.recent_pnl for s in engine_statuses.values())
        max_losing_streak = max(s.losing_streak for s in engine_statuses.values())

        # Aggregate reasons
        all_reasons = []
        for status in engine_statuses.values():
            all_reasons.extend(status.reasons_against)

        # System has edge if majority of engines have edge
        has_edge = engines_with_edge >= (total_engines / 2)

        # If no engine has edge, declare system-wide no edge
        if engines_with_edge == 0:
            all_reasons.append(NoEdgeReason.ENGINE_CONSENSUS)
            self._declare_no_edge("ALL", [NoEdgeReason.ENGINE_CONSENSUS], {
                "engines_without_edge": total_engines
            })

        confidence = engines_with_edge / total_engines

        recommendation = (
            f"{engines_with_edge}/{total_engines} engines have edge. "
            f"{'Proceed with caution.' if has_edge else 'Consider pausing all trading.'}"
        )

        return EdgeStatus(
            has_edge=has_edge,
            confidence=confidence,
            reasons_against=list(set(all_reasons)),
            recent_win_rate=avg_win_rate,
            recent_pnl=total_pnl,
            losing_streak=max_losing_streak,
            recommendation=recommendation
        )

    def declare_no_edge(
        self,
        engine: str,
        reason: NoEdgeReason,
        duration_hours: float = None,
        metrics: Dict = None
    ) -> NoEdgeEvent:
        """
        Manually declare no edge for an engine.

        Args:
            engine: Engine name or "ALL"
            reason: Primary reason
            duration_hours: How long to stay out
            metrics: Supporting metrics

        Returns:
            NoEdgeEvent
        """
        return self._declare_no_edge(
            engine,
            [reason],
            metrics or {},
            duration_hours or self.DEFAULT_PAUSE_HOURS
        )

    def _declare_no_edge(
        self,
        engine: str,
        reasons: List[NoEdgeReason],
        metrics: Dict,
        duration_hours: float = None
    ) -> NoEdgeEvent:
        """Internal method to declare no edge."""
        duration = duration_hours or self.DEFAULT_PAUSE_HOURS

        event = NoEdgeEvent(
            timestamp=datetime.now(timezone.utc),
            engine=engine,
            reasons=reasons,
            duration_hours=duration,
            metrics=metrics
        )

        # Store active event
        self.active_events[engine] = event
        self.event_history.append(event)

        # Save to disk
        self._save_event(event)

        # Log
        reason_strs = [r.value for r in reasons]
        logger.warning(
            f"[NoEdgeDetector] NO EDGE DECLARED for Engine {engine}: "
            f"{', '.join(reason_strs)}. Pausing for {duration}h"
        )

        return event

    def resolve_no_edge(self, engine: str) -> bool:
        """
        Resolve a no-edge state, allowing trading to resume.

        Args:
            engine: Engine name or "ALL"

        Returns:
            True if resolved, False if no active event
        """
        if engine not in self.active_events:
            return False

        event = self.active_events[engine]
        event.resolved = True
        event.resolved_at = datetime.now(timezone.utc)

        del self.active_events[engine]

        logger.info(f"[NoEdgeDetector] No-edge resolved for Engine {engine}")

        self._save_state()
        return True

    def _is_in_no_edge_state(self, engine: str) -> bool:
        """Check if engine is currently in no-edge state."""
        if engine not in self.active_events:
            # Also check system-wide
            if "ALL" in self.active_events:
                event = self.active_events["ALL"]
            else:
                return False
        else:
            event = self.active_events[engine]

        # Check if duration has expired
        now = datetime.now(timezone.utc)
        expiry = event.timestamp + timedelta(hours=event.duration_hours)

        if now >= expiry:
            # Auto-resolve expired events
            self.resolve_no_edge(engine)
            return False

        return True

    def _count_losing_streak(self, trades: List[Dict]) -> int:
        """Count consecutive losses from most recent trades."""
        streak = 0
        for trade in reversed(trades):
            if trade.get("outcome") == "loss":
                streak += 1
            else:
                break
        return streak

    def _generate_recommendation(
        self,
        reasons: List[NoEdgeReason],
        metrics: Dict
    ) -> str:
        """Generate actionable recommendation based on no-edge reasons."""
        recommendations = []

        for reason in reasons:
            if reason == NoEdgeReason.LOW_WIN_RATE:
                wr = metrics.get("win_rate", 0)
                recommendations.append(
                    f"Win rate ({wr:.1%}) below {self.MIN_WIN_RATE:.0%}. "
                    "Wait for conditions to improve."
                )

            elif reason == NoEdgeReason.LOSING_STREAK:
                streak = metrics.get("losing_streak", 0)
                recommendations.append(
                    f"{streak} consecutive losses. Take a break to avoid tilt."
                )

            elif reason == NoEdgeReason.HIGH_VOLATILITY:
                vol = metrics.get("volatility", 0)
                recommendations.append(
                    f"Volatility ({vol:.1f}) too high. Reduce position size or wait."
                )

            elif reason == NoEdgeReason.LOW_VOLATILITY:
                recommendations.append(
                    "Low volatility - no movement to capture. Wait for breakout."
                )

            elif reason == NoEdgeReason.UNCERTAIN_REGIME:
                recommendations.append(
                    "Market regime unclear. Wait for clearer direction."
                )

            elif reason == NoEdgeReason.VALIDATOR_FAILED:
                recommendations.append(
                    "Statistical validation failed - edge not significant."
                )

            elif reason == NoEdgeReason.RECENT_DRAWDOWN:
                pnl = metrics.get("total_pnl", 0)
                recommendations.append(
                    f"Recent drawdown ({pnl:.1f}%) exceeds limit. Reduce exposure."
                )

            elif reason == NoEdgeReason.CHOPPY_MARKET:
                recommendations.append(
                    "Choppy/ranging market. Trend strategies won't work."
                )

        return " | ".join(recommendations) if recommendations else "Wait for better conditions."

    def get_active_events(self) -> Dict[str, NoEdgeEvent]:
        """Get all active no-edge events."""
        # Clean up expired events first
        now = datetime.now(timezone.utc)
        expired = []

        for engine, event in self.active_events.items():
            expiry = event.timestamp + timedelta(hours=event.duration_hours)
            if now >= expiry:
                expired.append(engine)

        for engine in expired:
            self.resolve_no_edge(engine)

        return self.active_events.copy()

    def is_trading_allowed(self, engine: str) -> Tuple[bool, Optional[str]]:
        """
        Check if trading is allowed for an engine.

        Returns:
            Tuple of (allowed, reason_if_not_allowed)
        """
        # Check engine-specific
        if engine in self.active_events:
            event = self.active_events[engine]
            now = datetime.now(timezone.utc)
            expiry = event.timestamp + timedelta(hours=event.duration_hours)

            if now < expiry:
                remaining = (expiry - now).total_seconds() / 3600
                return False, f"No-edge active ({remaining:.1f}h remaining)"

        # Check system-wide
        if "ALL" in self.active_events:
            event = self.active_events["ALL"]
            now = datetime.now(timezone.utc)
            expiry = event.timestamp + timedelta(hours=event.duration_hours)

            if now < expiry:
                remaining = (expiry - now).total_seconds() / 3600
                return False, f"System-wide no-edge ({remaining:.1f}h remaining)"

        return True, None

    def get_summary(self) -> Dict[str, Any]:
        """Get detector summary."""
        active = self.get_active_events()

        return {
            "active_events": len(active),
            "engines_paused": list(active.keys()),
            "total_events": len(self.event_history),
            "thresholds": {
                "min_win_rate": self.MIN_WIN_RATE,
                "max_losing_streak": self.MAX_LOSING_STREAK,
                "max_drawdown": self.MAX_DRAWDOWN_PERCENT
            }
        }

    def format_status(self) -> str:
        """Format current status for dashboard."""
        lines = ["NO EDGE DETECTOR STATUS", "=" * 40]

        active = self.get_active_events()

        if not active:
            lines.append("All engines clear to trade")
        else:
            for engine, event in active.items():
                now = datetime.now(timezone.utc)
                expiry = event.timestamp + timedelta(hours=event.duration_hours)
                remaining = max(0, (expiry - now).total_seconds() / 3600)

                reasons = ", ".join(r.value for r in event.reasons)
                lines.append(f"Engine {engine}: PAUSED ({remaining:.1f}h)")
                lines.append(f"  Reason: {reasons}")

        lines.append("=" * 40)
        return "\n".join(lines)

    def _save_event(self, event: NoEdgeEvent):
        """Save event to disk."""
        try:
            entry = {
                "timestamp": event.timestamp.isoformat(),
                "engine": event.engine,
                "reasons": [r.value for r in event.reasons],
                "duration_hours": event.duration_hours,
                "metrics": event.metrics,
                "resolved": event.resolved,
                "resolved_at": event.resolved_at.isoformat() if event.resolved_at else None
            }

            with open(self.events_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')

        except Exception as e:
            logger.warning(f"[NoEdgeDetector] Failed to save event: {e}")

    def _save_state(self):
        """Save current state to disk."""
        try:
            state = {
                "active_events": {
                    engine: {
                        "timestamp": event.timestamp.isoformat(),
                        "reasons": [r.value for r in event.reasons],
                        "duration_hours": event.duration_hours
                    }
                    for engine, event in self.active_events.items()
                }
            }

            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            logger.warning(f"[NoEdgeDetector] Failed to save state: {e}")

    def _load_state(self):
        """Load state from disk."""
        try:
            if self.state_file.exists():
                with open(self.state_file) as f:
                    state = json.load(f)

                for engine, data in state.get("active_events", {}).items():
                    event = NoEdgeEvent(
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        engine=engine,
                        reasons=[NoEdgeReason(r) for r in data["reasons"]],
                        duration_hours=data["duration_hours"],
                        metrics={}
                    )
                    self.active_events[engine] = event

        except Exception as e:
            logger.warning(f"[NoEdgeDetector] Failed to load state: {e}")


# ==================== SINGLETON PATTERN ====================

_no_edge_detector: Optional[NoEdgeDetector] = None

def get_no_edge_detector() -> NoEdgeDetector:
    """Get singleton instance of NoEdgeDetector."""
    global _no_edge_detector
    if _no_edge_detector is None:
        _no_edge_detector = NoEdgeDetector()
    return _no_edge_detector
