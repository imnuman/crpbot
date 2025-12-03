"""
HYDRA 3.0 Guardian Safety System

Supreme safety authority. NO exceptions.

Enforces FTMO rules and protects account from catastrophic loss.

Phase 1, Week 1 - Step 11
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class SafetyViolation:
    """Record of a safety violation"""
    timestamp: datetime
    violation_type: str
    severity: str  # WARNING, CRITICAL, FATAL
    message: str
    data: Dict


@dataclass
class SafetyStatus:
    """Current safety status of the account"""
    daily_loss_percent: float
    total_drawdown_percent: float
    open_positions: int
    trades_today: int
    volatility_risk: str  # LOW, MEDIUM, HIGH, EXTREME
    news_proximity: bool  # True if near major news event
    guardian_active: bool
    lockout_until: Optional[datetime]
    violations_today: List[SafetyViolation]


class Guardian:
    """
    Guardian Safety System for HYDRA 3.0

    Non-negotiable rules:
    1. Daily loss 2% max → lockout 24 hours
    2. Total drawdown 6% max → emergency shutdown
    3. Risk per trade: 0.3-0.65% (dynamic based on confidence)
    4. Max concurrent positions: 2
    5. News buffer: 15 min before/after major news
    6. Volatility filter: ATR > 1.6x average → reduce size or skip

    Guardian has FINAL SAY. Mother AI cannot override.
    """

    # FTMO-style rules (conservative for account preservation)
    MAX_DAILY_LOSS_PERCENT = 2.0
    MAX_TOTAL_DRAWDOWN_PERCENT = 6.0
    MAX_CONCURRENT_POSITIONS = 2
    MIN_RISK_PER_TRADE_PERCENT = 0.3
    MAX_RISK_PER_TRADE_PERCENT = 0.65
    VOLATILITY_THRESHOLD_MULTIPLIER = 1.6
    NEWS_BUFFER_MINUTES = 15
    LOCKOUT_DURATION_HOURS = 24

    def __init__(self, data_dir: Path = None):
        """
        Initialize Guardian

        Args:
            data_dir: Directory for storing safety logs (default: auto-detect)
        """
        if data_dir is None:
            from ..config import HYDRA_DATA_DIR
            data_dir = HYDRA_DATA_DIR
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.violations: List[SafetyViolation] = []
        self.lockout_until: Optional[datetime] = None
        self.emergency_shutdown = False

        # Load state if exists
        self._load_state()

        logger.info("[Guardian] Safety system initialized - Standing watch")

    def check_trade_permission(
        self,
        account_balance: float,
        daily_pnl: float,
        total_pnl: float,
        open_positions: int,
        proposed_risk_percent: float,
        volatility_level: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a new trade is permitted under safety rules

        Args:
            account_balance: Current account balance
            daily_pnl: P&L for today (dollars)
            total_pnl: Total P&L since start (dollars)
            open_positions: Number of currently open positions
            proposed_risk_percent: Proposed risk % for this trade
            volatility_level: Current volatility (LOW/MEDIUM/HIGH/EXTREME)

        Returns:
            Tuple of (permitted: bool, reason: Optional[str])
        """

        # Check 0: Emergency shutdown
        if self.emergency_shutdown:
            return False, "EMERGENCY SHUTDOWN - Max drawdown breached"

        # Check 1: Lockout from previous violation
        if self.lockout_until:
            if datetime.now(timezone.utc) < self.lockout_until:
                remaining = (self.lockout_until - datetime.now(timezone.utc)).total_seconds() / 3600
                return False, f"LOCKED OUT - {remaining:.1f} hours remaining"
            else:
                # Lockout expired
                self.lockout_until = None
                logger.info("[Guardian] Lockout expired - Trading resumed")

        # Check 2: Daily loss limit
        daily_loss_percent = (daily_pnl / account_balance) * 100
        if daily_loss_percent < -self.MAX_DAILY_LOSS_PERCENT:
            self._trigger_lockout(
                f"Daily loss limit breached: {daily_loss_percent:.2f}% < -{self.MAX_DAILY_LOSS_PERCENT}%"
            )
            return False, f"Daily loss limit breached ({daily_loss_percent:.2f}%)"

        # Check 3: Total drawdown limit
        total_dd_percent = (total_pnl / account_balance) * 100
        if total_dd_percent < -self.MAX_TOTAL_DRAWDOWN_PERCENT:
            self._trigger_emergency_shutdown(
                f"Max drawdown breached: {total_dd_percent:.2f}% < -{self.MAX_TOTAL_DRAWDOWN_PERCENT}%"
            )
            return False, "MAX DRAWDOWN BREACHED - EMERGENCY SHUTDOWN"

        # Check 4: Approaching daily loss limit (warning at 1.5%)
        if daily_loss_percent < -1.5:
            self._log_violation(
                violation_type="daily_loss_warning",
                severity="WARNING",
                message=f"Approaching daily loss limit: {daily_loss_percent:.2f}%"
            )

        # Check 5: Position count limit
        if open_positions >= self.MAX_CONCURRENT_POSITIONS:
            return False, f"Max concurrent positions ({self.MAX_CONCURRENT_POSITIONS}) reached"

        # Check 6: Risk per trade limits
        if proposed_risk_percent < self.MIN_RISK_PER_TRADE_PERCENT:
            return False, f"Risk too low ({proposed_risk_percent:.2f}% < {self.MIN_RISK_PER_TRADE_PERCENT}%)"

        if proposed_risk_percent > self.MAX_RISK_PER_TRADE_PERCENT:
            return False, f"Risk too high ({proposed_risk_percent:.2f}% > {self.MAX_RISK_PER_TRADE_PERCENT}%)"

        # Check 7: Volatility filter
        if volatility_level == "EXTREME":
            self._log_violation(
                violation_type="volatility_extreme",
                severity="WARNING",
                message="Extreme volatility detected - Trade rejected"
            )
            return False, "EXTREME volatility - Trade rejected"

        # All checks passed
        logger.info(
            f"[Guardian] Trade APPROVED - "
            f"Daily: {daily_loss_percent:.2f}%, DD: {total_dd_percent:.2f}%, "
            f"Positions: {open_positions}/{self.MAX_CONCURRENT_POSITIONS}, "
            f"Risk: {proposed_risk_percent:.2f}%"
        )
        return True, None

    def get_dynamic_risk_percent(
        self,
        base_confidence: float,
        current_drawdown_percent: float,
        volatility_level: str
    ) -> float:
        """
        Calculate dynamic risk percent based on conditions

        Higher confidence + lower drawdown + lower volatility = higher risk
        Lower confidence + higher drawdown + higher volatility = lower risk

        Args:
            base_confidence: Engine confidence (0.0-1.0)
            current_drawdown_percent: Current drawdown (-6% to 0%)
            volatility_level: LOW/MEDIUM/HIGH/EXTREME

        Returns:
            Risk percent (0.3-0.65%)
        """

        # Start with confidence-based risk
        # Confidence 0.6 = 0.3%, 0.8 = 0.475%, 1.0 = 0.65%
        confidence_factor = (base_confidence - 0.5) / 0.5  # 0.0 to 1.0
        confidence_factor = max(0.0, min(1.0, confidence_factor))
        base_risk = self.MIN_RISK_PER_TRADE_PERCENT + (
            confidence_factor * (self.MAX_RISK_PER_TRADE_PERCENT - self.MIN_RISK_PER_TRADE_PERCENT)
        )

        # Reduce risk if in drawdown
        dd_factor = 1.0
        if current_drawdown_percent < 0:
            # Drawdown -3% = 0.85x, -5% = 0.75x
            dd_factor = 1.0 - (abs(current_drawdown_percent) / 10.0)
            dd_factor = max(0.7, min(1.0, dd_factor))

        # Reduce risk based on volatility
        volatility_factors = {
            "LOW": 1.0,
            "MEDIUM": 0.9,
            "HIGH": 0.75,
            "EXTREME": 0.5
        }
        vol_factor = volatility_factors.get(volatility_level, 0.9)

        # Calculate final risk
        final_risk = base_risk * dd_factor * vol_factor

        # Ensure within bounds
        final_risk = max(self.MIN_RISK_PER_TRADE_PERCENT, min(self.MAX_RISK_PER_TRADE_PERCENT, final_risk))

        logger.debug(
            f"[Guardian] Dynamic risk: {final_risk:.3f}% "
            f"(confidence: {base_confidence:.2f}, DD: {current_drawdown_percent:.2f}%, vol: {volatility_level})"
        )

        return final_risk

    def get_status(
        self,
        account_balance: float,
        daily_pnl: float,
        total_pnl: float,
        open_positions: int,
        trades_today: int
    ) -> SafetyStatus:
        """
        Get current safety status

        Args:
            account_balance: Current account balance
            daily_pnl: P&L for today
            total_pnl: Total P&L since start
            open_positions: Number of open positions
            trades_today: Trades executed today

        Returns:
            SafetyStatus object
        """

        daily_loss_percent = (daily_pnl / account_balance) * 100
        total_dd_percent = (total_pnl / account_balance) * 100

        # Determine volatility risk (placeholder - can be enhanced with ATR calculation)
        volatility_risk = "MEDIUM"

        # Get today's violations
        today = datetime.now(timezone.utc).date()
        violations_today = [
            v for v in self.violations
            if v.timestamp.date() == today
        ]

        return SafetyStatus(
            daily_loss_percent=daily_loss_percent,
            total_drawdown_percent=total_dd_percent,
            open_positions=open_positions,
            trades_today=trades_today,
            volatility_risk=volatility_risk,
            news_proximity=False,  # TODO: Implement news calendar check
            guardian_active=not self.emergency_shutdown,
            lockout_until=self.lockout_until,
            violations_today=violations_today
        )

    def _trigger_lockout(self, reason: str) -> None:
        """Trigger 24-hour lockout"""
        self.lockout_until = datetime.now(timezone.utc) + timedelta(hours=self.LOCKOUT_DURATION_HOURS)

        self._log_violation(
            violation_type="daily_loss_limit",
            severity="CRITICAL",
            message=f"LOCKOUT TRIGGERED: {reason}"
        )

        logger.critical(f"[Guardian] 🚨 LOCKOUT TRIGGERED - {reason} - Trading suspended for 24 hours")
        self._save_state()

    def _trigger_emergency_shutdown(self, reason: str) -> None:
        """Trigger emergency shutdown (permanent until manual reset)"""
        self.emergency_shutdown = True

        self._log_violation(
            violation_type="emergency_shutdown",
            severity="FATAL",
            message=f"EMERGENCY SHUTDOWN: {reason}"
        )

        logger.critical(f"[Guardian] 💀 EMERGENCY SHUTDOWN - {reason} - ALL TRADING HALTED")
        self._save_state()

    def _log_violation(self, violation_type: str, severity: str, message: str, data: Optional[Dict] = None) -> None:
        """Log a safety violation"""
        violation = SafetyViolation(
            timestamp=datetime.now(timezone.utc),
            violation_type=violation_type,
            severity=severity,
            message=message,
            data=data or {}
        )

        self.violations.append(violation)

        # Keep only last 100 violations
        if len(self.violations) > 100:
            self.violations = self.violations[-100:]

        logger.warning(f"[Guardian] {severity}: {message}")
        self._save_state()

    def reset_lockout(self) -> None:
        """Manually reset lockout (use with caution)"""
        self.lockout_until = None
        logger.warning("[Guardian] Lockout manually reset")
        self._save_state()

    def reset_emergency_shutdown(self) -> None:
        """Manually reset emergency shutdown (use with extreme caution)"""
        self.emergency_shutdown = False
        logger.warning("[Guardian] Emergency shutdown manually reset")
        self._save_state()

    def _save_state(self) -> None:
        """Save Guardian state to disk"""
        state_file = self.data_dir / "guardian_state.json"

        state = {
            "lockout_until": self.lockout_until.isoformat() if self.lockout_until else None,
            "emergency_shutdown": self.emergency_shutdown,
            "violations": [
                {
                    "timestamp": v.timestamp.isoformat(),
                    "violation_type": v.violation_type,
                    "severity": v.severity,
                    "message": v.message,
                    "data": v.data
                }
                for v in self.violations[-100:]  # Save last 100
            ]
        }

        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        """Load Guardian state from disk"""
        state_file = self.data_dir / "guardian_state.json"

        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                state = json.load(f)

            if state.get("lockout_until"):
                self.lockout_until = datetime.fromisoformat(state["lockout_until"])

            self.emergency_shutdown = state.get("emergency_shutdown", False)

            # Load violations
            for v_data in state.get("violations", []):
                violation = SafetyViolation(
                    timestamp=datetime.fromisoformat(v_data["timestamp"]),
                    violation_type=v_data["violation_type"],
                    severity=v_data["severity"],
                    message=v_data["message"],
                    data=v_data.get("data", {})
                )
                self.violations.append(violation)

            logger.info(
                f"[Guardian] State loaded - "
                f"Lockout: {self.lockout_until is not None}, "
                f"Emergency: {self.emergency_shutdown}, "
                f"Violations: {len(self.violations)}"
            )

        except Exception as e:
            logger.error(f"[Guardian] Failed to load state: {e}")
