"""
HYDRA 3.0 - Guardian (Layer 10)

THE GUARDIAN NEVER SLEEPS. NEVER OVERRIDE.

Hard limits that protect maker's capital above all else.
Every rule in this file is SACRED and CANNOT be bypassed.
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from loguru import logger
import json
import threading
from pathlib import Path


class Guardian:
    """
    Guardian enforces hard limits on all trading activity.

    Sacred Rules (NEVER OVERRIDE):
    1. Daily loss limit: 2%
    2. Max drawdown: 6%
    3. Regime unclear: >2 hours CHOPPY
    4. Correlation spike: >0.8
    5. Risk per trade: Max 1%
    6. Concurrent positions: Max 3
    7. Exotic forex: 50% size, no overnight
    8. Crypto meme: 50% size, max 4hr hold
    9. Emergency: 3% daily loss → OFFLINE 24hrs
    """

    # Hard limits - NEVER CHANGE WITHOUT EXTREME CAUTION
    DAILY_LOSS_LIMIT = 0.02  # 2%
    MAX_DRAWDOWN = 0.06      # 6%
    EMERGENCY_LOSS = 0.03    # 3% - triggers 24hr shutdown
    MAX_RISK_PER_TRADE = 0.01  # 1%
    MAX_CONCURRENT_POSITIONS = 3
    CHOPPY_REGIME_MAX_HOURS = 2
    CORRELATION_SPIKE_THRESHOLD = 0.8

    # Circuit Breaker Settings (MOD 6)
    CONSECUTIVE_LOSS_REDUCE = 3   # 3 losses → 50% position size
    CONSECUTIVE_LOSS_PAUSE = 5    # 5 losses → 24hr pause
    REDUCED_SIZE_MULTIPLIER = 0.5  # 50% of normal

    # Asset-specific modifiers
    EXOTIC_FOREX_SIZE_MODIFIER = 0.5  # 50% of normal
    MEME_PERP_SIZE_MODIFIER = 0.3     # 30% of normal
    MEME_PERP_MAX_HOLD_HOURS = 4

    def __init__(self, account_balance: float, state_file: Optional[str] = None):
        """
        Initialize Guardian.

        Args:
            account_balance: Starting account balance
            state_file: Path to persist Guardian state (for emergency shutdowns)
        """
        # Thread-safety lock for state modifications
        self._lock = threading.Lock()

        self.account_balance = account_balance
        self.starting_balance = account_balance
        self.daily_pnl = 0.0
        self.daily_pnl_reset_time = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        self.peak_balance = account_balance
        self.current_drawdown = 0.0
        self.open_positions = []
        self.regime_choppy_since = None
        self.emergency_shutdown_until = None

        # Circuit breaker tracking (MOD 6)
        self.consecutive_losses = 0
        self.circuit_breaker_active = False  # True = reduced mode

        self.state_file = state_file or "data/hydra/guardian_state.json"
        self._load_state()

        logger.info("Guardian initialized")
        logger.info(f"Account balance: ${account_balance:,.2f}")
        logger.info(f"Daily loss limit: {self.DAILY_LOSS_LIMIT*100}%")
        logger.info(f"Max drawdown: {self.MAX_DRAWDOWN*100}%")
        logger.info(f"Emergency shutdown at: {self.EMERGENCY_LOSS*100}% daily loss")

    def validate_trade(
        self,
        asset: str,
        asset_type: str,
        direction: str,
        position_size_usd: float,
        entry_price: float,
        sl_price: float,
        regime: str,
        current_positions: List[Dict],
        strategy_correlations: Optional[List[float]] = None
    ) -> Dict:
        """
        Check all Guardian rules before allowing a trade.

        Args:
            asset: Symbol (e.g., "USD/TRY", "BONK")
            asset_type: "exotic_forex" or "meme_perp" or "standard"
            direction: "LONG" or "SHORT"
            position_size_usd: Position size in USD
            entry_price: Entry price
            sl_price: Stop loss price
            regime: Current market regime
            current_positions: List of open positions
            strategy_correlations: Correlation between strategies

        Returns:
            Dict with keys: "approved" (bool), "rejection_reason" (str), "adjusted_size" (Optional[float])
        """
        # Rule 9: Emergency shutdown check (FIRST - overrides everything)
        if self.emergency_shutdown_until:
            if datetime.now(timezone.utc) < self.emergency_shutdown_until:
                hours_remaining = (self.emergency_shutdown_until - datetime.now(timezone.utc)).seconds / 3600
                return {
                    "approved": False,
                    "rejection_reason": f"EMERGENCY SHUTDOWN ACTIVE - {hours_remaining:.1f} hours remaining",
                    "adjusted_size": None
                }
            else:
                # Shutdown period over
                logger.warning("Emergency shutdown period ended - resuming trading")
                self.emergency_shutdown_until = None
                self._save_state()

        # Reset daily P&L if new day
        self._reset_daily_pnl_if_needed()

        # Circuit Breaker: 5 consecutive losses → 24hr pause
        if self.consecutive_losses >= self.CONSECUTIVE_LOSS_PAUSE:
            self.trigger_emergency_shutdown()
            return {
                "approved": False,
                "rejection_reason": f"Circuit breaker: {self.consecutive_losses} consecutive losses - 24hr pause",
                "adjusted_size": None
            }

        # Circuit Breaker: 3 consecutive losses → 50% position size
        size_multiplier = 1.0
        if self.consecutive_losses >= self.CONSECUTIVE_LOSS_REDUCE:
            size_multiplier = self.REDUCED_SIZE_MULTIPLIER
            self.circuit_breaker_active = True
            logger.warning(f"Circuit breaker: {self.consecutive_losses} consecutive losses - reducing position size to 50%")

        # Rule 1: Daily loss limit (2%)
        daily_loss_percent = self.daily_pnl / self.starting_balance if self.starting_balance != 0 else 0
        if daily_loss_percent <= -self.DAILY_LOSS_LIMIT:
            logger.critical(f"DAILY LOSS LIMIT HIT: {daily_loss_percent*100:.2f}% - ALL TRADING STOPPED")
            return {
                "approved": False,
                "rejection_reason": f"Daily loss limit hit ({daily_loss_percent*100:.2f}%)",
                "adjusted_size": None
            }

        # Check if this trade would push us over daily limit
        if entry_price != 0:
            risk_amount = abs(entry_price - sl_price) * (position_size_usd / entry_price)
            potential_loss = self.daily_pnl - risk_amount
            if self.starting_balance != 0 and potential_loss / self.starting_balance <= -self.DAILY_LOSS_LIMIT:
                logger.warning("Trade would exceed daily loss limit - blocking")
                return {
                    "approved": False,
                    "rejection_reason": "Would exceed daily loss limit",
                    "adjusted_size": None
                }

        # Rule 2: Max drawdown (6%)
        self.current_drawdown = (self.peak_balance - self.account_balance) / self.peak_balance if self.peak_balance != 0 else 0
        if self.current_drawdown >= self.MAX_DRAWDOWN:
            logger.critical(f"MAX DRAWDOWN HIT: {self.current_drawdown*100:.2f}% - SURVIVAL MODE")
            # In survival mode, reduce all position sizes by 50%
            adjusted_size = position_size_usd * 0.5
            return {
                "approved": True,
                "rejection_reason": f"Survival mode (DD: {self.current_drawdown*100:.2f}%)",
                "adjusted_size": adjusted_size
            }

        # Rule 3: Regime unclear (>2 hours CHOPPY)
        if regime == "CHOPPY":
            if self.regime_choppy_since is None:
                self.regime_choppy_since = datetime.now(timezone.utc)

            choppy_duration = (datetime.now(timezone.utc) - self.regime_choppy_since).total_seconds() / 3600
            if choppy_duration > self.CHOPPY_REGIME_MAX_HOURS:
                logger.warning(f"Regime CHOPPY for {choppy_duration:.1f} hours - staying CASH")
                return {
                    "approved": False,
                    "rejection_reason": f"Regime unclear for {choppy_duration:.1f} hours",
                    "adjusted_size": None
                }
        else:
            # Regime cleared
            self.regime_choppy_since = None

        # Rule 4: Correlation spike (>0.8)
        if strategy_correlations and any(corr > self.CORRELATION_SPIKE_THRESHOLD for corr in strategy_correlations):
            max_corr = max(strategy_correlations)
            logger.warning(f"High correlation detected: {max_corr:.2f} - cutting exposure 75%")
            adjusted_size = position_size_usd * 0.25  # 75% cut = 25% of original
            return {
                "approved": True,
                "rejection_reason": f"High correlation ({max_corr:.2f})",
                "adjusted_size": adjusted_size
            }

        # Rule 5: Risk per trade (max 1%)
        if entry_price != 0:
            risk_amount = abs(entry_price - sl_price) * (position_size_usd / entry_price)
            risk_percent = risk_amount / self.account_balance if self.account_balance != 0 else 0
            if risk_percent > self.MAX_RISK_PER_TRADE:
                logger.warning(f"Trade risk {risk_percent*100:.2f}% exceeds max {self.MAX_RISK_PER_TRADE*100}%")
                # Adjust position size to meet risk limit
                if abs(entry_price - sl_price) > 0:
                    adjusted_size = (self.MAX_RISK_PER_TRADE * self.account_balance) / abs(entry_price - sl_price)
                    return {
                        "approved": True,
                        "rejection_reason": f"Risk adjusted to {self.MAX_RISK_PER_TRADE*100}%",
                        "adjusted_size": adjusted_size
                    }

        # Rule 6: Concurrent positions (max 3)
        if len(current_positions) >= self.MAX_CONCURRENT_POSITIONS:
            logger.warning(f"{len(current_positions)} positions open - max {self.MAX_CONCURRENT_POSITIONS}")
            return {
                "approved": False,
                "rejection_reason": f"Max {self.MAX_CONCURRENT_POSITIONS} concurrent positions",
                "adjusted_size": None
            }

        # Rule 7: Exotic forex special rules
        if asset_type == "exotic_forex":
            # 50% size modifier
            adjusted_size = position_size_usd * self.EXOTIC_FOREX_SIZE_MODIFIER

            # No overnight holds
            # Check if position would be held overnight (simplified - checks if after 4PM EST)
            current_hour = datetime.now(timezone.utc).hour
            if current_hour >= 21 or current_hour < 13:  # Outside 8AM-4PM EST
                logger.warning(f"Exotic forex after hours - no overnight holds allowed")
                return {
                    "approved": False,
                    "rejection_reason": "No overnight exotic forex positions",
                    "adjusted_size": None
                }

            logger.info(f"Exotic forex: Size reduced 50% to ${adjusted_size:,.2f}")
            return {
                "approved": True,
                "rejection_reason": "Exotic forex: 50% size, no overnight",
                "adjusted_size": adjusted_size
            }

        # Rule 8: Crypto meme special rules
        if asset_type == "meme_perp":
            # 30% size modifier (even more conservative than exotic forex)
            adjusted_size = position_size_usd * self.MEME_PERP_SIZE_MODIFIER

            logger.info(f"Meme perp: Size reduced to 30% (${adjusted_size:,.2f}), max hold 4hrs")
            return {
                "approved": True,
                "rejection_reason": f"Meme perp: 30% size, max {self.MEME_PERP_MAX_HOLD_HOURS}hr hold",
                "adjusted_size": adjusted_size
            }

        # All checks passed - apply circuit breaker multiplier if active
        final_size = position_size_usd * size_multiplier
        reason = "All Guardian checks passed"
        if size_multiplier < 1.0:
            reason = f"Approved (circuit breaker: {size_multiplier:.0%} size due to {self.consecutive_losses} losses)"

        return {
            "approved": True,
            "rejection_reason": reason,
            "adjusted_size": final_size
        }

    def update_account_state(self, pnl: float, position_closed: bool = False, won: bool = None):
        """
        Update account state after trade result.

        Args:
            pnl: Profit/loss from trade
            position_closed: True if a position was closed
            won: True if trade was a winner (for circuit breaker tracking)
        """
        self.account_balance += pnl
        self.daily_pnl += pnl

        # Track consecutive losses for circuit breaker
        if won is not None:
            if won:
                # Win resets consecutive losses
                self.consecutive_losses = 0
                self.circuit_breaker_active = False
                logger.info("Circuit breaker: Reset after winning trade")
            else:
                # Loss increments counter
                self.consecutive_losses += 1
                logger.warning(f"Circuit breaker: Consecutive losses now {self.consecutive_losses}")

        # Update peak balance for drawdown calculation
        if self.account_balance > self.peak_balance:
            self.peak_balance = self.account_balance

        # Recalculate drawdown
        self.current_drawdown = (self.peak_balance - self.account_balance) / self.peak_balance if self.peak_balance != 0 else 0

        # Check emergency shutdown (3% daily loss)
        daily_loss_percent = self.daily_pnl / self.starting_balance if self.starting_balance != 0 else 0
        if daily_loss_percent <= -self.EMERGENCY_LOSS:
            self.trigger_emergency_shutdown()

        # Check circuit breaker for pause (5 consecutive losses)
        if self.consecutive_losses >= self.CONSECUTIVE_LOSS_PAUSE:
            logger.critical(f"Circuit breaker: {self.consecutive_losses} consecutive losses - triggering 24hr pause")
            self.trigger_emergency_shutdown()

        # Log state
        logger.info(f"Account updated: Balance=${self.account_balance:,.2f}, Daily P&L=${self.daily_pnl:,.2f}, DD={self.current_drawdown*100:.2f}%, Losses={self.consecutive_losses}")

        self._save_state()

    def trigger_emergency_shutdown(self):
        """
        Trigger 24-hour emergency shutdown after 3% daily loss.
        """
        self.emergency_shutdown_until = datetime.now(timezone.utc) + timedelta(hours=24)

        logger.critical("╔══════════════════════════════════════════════════╗")
        logger.critical("║  EMERGENCY SHUTDOWN TRIGGERED                    ║")
        logger.critical(f"║  Daily loss: {(self.daily_pnl/self.starting_balance)*100:.2f}%                                    ║")
        logger.critical(f"║  Shutdown until: {self.emergency_shutdown_until.strftime('%Y-%m-%d %H:%M UTC')}      ║")
        logger.critical("║  ALL TRADING SUSPENDED FOR 24 HOURS              ║")
        logger.critical("╚══════════════════════════════════════════════════╝")

        self._save_state()

    def _reset_daily_pnl_if_needed(self):
        """Reset daily P&L at midnight UTC."""
        now = datetime.now(timezone.utc)
        current_day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        if now >= self.daily_pnl_reset_time + timedelta(days=1):
            logger.info(f"New trading day - Daily P&L reset (was ${self.daily_pnl:,.2f})")
            self.daily_pnl = 0.0
            self.daily_pnl_reset_time = current_day_start
            self._save_state()

    def _save_state(self):
        """Persist Guardian state to file."""
        state = {
            "account_balance": self.account_balance,
            "starting_balance": self.starting_balance,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_reset_time": self.daily_pnl_reset_time.isoformat(),
            "peak_balance": self.peak_balance,
            "current_drawdown": self.current_drawdown,
            "emergency_shutdown_until": self.emergency_shutdown_until.isoformat() if self.emergency_shutdown_until else None,
            "consecutive_losses": self.consecutive_losses,
            "circuit_breaker_active": self.circuit_breaker_active,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }

        Path(self.state_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load Guardian state from file if exists."""
        try:
            if Path(self.state_file).exists():
                with open(self.state_file, 'r') as f:
                    state = json.load(f)

                self.account_balance = state["account_balance"]
                self.starting_balance = state["starting_balance"]
                self.daily_pnl = state["daily_pnl"]
                self.daily_pnl_reset_time = datetime.fromisoformat(state["daily_pnl_reset_time"])
                self.peak_balance = state["peak_balance"]
                self.current_drawdown = state["current_drawdown"]

                if state.get("emergency_shutdown_until"):
                    self.emergency_shutdown_until = datetime.fromisoformat(state["emergency_shutdown_until"])

                # Load circuit breaker state
                self.consecutive_losses = state.get("consecutive_losses", 0)
                self.circuit_breaker_active = state.get("circuit_breaker_active", False)

                logger.info("Guardian state loaded from file")
                logger.info(f"Balance: ${self.account_balance:,.2f}, Daily P&L: ${self.daily_pnl:,.2f}, DD: {self.current_drawdown*100:.2f}%, Losses: {self.consecutive_losses}")
        except Exception as e:
            logger.warning(f"Could not load Guardian state: {e} - starting fresh")

    def get_status(self) -> Dict:
        """
        Get current Guardian status.

        Returns:
            Dict with all Guardian metrics
        """
        return {
            "account_balance": self.account_balance,
            "starting_balance": self.starting_balance,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_percent": (self.daily_pnl / self.starting_balance) * 100 if self.starting_balance != 0 else 0,
            "peak_balance": self.peak_balance,
            "current_drawdown": self.current_drawdown,
            "current_drawdown_percent": self.current_drawdown * 100,
            "daily_loss_limit": self.DAILY_LOSS_LIMIT * 100,
            "max_drawdown_limit": self.MAX_DRAWDOWN * 100,
            "emergency_shutdown_active": self.emergency_shutdown_until is not None,
            "emergency_shutdown_until": self.emergency_shutdown_until.isoformat() if self.emergency_shutdown_until else None,
            "trading_allowed": self.emergency_shutdown_until is None or datetime.now(timezone.utc) >= self.emergency_shutdown_until,
            # Circuit breaker status
            "consecutive_losses": self.consecutive_losses,
            "circuit_breaker_active": self.circuit_breaker_active,
            "position_size_multiplier": self.REDUCED_SIZE_MULTIPLIER if self.circuit_breaker_active else 1.0,
        }



# ==================== SINGLETON PATTERN ====================

_guardian = None
_guardian_lock = threading.Lock()

def get_guardian(account_balance: float = 10000.0) -> Guardian:
    """Get singleton instance of Guardian (thread-safe)."""
    global _guardian
    if _guardian is None:
        with _guardian_lock:
            # Double-check pattern for thread safety
            if _guardian is None:
                _guardian = Guardian(account_balance=account_balance)
    return _guardian
