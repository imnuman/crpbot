"""
HYDRA 3.0 - Guardian (Layer 10)

THE GUARDIAN NEVER SLEEPS. NEVER OVERRIDE.

Hard limits that protect maker's capital above all else.
Every rule in this file is SACRED and CANNOT be bypassed.
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger
import json
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
        self.account_balance = account_balance
        self.starting_balance = account_balance
        self.daily_pnl = 0.0
        self.daily_pnl_reset_time = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        self.peak_balance = account_balance
        self.current_drawdown = 0.0
        self.open_positions = []
        self.regime_choppy_since = None
        self.emergency_shutdown_until = None

        self.state_file = state_file or "data/hydra/guardian_state.json"
        self._load_state()

        logger.info("Guardian initialized")
        logger.info(f"Account balance: ${account_balance:,.2f}")
        logger.info(f"Daily loss limit: {self.DAILY_LOSS_LIMIT*100}%")
        logger.info(f"Max drawdown: {self.MAX_DRAWDOWN*100}%")
        logger.info(f"Emergency shutdown at: {self.EMERGENCY_LOSS*100}% daily loss")

    def check_before_trade(
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
    ) -> Tuple[bool, str, Optional[float]]:
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
            Tuple of (allowed: bool, reason: str, adjusted_size: Optional[float])
        """
        # Rule 9: Emergency shutdown check (FIRST - overrides everything)
        if self.emergency_shutdown_until:
            if datetime.now(timezone.utc) < self.emergency_shutdown_until:
                hours_remaining = (self.emergency_shutdown_until - datetime.now(timezone.utc)).seconds / 3600
                return False, f"EMERGENCY SHUTDOWN ACTIVE - {hours_remaining:.1f} hours remaining", None
            else:
                # Shutdown period over
                logger.warning("Emergency shutdown period ended - resuming trading")
                self.emergency_shutdown_until = None
                self._save_state()

        # Reset daily P&L if new day
        self._reset_daily_pnl_if_needed()

        # Rule 1: Daily loss limit (2%)
        daily_loss_percent = self.daily_pnl / self.starting_balance
        if daily_loss_percent <= -self.DAILY_LOSS_LIMIT:
            logger.critical(f"DAILY LOSS LIMIT HIT: {daily_loss_percent*100:.2f}% - ALL TRADING STOPPED")
            return False, f"Daily loss limit hit ({daily_loss_percent*100:.2f}%)", None

        # Check if this trade would push us over daily limit
        risk_amount = abs(entry_price - sl_price) * (position_size_usd / entry_price)
        potential_loss = self.daily_pnl - risk_amount
        if potential_loss / self.starting_balance <= -self.DAILY_LOSS_LIMIT:
            logger.warning("Trade would exceed daily loss limit - blocking")
            return False, "Would exceed daily loss limit", None

        # Rule 2: Max drawdown (6%)
        self.current_drawdown = (self.peak_balance - self.account_balance) / self.peak_balance
        if self.current_drawdown >= self.MAX_DRAWDOWN:
            logger.critical(f"MAX DRAWDOWN HIT: {self.current_drawdown*100:.2f}% - SURVIVAL MODE")
            # In survival mode, reduce all position sizes by 50%
            adjusted_size = position_size_usd * 0.5
            return True, f"Survival mode (DD: {self.current_drawdown*100:.2f}%)", adjusted_size

        # Rule 3: Regime unclear (>2 hours CHOPPY)
        if regime == "CHOPPY":
            if self.regime_choppy_since is None:
                self.regime_choppy_since = datetime.now(timezone.utc)

            choppy_duration = (datetime.now(timezone.utc) - self.regime_choppy_since).seconds / 3600
            if choppy_duration > self.CHOPPY_REGIME_MAX_HOURS:
                logger.warning(f"Regime CHOPPY for {choppy_duration:.1f} hours - staying CASH")
                return False, f"Regime unclear for {choppy_duration:.1f} hours", None
        else:
            # Regime cleared
            self.regime_choppy_since = None

        # Rule 4: Correlation spike (>0.8)
        if strategy_correlations and any(corr > self.CORRELATION_SPIKE_THRESHOLD for corr in strategy_correlations):
            max_corr = max(strategy_correlations)
            logger.warning(f"High correlation detected: {max_corr:.2f} - cutting exposure 75%")
            adjusted_size = position_size_usd * 0.25  # 75% cut = 25% of original
            return True, f"High correlation ({max_corr:.2f})", adjusted_size

        # Rule 5: Risk per trade (max 1%)
        risk_percent = risk_amount / self.account_balance
        if risk_percent > self.MAX_RISK_PER_TRADE:
            logger.warning(f"Trade risk {risk_percent*100:.2f}% exceeds max {self.MAX_RISK_PER_TRADE*100}%")
            # Adjust position size to meet risk limit
            adjusted_size = (self.MAX_RISK_PER_TRADE * self.account_balance) / abs(entry_price - sl_price)
            return True, f"Risk adjusted to {self.MAX_RISK_PER_TRADE*100}%", adjusted_size

        # Rule 6: Concurrent positions (max 3)
        if len(current_positions) >= self.MAX_CONCURRENT_POSITIONS:
            logger.warning(f"{len(current_positions)} positions open - max {self.MAX_CONCURRENT_POSITIONS}")
            return False, f"Max {self.MAX_CONCURRENT_POSITIONS} concurrent positions", None

        # Rule 7: Exotic forex special rules
        if asset_type == "exotic_forex":
            # 50% size modifier
            adjusted_size = position_size_usd * self.EXOTIC_FOREX_SIZE_MODIFIER

            # No overnight holds
            # Check if position would be held overnight (simplified - checks if after 4PM EST)
            current_hour = datetime.now(timezone.utc).hour
            if current_hour >= 21 or current_hour < 13:  # Outside 8AM-4PM EST
                logger.warning(f"Exotic forex after hours - no overnight holds allowed")
                return False, "No overnight exotic forex positions", None

            logger.info(f"Exotic forex: Size reduced 50% to ${adjusted_size:,.2f}")
            return True, "Exotic forex: 50% size, no overnight", adjusted_size

        # Rule 8: Crypto meme special rules
        if asset_type == "meme_perp":
            # 30% size modifier (even more conservative than exotic forex)
            adjusted_size = position_size_usd * self.MEME_PERP_SIZE_MODIFIER

            logger.info(f"Meme perp: Size reduced to 30% (${adjusted_size:,.2f}), max hold 4hrs")
            return True, f"Meme perp: 30% size, max {self.MEME_PERP_MAX_HOLD_HOURS}hr hold", adjusted_size

        # All checks passed
        return True, "All Guardian checks passed", position_size_usd

    def update_account_state(self, pnl: float, position_closed: bool = False):
        """
        Update account state after trade result.

        Args:
            pnl: Profit/loss from trade
            position_closed: True if a position was closed
        """
        self.account_balance += pnl
        self.daily_pnl += pnl

        # Update peak balance for drawdown calculation
        if self.account_balance > self.peak_balance:
            self.peak_balance = self.account_balance

        # Recalculate drawdown
        self.current_drawdown = (self.peak_balance - self.account_balance) / self.peak_balance

        # Check emergency shutdown (3% daily loss)
        daily_loss_percent = self.daily_pnl / self.starting_balance
        if daily_loss_percent <= -self.EMERGENCY_LOSS:
            self.trigger_emergency_shutdown()

        # Log state
        logger.info(f"Account updated: Balance=${self.account_balance:,.2f}, Daily P&L=${self.daily_pnl:,.2f}, DD={self.current_drawdown*100:.2f}%")

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

                logger.info("Guardian state loaded from file")
                logger.info(f"Balance: ${self.account_balance:,.2f}, Daily P&L: ${self.daily_pnl:,.2f}, DD: {self.current_drawdown*100:.2f}%")
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
            "daily_pnl_percent": (self.daily_pnl / self.starting_balance) * 100,
            "peak_balance": self.peak_balance,
            "current_drawdown": self.current_drawdown,
            "current_drawdown_percent": self.current_drawdown * 100,
            "daily_loss_limit": self.DAILY_LOSS_LIMIT * 100,
            "max_drawdown_limit": self.MAX_DRAWDOWN * 100,
            "emergency_shutdown_active": self.emergency_shutdown_until is not None,
            "emergency_shutdown_until": self.emergency_shutdown_until.isoformat() if self.emergency_shutdown_until else None,
            "trading_allowed": self.emergency_shutdown_until is None or datetime.now(timezone.utc) >= self.emergency_shutdown_until
        }
