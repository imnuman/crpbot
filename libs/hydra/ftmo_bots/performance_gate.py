"""
Performance Gate - Auto-disable underperforming bots based on trade history.

Checks the trade ledger and disables bots that fall below performance thresholds.
This prevents continued losses from bots that aren't working.

Created: 2025-12-11
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Set, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Performance thresholds
MIN_TRADES_FOR_EVALUATION = 5  # Need at least 5 trades to evaluate
MIN_WIN_RATE = 30.0  # Minimum 30% win rate to stay enabled
MAX_CONSECUTIVE_LOSSES = 4  # Disable after 4 consecutive losses
MAX_LOSS_PER_BOT_USD = 200.0  # Max $200 loss per bot before disable


class PerformanceGate:
    """
    Gates bot execution based on historical performance.

    Uses trade ledger data to:
    1. Track win rates per bot
    2. Auto-disable bots below thresholds
    3. Re-enable bots that improve in paper mode
    """

    def __init__(self, ledger: Any):
        """
        Initialize performance gate.

        Args:
            ledger: TradeLedger instance for querying trade history
        """
        self.ledger = ledger
        self._disabled_bots: Set[str] = set()
        self._bot_stats: Dict[str, Dict] = {}
        self._last_check: Optional[datetime] = None

        # Load initial stats
        self.refresh_stats()

    def refresh_stats(self) -> Dict[str, Dict]:
        """Refresh bot statistics from ledger."""
        try:
            stats = self.ledger.get_stats(days=30)
            self._bot_stats = stats.get("by_bot", {})
            self._last_check = datetime.now(timezone.utc)

            # Check each bot against thresholds
            newly_disabled = []
            for bot_name, bot_stats in self._bot_stats.items():
                trades = bot_stats.get("trades", 0)
                win_rate = bot_stats.get("win_rate", 0)
                pnl = bot_stats.get("pnl", 0)

                should_disable = False
                reason = ""

                # Check thresholds only if enough trades
                if trades >= MIN_TRADES_FOR_EVALUATION:
                    if win_rate < MIN_WIN_RATE:
                        should_disable = True
                        reason = f"Win rate {win_rate:.1f}% < {MIN_WIN_RATE}%"

                    if pnl < -MAX_LOSS_PER_BOT_USD:
                        should_disable = True
                        reason = f"P&L ${pnl:.2f} < -${MAX_LOSS_PER_BOT_USD}"

                if should_disable and bot_name not in self._disabled_bots:
                    self._disabled_bots.add(bot_name)
                    newly_disabled.append((bot_name, reason))
                    logger.warning(f"[PerformanceGate] Auto-disabled {bot_name}: {reason}")

            if newly_disabled:
                logger.info(f"[PerformanceGate] Disabled {len(newly_disabled)} bots: {newly_disabled}")

            return self._bot_stats

        except Exception as e:
            logger.error(f"[PerformanceGate] Failed to refresh stats: {e}")
            return {}

    def is_bot_allowed(self, bot_name: str) -> tuple[bool, str]:
        """
        Check if a bot is allowed to trade.

        Args:
            bot_name: Name of the bot to check

        Returns:
            (allowed, reason) - tuple of bool and explanation string
        """
        # Normalize bot name (handle variations)
        normalized = self._normalize_bot_name(bot_name)

        # Check if explicitly disabled
        if normalized in self._disabled_bots:
            stats = self._bot_stats.get(normalized, {})
            wr = stats.get("win_rate", 0)
            pnl = stats.get("pnl", 0)
            return False, f"Auto-disabled (WR={wr:.1f}%, P&L=${pnl:.2f})"

        # Check if we have stats for this bot
        if normalized in self._bot_stats:
            stats = self._bot_stats[normalized]
            trades = stats.get("trades", 0)
            win_rate = stats.get("win_rate", 0)
            pnl = stats.get("pnl", 0)

            # Only block if enough data AND below threshold
            if trades >= MIN_TRADES_FOR_EVALUATION:
                if win_rate < MIN_WIN_RATE:
                    self._disabled_bots.add(normalized)
                    return False, f"WR {win_rate:.1f}% < {MIN_WIN_RATE}% min"

                if pnl < -MAX_LOSS_PER_BOT_USD:
                    self._disabled_bots.add(normalized)
                    return False, f"P&L ${pnl:.2f} exceeds -${MAX_LOSS_PER_BOT_USD} limit"

        return True, "OK"

    def _normalize_bot_name(self, bot_name: str) -> str:
        """Normalize bot name for consistent lookup."""
        # Handle common variations
        name_map = {
            "gold_london": "GoldLondon",
            "GoldLondonReversal": "GoldLondon",
            "eurusd": "EURUSDBreak",
            "EURUSDBreakout": "EURUSDBreak",
            "us30": "US30ORB",
            "US30ORB": "US30ORB",
            "gold_ny": "GoldNYRever",
            "GoldNYReversion": "GoldNYRever",
            "london_eur": "LondonBreak",
            "LondonBreakout": "LondonBreak",
            "nas100": "NAS100Gap",
            "NAS100Gap": "NAS100Gap",
            "hf_scalper": "HFScalper",
            "HFScalper": "HFScalper",
        }
        return name_map.get(bot_name, bot_name)

    def get_bot_status(self) -> Dict[str, Dict]:
        """Get status of all bots."""
        status = {}
        for bot_name, stats in self._bot_stats.items():
            allowed, reason = self.is_bot_allowed(bot_name)
            status[bot_name] = {
                "allowed": allowed,
                "reason": reason,
                "trades": stats.get("trades", 0),
                "win_rate": stats.get("win_rate", 0),
                "pnl": stats.get("pnl", 0)
            }
        return status

    def force_enable(self, bot_name: str) -> bool:
        """Force enable a disabled bot (for manual override)."""
        normalized = self._normalize_bot_name(bot_name)
        if normalized in self._disabled_bots:
            self._disabled_bots.remove(normalized)
            logger.info(f"[PerformanceGate] Force-enabled {bot_name}")
            return True
        return False

    def get_disabled_bots(self) -> Set[str]:
        """Get set of currently disabled bots."""
        return self._disabled_bots.copy()


# Singleton instance
_gate: Optional[PerformanceGate] = None


def get_performance_gate(ledger: Any) -> PerformanceGate:
    """Get or create performance gate singleton."""
    global _gate
    if _gate is None:
        _gate = PerformanceGate(ledger)
    return _gate
