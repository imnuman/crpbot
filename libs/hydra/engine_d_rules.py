"""
HYDRA 3.0 - Engine D Special Rules

Engine D (Gemini) handles regime transitions with special rules:
1. Can only activate ONCE every 14 days (regime changes are rare)
2. Must show positive expectancy before activation
3. ATR must expand 2x or more for trigger

Regime transitions are HIGH VALUE but RARE events.
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Configuration
ENGINE_D_COOLDOWN_DAYS = 14  # Only once every 14 days
MIN_EXPECTANCY = 0.0  # Must be positive
MIN_ATR_EXPANSION = 2.0  # ATR must double


@dataclass
class EngineDState:
    """Engine D activation state."""
    last_activation: Optional[datetime]
    total_activations: int
    total_trades: int
    wins: int
    losses: int
    total_pnl_pct: float

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades

    @property
    def avg_pnl(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl_pct / self.total_trades

    @property
    def expectancy(self) -> float:
        """Calculate expectancy: (WR * avg_win) - ((1-WR) * avg_loss)."""
        if self.total_trades == 0:
            return 0.0  # No data = neutral
        # Simplified: avg_pnl already accounts for wins/losses
        return self.avg_pnl

    @property
    def days_since_activation(self) -> Optional[int]:
        if self.last_activation is None:
            return None
        return (datetime.now() - self.last_activation).days

    @property
    def can_activate(self) -> bool:
        """Check if cooldown period has passed."""
        if self.last_activation is None:
            return True  # Never activated
        return self.days_since_activation >= ENGINE_D_COOLDOWN_DAYS

    @property
    def days_until_available(self) -> int:
        """Days until Engine D can activate again."""
        if self.can_activate:
            return 0
        return ENGINE_D_COOLDOWN_DAYS - self.days_since_activation

    def to_dict(self) -> dict:
        return {
            "last_activation": self.last_activation.isoformat() if self.last_activation else None,
            "total_activations": self.total_activations,
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "total_pnl_pct": self.total_pnl_pct,
            "win_rate": self.win_rate,
            "expectancy": self.expectancy,
            "days_since_activation": self.days_since_activation,
            "can_activate": self.can_activate,
            "days_until_available": self.days_until_available,
        }


class EngineDController:
    """
    Controls Engine D activation with special rules.

    Engine D handles regime transitions which are:
    - RARE (only happen occasionally)
    - HIGH VALUE (when they do happen)
    - RISKY (false signals are costly)

    Rules:
    1. 14-day cooldown between activations
    2. Must have positive expectancy (or no history)
    3. ATR must show 2x expansion
    """

    _instance: Optional["EngineDController"] = None

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
            if os.path.exists("/root/crpbot"):
                data_dir = Path("/root/crpbot/data/hydra")
            else:
                data_dir = Path.home() / "crpbot" / "data" / "hydra"

        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.state_file = self.data_dir / "engine_d_state.json"
        self.state = self._load_state()

        self._initialized = True
        logger.info(f"EngineDController initialized. State: {self.state.to_dict()}")

    def _load_state(self) -> EngineDState:
        """Load state from file or create default."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                return EngineDState(
                    last_activation=datetime.fromisoformat(data["last_activation"]) if data.get("last_activation") else None,
                    total_activations=data.get("total_activations", 0),
                    total_trades=data.get("total_trades", 0),
                    wins=data.get("wins", 0),
                    losses=data.get("losses", 0),
                    total_pnl_pct=data.get("total_pnl_pct", 0.0),
                )
            except Exception as e:
                logger.warning(f"Failed to load Engine D state: {e}")

        return EngineDState(
            last_activation=None,
            total_activations=0,
            total_trades=0,
            wins=0,
            losses=0,
            total_pnl_pct=0.0,
        )

    def _save_state(self):
        """Save state to file."""
        try:
            data = {
                "last_activation": self.state.last_activation.isoformat() if self.state.last_activation else None,
                "total_activations": self.state.total_activations,
                "total_trades": self.state.total_trades,
                "wins": self.state.wins,
                "losses": self.state.losses,
                "total_pnl_pct": self.state.total_pnl_pct,
            }
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save Engine D state: {e}")

    def check_activation(self, atr_multiplier: float, confidence: float = 0.70) -> dict:
        """
        Check if Engine D can activate.

        Args:
            atr_multiplier: Current ATR vs baseline (must be >= 2.0)
            confidence: Engine's confidence level (must be >= 0.70)

        Returns:
            Dict with allowed/denied and reason
        """
        result = {
            "allowed": False,
            "engine": "D",
            "trigger_type": "regime_transition",
            "atr_multiplier": atr_multiplier,
            "confidence": confidence,
            "reason": "",
            "state": self.state.to_dict(),
        }

        # Check 1: Cooldown period
        if not self.state.can_activate:
            result["reason"] = f"Cooldown active: {self.state.days_until_available} days remaining"
            return result

        # Check 2: ATR threshold
        if atr_multiplier < MIN_ATR_EXPANSION:
            result["reason"] = f"ATR multiplier {atr_multiplier:.1f}x below {MIN_ATR_EXPANSION}x threshold"
            return result

        # Check 3: Confidence threshold
        if confidence < 0.70:
            result["reason"] = f"Confidence {confidence:.0%} below 70% minimum"
            return result

        # Check 4: Expectancy (if we have history)
        if self.state.total_trades >= 3:  # Need at least 3 trades for expectancy
            if self.state.expectancy < MIN_EXPECTANCY:
                result["reason"] = f"Negative expectancy ({self.state.expectancy:.2f}%) - Engine D paused"
                return result

        # All checks passed
        result["allowed"] = True
        result["reason"] = f"Regime transition detected (ATR {atr_multiplier:.1f}x) - activation allowed"
        return result

    def record_activation(self):
        """Record that Engine D was activated."""
        self.state.last_activation = datetime.now()
        self.state.total_activations += 1
        self._save_state()
        logger.info(f"Engine D activated. Total activations: {self.state.total_activations}")

    def record_trade_result(self, pnl_pct: float, won: bool):
        """
        Record a trade result from Engine D.

        Args:
            pnl_pct: P&L percentage (positive or negative)
            won: Whether the trade was a win
        """
        self.state.total_trades += 1
        self.state.total_pnl_pct += pnl_pct

        if won:
            self.state.wins += 1
        else:
            self.state.losses += 1

        self._save_state()
        logger.info(
            f"Engine D trade recorded: {'WIN' if won else 'LOSS'} {pnl_pct:+.2f}% | "
            f"WR: {self.state.win_rate:.1%} | Expectancy: {self.state.expectancy:.2f}%"
        )

    def get_status(self) -> str:
        """Get human-readable status."""
        lines = ["=== Engine D Status ==="]
        lines.append(f"Specialty: Regime Transitions (ATR 2x expansion)")
        lines.append(f"Cooldown: {ENGINE_D_COOLDOWN_DAYS} days")
        lines.append("")

        if self.state.last_activation:
            lines.append(f"Last Activation: {self.state.last_activation.strftime('%Y-%m-%d %H:%M')}")
            lines.append(f"Days Since: {self.state.days_since_activation}")
        else:
            lines.append("Last Activation: Never")

        lines.append(f"Can Activate: {'Yes' if self.state.can_activate else f'No ({self.state.days_until_available}d remaining)'}")
        lines.append("")
        lines.append(f"Total Activations: {self.state.total_activations}")
        lines.append(f"Total Trades: {self.state.total_trades}")

        if self.state.total_trades > 0:
            lines.append(f"Win Rate: {self.state.win_rate:.1%}")
            lines.append(f"Total P&L: {self.state.total_pnl_pct:+.2f}%")
            lines.append(f"Expectancy: {self.state.expectancy:+.2f}%")

        return "\n".join(lines)

    def reset(self):
        """Reset Engine D state (for testing)."""
        self.state = EngineDState(
            last_activation=None,
            total_activations=0,
            total_trades=0,
            wins=0,
            losses=0,
            total_pnl_pct=0.0,
        )
        self._save_state()
        logger.info("Engine D state reset")


# Singleton accessor
_controller_instance: Optional[EngineDController] = None


def get_engine_d_controller() -> EngineDController:
    """Get or create the Engine D controller singleton."""
    global _controller_instance
    if _controller_instance is None:
        _controller_instance = EngineDController()
    return _controller_instance


# Convenience functions
def check_engine_d_activation(atr_multiplier: float, confidence: float = 0.70) -> dict:
    """Quick check if Engine D can activate."""
    return get_engine_d_controller().check_activation(atr_multiplier, confidence)


def record_engine_d_activation():
    """Record Engine D activation."""
    get_engine_d_controller().record_activation()


def record_engine_d_trade(pnl_pct: float, won: bool):
    """Record Engine D trade result."""
    get_engine_d_controller().record_trade_result(pnl_pct, won)
