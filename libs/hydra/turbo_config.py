"""
HYDRA 4.0 - Turbo Configuration

Centralized configuration for HYDRA 4.0 turbo mode.
Supports FTMO prep mode with higher exploration rate.
"""

import logging
from dataclasses import dataclass
from typing import Optional
import os

logger = logging.getLogger(__name__)


@dataclass
class TurboConfig:
    """
    Configuration for HYDRA 4.0 turbo mode.

    Modes:
    - FTMO_PREP_MODE: Higher exploration (50%), more aggressive strategy discovery
    - NORMAL_MODE: Conservative exploration (20%), focus on proven strategies
    """

    # Mode toggle
    FTMO_PREP_MODE: bool = True

    # Exploration rates
    NORMAL_EXPLORE_RATE: float = 0.20  # 20% exploration in normal mode
    FTMO_PREP_EXPLORE_RATE: float = 0.50  # 50% exploration in FTMO prep

    # Batch generation settings
    BATCH_SIZE_NORMAL: int = 100
    BATCH_SIZE_FTMO_PREP: int = 250
    MAX_STRATEGIES_PER_DAY: int = 1000

    # Breeding settings
    BREED_FREQUENCY_NORMAL: int = 24  # Breed every 24 hours
    BREED_FREQUENCY_FTMO_PREP: int = 6  # Breed every 6 hours

    # Tournament settings
    MIN_RANK_FOR_PAPER: float = 50.0  # Min rank score to enter paper trading
    MIN_RANK_FOR_LIVE: float = 70.0  # Min rank score to consider for live

    # Gate thresholds
    PAPER_GATE_MIN_TRADES: int = 5
    PAPER_GATE_MIN_WR: float = 0.65
    CONFIDENCE_GATE_THRESHOLD: float = 0.80

    # Evolution settings
    EVOLUTION_HOUR: int = 0  # Midnight UTC
    MAX_PREVENTION_RULES: int = 50

    # Cost controls
    MAX_DAILY_API_COST: float = 10.0  # $10/day max for strategy generation
    COST_PER_BATCH: float = 1.50  # Estimated cost per 1000 strategies

    def get_explore_rate(self) -> float:
        """Get current exploration rate based on mode."""
        if self.FTMO_PREP_MODE:
            return self.FTMO_PREP_EXPLORE_RATE
        return self.NORMAL_EXPLORE_RATE

    def get_batch_size(self) -> int:
        """Get batch size based on mode."""
        if self.FTMO_PREP_MODE:
            return self.BATCH_SIZE_FTMO_PREP
        return self.BATCH_SIZE_NORMAL

    def get_breed_frequency(self) -> int:
        """Get breeding frequency in hours based on mode."""
        if self.FTMO_PREP_MODE:
            return self.BREED_FREQUENCY_FTMO_PREP
        return self.BREED_FREQUENCY_NORMAL

    def get_max_batches_per_day(self) -> int:
        """Calculate max batches per day based on cost limit."""
        return int(self.MAX_DAILY_API_COST / self.COST_PER_BATCH)

    def should_explore(self) -> bool:
        """Random check if should explore vs exploit."""
        import random
        return random.random() < self.get_explore_rate()

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "ftmo_prep_mode": self.FTMO_PREP_MODE,
            "explore_rate": self.get_explore_rate(),
            "batch_size": self.get_batch_size(),
            "breed_frequency_hours": self.get_breed_frequency(),
            "max_batches_per_day": self.get_max_batches_per_day(),
            "paper_gate_min_trades": self.PAPER_GATE_MIN_TRADES,
            "paper_gate_min_wr": self.PAPER_GATE_MIN_WR,
            "confidence_threshold": self.CONFIDENCE_GATE_THRESHOLD,
        }

    @classmethod
    def from_env(cls) -> "TurboConfig":
        """Create config from environment variables."""
        config = cls()

        # Override from env
        if os.getenv("HYDRA_FTMO_PREP_MODE"):
            config.FTMO_PREP_MODE = os.getenv("HYDRA_FTMO_PREP_MODE", "true").lower() == "true"

        if os.getenv("HYDRA_EXPLORE_RATE"):
            config.NORMAL_EXPLORE_RATE = float(os.getenv("HYDRA_EXPLORE_RATE", "0.20"))

        if os.getenv("HYDRA_BATCH_SIZE"):
            config.BATCH_SIZE_NORMAL = int(os.getenv("HYDRA_BATCH_SIZE", "100"))

        if os.getenv("HYDRA_MAX_DAILY_COST"):
            config.MAX_DAILY_API_COST = float(os.getenv("HYDRA_MAX_DAILY_COST", "10.0"))

        return config


# Singleton instance
_config_instance: Optional[TurboConfig] = None


def get_turbo_config() -> TurboConfig:
    """Get or create the turbo config singleton."""
    global _config_instance
    if _config_instance is None:
        _config_instance = TurboConfig.from_env()
        logger.info(f"[TurboConfig] Mode: {'FTMO_PREP' if _config_instance.FTMO_PREP_MODE else 'NORMAL'}")
        logger.info(f"[TurboConfig] Explore rate: {_config_instance.get_explore_rate()*100:.0f}%")
    return _config_instance


def set_ftmo_prep_mode(enabled: bool):
    """Toggle FTMO prep mode."""
    config = get_turbo_config()
    config.FTMO_PREP_MODE = enabled
    logger.info(f"[TurboConfig] FTMO prep mode: {'ENABLED' if enabled else 'DISABLED'}")
