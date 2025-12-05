"""
HYDRA 3.0 - Engine Specialization

Each engine has ONE specialty only:
- Engine A (DeepSeek): ONLY liquidation cascades ($20M+ trigger)
- Engine B (Claude): ONLY funding rate extremes (>0.5%)
- Engine C (Grok): ONLY orderbook imbalance (>2.5:1)
- Engine D (Gemini): ONLY regime transitions (ATR 2× expansion)

Engines MUST reject any trade outside their specialty.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class Specialty(Enum):
    """Engine specialties - one per engine."""
    LIQUIDATION_CASCADE = "liquidation_cascade"   # Engine A
    FUNDING_EXTREME = "funding_extreme"           # Engine B
    ORDERBOOK_IMBALANCE = "orderbook_imbalance"   # Engine C
    REGIME_TRANSITION = "regime_transition"       # Engine D


@dataclass
class SpecialtyConfig:
    """Configuration for a specialty."""
    specialty: Specialty
    engine_id: str
    description: str

    # Trigger thresholds
    trigger_threshold: float
    trigger_unit: str

    # Validation rules
    min_confidence: float = 0.70  # 70% minimum

    def to_dict(self) -> dict:
        return {
            "specialty": self.specialty.value,
            "engine_id": self.engine_id,
            "description": self.description,
            "trigger_threshold": self.trigger_threshold,
            "trigger_unit": self.trigger_unit,
            "min_confidence": self.min_confidence,
        }


# Engine specialty assignments
# NOTE: Thresholds lowered from original extreme values to trigger during normal trading
ENGINE_SPECIALTIES = {
    "A": SpecialtyConfig(
        specialty=Specialty.LIQUIDATION_CASCADE,
        engine_id="A",
        description="Liquidation cascades - forced liquidations triggering price moves",
        trigger_threshold=1_000_000,  # $1M+ in liquidations (was $20M)
        trigger_unit="USD",
    ),
    "B": SpecialtyConfig(
        specialty=Specialty.FUNDING_EXTREME,
        engine_id="B",
        description="Funding rate extremes - crowded trades about to reverse",
        trigger_threshold=0.05,  # >0.05% funding rate (was 0.5%)
        trigger_unit="percent",
    ),
    "C": SpecialtyConfig(
        specialty=Specialty.ORDERBOOK_IMBALANCE,
        engine_id="C",
        description="Orderbook imbalance - one-sided pressure",
        trigger_threshold=1.5,  # 1.5:1 bid/ask ratio (was 2.5:1)
        trigger_unit="ratio",
    ),
    "D": SpecialtyConfig(
        specialty=Specialty.REGIME_TRANSITION,
        engine_id="D",
        description="Regime transitions - volatility changes signaling trend shifts",
        trigger_threshold=1.2,  # ATR 1.2× expansion (was 2×)
        trigger_unit="multiplier",
    ),
}


class SpecialtyValidator:
    """
    Validates that engines only trade their specialty.
    """

    def __init__(self):
        self.specialties = ENGINE_SPECIALTIES

    def get_specialty(self, engine_id: str) -> Optional[SpecialtyConfig]:
        """Get specialty config for an engine."""
        return self.specialties.get(engine_id.upper())

    def validate_trigger(
        self,
        engine_id: str,
        trigger_type: str,
        trigger_value: float,
        confidence: float
    ) -> dict:
        """
        Validate if a trade trigger matches engine's specialty.

        Args:
            engine_id: Engine ID (A, B, C, D)
            trigger_type: Type of trigger detected
            trigger_value: Value of the trigger
            confidence: Engine's confidence (0-1)

        Returns:
            Validation result dict
        """
        result = {
            "valid": False,
            "engine_id": engine_id,
            "trigger_type": trigger_type,
            "trigger_value": trigger_value,
            "confidence": confidence,
            "reason": "",
        }

        specialty = self.get_specialty(engine_id)
        if not specialty:
            result["reason"] = f"Unknown engine: {engine_id}"
            return result

        # Check confidence threshold
        if confidence < specialty.min_confidence:
            result["reason"] = f"Confidence {confidence:.0%} below minimum {specialty.min_confidence:.0%}"
            return result

        # Check trigger matches specialty
        expected_trigger = specialty.specialty.value
        if trigger_type != expected_trigger:
            result["reason"] = f"Engine {engine_id} specialty is {expected_trigger}, not {trigger_type}"
            return result

        # Check trigger threshold
        if not self._check_threshold(specialty, trigger_value):
            result["reason"] = f"Trigger value {trigger_value} below threshold {specialty.trigger_threshold}"
            return result

        # All checks passed
        result["valid"] = True
        result["reason"] = "Trigger matches specialty and meets thresholds"
        return result

    def _check_threshold(self, specialty: SpecialtyConfig, value: float) -> bool:
        """Check if trigger value meets threshold."""
        if specialty.specialty == Specialty.LIQUIDATION_CASCADE:
            return value >= specialty.trigger_threshold  # $20M+
        elif specialty.specialty == Specialty.FUNDING_EXTREME:
            return abs(value) >= specialty.trigger_threshold  # ±0.5%
        elif specialty.specialty == Specialty.ORDERBOOK_IMBALANCE:
            return value >= specialty.trigger_threshold or value <= (1 / specialty.trigger_threshold)  # 2.5:1 either way
        elif specialty.specialty == Specialty.REGIME_TRANSITION:
            return value >= specialty.trigger_threshold  # 2× ATR
        return False

    def check_liquidation_cascade(self, total_liquidations_usd: float) -> bool:
        """Check if liquidation cascade trigger is met."""
        return total_liquidations_usd >= ENGINE_SPECIALTIES["A"].trigger_threshold

    def check_funding_extreme(self, funding_rate_pct: float) -> bool:
        """Check if funding rate extreme trigger is met."""
        return abs(funding_rate_pct) >= ENGINE_SPECIALTIES["B"].trigger_threshold

    def check_orderbook_imbalance(self, bid_ask_ratio: float) -> bool:
        """Check if orderbook imbalance trigger is met."""
        threshold = ENGINE_SPECIALTIES["C"].trigger_threshold
        return bid_ask_ratio >= threshold or bid_ask_ratio <= (1 / threshold)

    def check_regime_transition(self, atr_multiplier: float) -> bool:
        """Check if regime transition trigger is met."""
        return atr_multiplier >= ENGINE_SPECIALTIES["D"].trigger_threshold

    def get_active_triggers(self, market_data: Dict[str, Any]) -> Dict[str, bool]:
        """
        Check which specialty triggers are currently active.

        Args:
            market_data: Dict containing:
                - liquidation_total_usd: Total liquidations in USD
                - funding_rate_pct: Current funding rate %
                - bid_ask_ratio: Order book bid/ask ratio
                - atr_multiplier: Current ATR vs baseline

        Returns:
            Dict of engine_id -> trigger_active
        """
        return {
            "A": self.check_liquidation_cascade(market_data.get("liquidation_total_usd", 0)),
            "B": self.check_funding_extreme(market_data.get("funding_rate_pct", 0)),
            "C": self.check_orderbook_imbalance(market_data.get("bid_ask_ratio", 1.0)),
            "D": self.check_regime_transition(market_data.get("atr_multiplier", 1.0)),
        }

    def format_specialties(self) -> str:
        """Format specialties for display."""
        lines = ["=== Engine Specialties ==="]
        for engine_id, config in self.specialties.items():
            lines.append(f"Engine {engine_id}: {config.specialty.value}")
            lines.append(f"  {config.description}")
            lines.append(f"  Trigger: {config.trigger_threshold} {config.trigger_unit}")
            lines.append(f"  Min confidence: {config.min_confidence:.0%}")
        return "\n".join(lines)


# Singleton
_validator_instance: Optional[SpecialtyValidator] = None


def get_specialty_validator() -> SpecialtyValidator:
    """Get or create the specialty validator singleton."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = SpecialtyValidator()
    return _validator_instance


# Convenience functions
def validate_engine_trade(
    engine_id: str,
    trigger_type: str,
    trigger_value: float,
    confidence: float
) -> dict:
    """Quick validation of engine trade."""
    return get_specialty_validator().validate_trigger(
        engine_id, trigger_type, trigger_value, confidence
    )


def get_engine_specialty(engine_id: str) -> Optional[SpecialtyConfig]:
    """Get specialty for an engine."""
    return get_specialty_validator().get_specialty(engine_id)
