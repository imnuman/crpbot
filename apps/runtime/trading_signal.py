"""Signal generation and formatting."""
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from loguru import logger


@dataclass
class Signal:
    """Trading signal data structure."""

    pair: str
    direction: str  # 'long' or 'short'
    entry_price: float
    tp_price: float
    sl_price: float
    confidence: float
    tier: str  # 'high', 'medium', 'low'
    rr_ratio: float  # Risk:Reward ratio
    timestamp: datetime
    latency_ms: float = 0.0
    spread_bps: float = 0.0
    slippage_bps: float = 0.0
    mode: str = "dryrun"  # 'dryrun' or 'live'

    def to_dict(self) -> dict[str, Any]:
        """Convert signal to dictionary."""
        return {
            "pair": self.pair,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "tp_price": self.tp_price,
            "sl_price": self.sl_price,
            "confidence": self.confidence,
            "tier": self.tier,
            "rr_ratio": self.rr_ratio,
            "timestamp": self.timestamp.isoformat(),
            "latency_ms": self.latency_ms,
            "spread_bps": self.spread_bps,
            "slippage_bps": self.slippage_bps,
            "mode": self.mode,
        }

    def format_message(self) -> str:
        """
        Format signal as human-readable message.

        Returns:
            Formatted message string
        """
        direction_emoji = "🟢" if self.direction == "long" else "🔴"
        tier_emoji = {"high": "🔥", "medium": "⚡", "low": "💡"}.get(self.tier, "📊")

        message = (
            f"{tier_emoji} {self.tier.upper()} {self.direction.upper()} Signal\n"
            f"{direction_emoji} Pair: {self.pair}\n"
            f"💵 Entry: ${self.entry_price:,.2f}\n"
            f"🎯 TP: ${self.tp_price:,.2f}\n"
            f"🛑 SL: ${self.sl_price:,.2f}\n"
            f"📊 R:R: {self.rr_ratio:.2f}\n"
            f"🎲 Confidence: {self.confidence:.1%}\n"
            f"⏱️  Latency: {self.latency_ms:.0f}ms"
        )

        return message


def determine_tier(confidence: float) -> str:
    """
    Determine confidence tier from confidence score.

    Args:
        confidence: Confidence score (0.0-1.0)

    Returns:
        Tier: 'high', 'medium', or 'low'
    """
    if confidence >= 0.75:
        return "high"
    elif confidence >= 0.65:
        return "medium"
    else:
        return "low"


def calculate_tp_sl(
    entry_price: float, direction: str, risk_reward_ratio: float = 2.0, risk_pct: float = 0.01
) -> tuple[float, float]:
    """
    Calculate take-profit and stop-loss prices.

    Args:
        entry_price: Entry price
        direction: 'long' or 'short'
        risk_reward_ratio: Risk:Reward ratio (default: 2.0)
        risk_pct: Risk percentage (default: 1%)

    Returns:
        Tuple of (tp_price, sl_price)
    """
    risk_amount = entry_price * risk_pct

    if direction == "long":
        sl_price = entry_price - risk_amount
        tp_price = entry_price + (risk_amount * risk_reward_ratio)
    else:  # short
        sl_price = entry_price + risk_amount
        tp_price = entry_price - (risk_amount * risk_reward_ratio)

    return tp_price, sl_price


def generate_signal(
    pair: str,
    direction: str,
    entry_price: float,
    confidence: float,
    mode: str = "dryrun",
    risk_reward_ratio: float = 2.0,
    latency_ms: float = 0.0,
    spread_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> Signal:
    """
    Generate a trading signal.

    Args:
        pair: Trading pair (e.g., 'BTC-USD')
        direction: 'long' or 'short'
        entry_price: Entry price
        confidence: Confidence score (0.0-1.0)
        mode: Mode tag ('dryrun' or 'live')
        risk_reward_ratio: Risk:Reward ratio
        latency_ms: Decision latency in milliseconds
        spread_bps: Spread in basis points
        slippage_bps: Slippage in basis points

    Returns:
        Signal object
    """
    tier = determine_tier(confidence)
    tp_price, sl_price = calculate_tp_sl(entry_price, direction, risk_reward_ratio)

    signal = Signal(
        pair=pair,
        direction=direction,
        entry_price=entry_price,
        tp_price=tp_price,
        sl_price=sl_price,
        confidence=confidence,
        tier=tier,
        rr_ratio=risk_reward_ratio,
        timestamp=datetime.utcnow(),
        latency_ms=latency_ms,
        spread_bps=spread_bps,
        slippage_bps=slippage_bps,
        mode=mode,
    )

    logger.info(f"Generated {mode} signal: {pair} {direction} @ {entry_price:.2f} (confidence: {confidence:.1%}, tier: {tier})")

    return signal

