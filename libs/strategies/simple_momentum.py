"""
Simple Momentum Strategy - NO LLM, PURE MATH

Generates ACTUAL LONG/SHORT signals based on clear mathematical rules.
NO conservative LLM that refuses to trade.

Strategy:
- Hurst > 0.55 + Kalman momentum > 0 = BUY
- Hurst < 0.45 + Kalman momentum < 0 = SELL
- Otherwise = HOLD

Entry/Exit:
- Stop Loss: 4% (widened for crypto)
- Take Profit: 8% (1:2 R:R)
- Fast execution: Make decisions in <1ms
"""

from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class SimpleSignal:
    """Simple trading signal"""
    direction: str  # "long", "short", "hold"
    confidence: float  # 0.0 to 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    reasoning: str


class SimpleMomentumStrategy:
    """Pure mathematical momentum strategy - NO LLM delays"""

    def __init__(self):
        self.stop_loss_pct = 0.04  # 4%
        self.take_profit_pct = 0.08  # 8% (1:2 R:R)

    def generate_signal(
        self,
        current_price: float,
        hurst: float,
        kalman_momentum: float,
        entropy: float,
        regime: str
    ) -> SimpleSignal:
        """
        Generate signal based on pure math rules

        Args:
            current_price: Current market price
            hurst: Hurst exponent (0.5 = random, >0.5 = trending, <0.5 = mean-reverting)
            kalman_momentum: Kalman filter momentum
            entropy: Shannon entropy
            regime: Current Markov regime

        Returns:
            SimpleSignal with direction and prices
        """

        # Default to HOLD
        direction = "hold"
        confidence = 0.5
        reasoning = "Insufficient momentum"

        # RULE 1: Strong uptrend
        if hurst > 0.55 and kalman_momentum > 10:
            direction = "long"
            confidence = min(0.3 + (hurst - 0.5) * 2 + (kalman_momentum / 100), 0.85)
            reasoning = f"Strong uptrend: Hurst {hurst:.3f}, Momentum +{kalman_momentum:.1f}"

        # RULE 2: Moderate uptrend
        elif hurst > 0.52 and kalman_momentum > 5:
            direction = "long"
            confidence = min(0.25 + (hurst - 0.5) * 1.5 + (kalman_momentum / 150), 0.70)
            reasoning = f"Moderate uptrend: Hurst {hurst:.3f}, Momentum +{kalman_momentum:.1f}"

        # RULE 3: Weak uptrend (still tradeable with 4% SL)
        elif hurst > 0.52 or kalman_momentum > 15:
            direction = "long"
            confidence = 0.30
            reasoning = f"Weak uptrend: Hurst {hurst:.3f}, Momentum {kalman_momentum:.1f}"

        # RULE 4: Strong downtrend
        elif hurst < 0.45 and kalman_momentum < -10:
            direction = "short"
            confidence = min(0.3 + (0.5 - hurst) * 2 + (abs(kalman_momentum) / 100), 0.85)
            reasoning = f"Strong downtrend: Hurst {hurst:.3f}, Momentum {kalman_momentum:.1f}"

        # RULE 5: Moderate downtrend
        elif hurst < 0.48 and kalman_momentum < -5:
            direction = "short"
            confidence = min(0.25 + (0.5 - hurst) * 1.5 + (abs(kalman_momentum) / 150), 0.70)
            reasoning = f"Moderate downtrend: Hurst {hurst:.3f}, Momentum {kalman_momentum:.1f}"

        # RULE 6: Weak downtrend
        elif hurst < 0.48 or kalman_momentum < -15:
            direction = "short"
            confidence = 0.30
            reasoning = f"Weak downtrend: Hurst {hurst:.3f}, Momentum {kalman_momentum:.1f}"

        # Calculate entry/SL/TP
        entry_price = current_price

        if direction == "long":
            stop_loss = current_price * (1 - self.stop_loss_pct)
            take_profit = current_price * (1 + self.take_profit_pct)
        elif direction == "short":
            stop_loss = current_price * (1 + self.stop_loss_pct)
            take_profit = current_price * (1 - self.take_profit_pct)
        else:  # hold
            stop_loss = 0
            take_profit = 0

        return SimpleSignal(
            direction=direction,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning
        )
