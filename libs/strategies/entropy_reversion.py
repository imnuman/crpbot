"""
Entropy-Based Mean Reversion Strategy - MATH ONLY

Uses Shannon entropy + Hurst exponent to detect overextended markets
and trade reversions back to the mean.

Strategy:
- Low entropy (< 0.7) + mean-reverting Hurst (< 0.45) = FADE the move
- High entropy (> 0.85) = Avoid (unpredictable market)
- Monte Carlo VaR for position sizing

Entry/Exit:
- Stop Loss: 3% (tighter for mean reversion)
- Take Profit: 6% (1:2 R:R)
- Fast execution: <1ms decisions
"""

from typing import Optional
from dataclasses import dataclass

@dataclass
class EntropySignal:
    """Entropy-based reversion signal"""
    direction: str  # "long", "short", "hold"
    confidence: float  # 0.0 to 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    reasoning: str


class EntropyReversionStrategy:
    """Shannon entropy + mean reversion strategy"""

    def __init__(self):
        self.stop_loss_pct = 0.03  # 3% (tighter for reversions)
        self.take_profit_pct = 0.06  # 6% (1:2 R:R)

    def generate_signal(
        self,
        current_price: float,
        hurst: float,
        kalman_momentum: float,
        entropy: float,
        var_95: float,  # Value at Risk (95%)
        regime: str
    ) -> EntropySignal:
        """
        Generate signal based on entropy + mean reversion

        Args:
            current_price: Current market price
            hurst: Hurst exponent (<0.5 = mean-reverting)
            kalman_momentum: Kalman momentum (direction of move to fade)
            entropy: Shannon entropy (low = predictable)
            var_95: Monte Carlo VaR at 95% (risk measure)
            regime: Current Markov regime

        Returns:
            EntropySignal with direction and prices
        """

        direction = "hold"
        confidence = 0.5
        reasoning = "Insufficient mean reversion signal"

        # High entropy = unpredictable, skip
        if entropy > 0.85:
            reasoning = f"High entropy {entropy:.3f} - market too random"
            return self._build_signal(direction, confidence, current_price, reasoning)

        # RULE 1: Strong mean reversion opportunity
        # Low entropy + mean-reverting Hurst + momentum overextended
        if entropy < 0.65 and hurst < 0.40 and abs(kalman_momentum) > 20:
            # Fade the move: if momentum is up, go SHORT (expect reversal down)
            direction = "short" if kalman_momentum > 0 else "long"
            confidence = min(0.40 + (0.50 - hurst) * 1.5 + (0.85 - entropy) * 0.5, 0.80)
            reasoning = f"Strong mean reversion: entropy {entropy:.3f}, Hurst {hurst:.3f}, momentum {kalman_momentum:+.1f} (FADING)"

        # RULE 2: Moderate mean reversion
        elif entropy < 0.70 and hurst < 0.45 and abs(kalman_momentum) > 10:
            direction = "short" if kalman_momentum > 0 else "long"
            confidence = min(0.30 + (0.50 - hurst) * 1.2 + (0.85 - entropy) * 0.3, 0.65)
            reasoning = f"Moderate mean reversion: entropy {entropy:.3f}, Hurst {hurst:.3f}, momentum {kalman_momentum:+.1f}"

        # RULE 3: Weak mean reversion (small moves)
        elif entropy < 0.75 and hurst < 0.48 and abs(kalman_momentum) > 5:
            direction = "short" if kalman_momentum > 0 else "long"
            confidence = 0.35
            reasoning = f"Weak mean reversion: entropy {entropy:.3f}, Hurst {hurst:.3f}"

        # RULE 4: VaR-based risk adjustment
        # If VaR is too high (risky market), reduce confidence
        if abs(var_95) > 5.0:  # VaR > 5%
            confidence *= 0.8
            reasoning += f" | High VaR {var_95:.1f}% (reduced confidence)"

        return self._build_signal(direction, confidence, current_price, reasoning)

    def _build_signal(
        self,
        direction: str,
        confidence: float,
        current_price: float,
        reasoning: str
    ) -> EntropySignal:
        """Build EntropySignal with prices"""

        if direction == "long":
            stop_loss = current_price * (1 - self.stop_loss_pct)
            take_profit = current_price * (1 + self.take_profit_pct)
        elif direction == "short":
            stop_loss = current_price * (1 + self.stop_loss_pct)
            take_profit = current_price * (1 - self.take_profit_pct)
        else:  # hold
            stop_loss = 0
            take_profit = 0

        return EntropySignal(
            direction=direction,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning
        )
