"""
Market Regime-Based Trading Strategy

Adjusts trading behavior based on detected market regime:
- TRENDING markets: Follow trend (LONG in uptrend, SHORT in downtrend)
- RANGING markets: Mean reversion or wait (reduce position sizes)
- HIGH VOL: Reduce risk, widen stops
- LOW VOL: Normal operation
"""
import logging
from typing import Dict, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class RegimeStrategy(Enum):
    """Trading strategies for different market regimes"""
    TREND_FOLLOWING_LONG = "trend_following_long"
    TREND_FOLLOWING_SHORT = "trend_following_short"
    MEAN_REVERSION = "mean_reversion"
    RANGE_TRADING = "range_trading"
    MOMENTUM_TRADING = "momentum_trading"
    WAIT_FOR_BREAKOUT = "wait_for_breakout"
    REDUCE_RISK = "reduce_risk"


class RegimeStrategyManager:
    """
    Manage trading strategy based on market regime

    Integrates with Markov Chain regime detector to:
    1. Filter signals inappropriate for current regime
    2. Adjust position sizes based on regime risk
    3. Modify exit strategies per regime
    """

    def __init__(self):
        """Initialize regime strategy manager"""
        # Regime-specific trading rules
        self.regime_rules = {
            'Bull Trend': {
                'allowed_directions': ['long'],
                'position_size_multiplier': 1.0,
                'confidence_threshold': 0.65,
                'stop_loss_multiplier': 1.0,
                'strategy': RegimeStrategy.TREND_FOLLOWING_LONG,
                'description': 'Follow uptrend - LONG positions preferred'
            },
            'Bear Trend': {
                'allowed_directions': ['short'],
                'position_size_multiplier': 1.0,
                'confidence_threshold': 0.65,
                'stop_loss_multiplier': 1.0,
                'strategy': RegimeStrategy.TREND_FOLLOWING_SHORT,
                'description': 'Follow downtrend - SHORT positions preferred'
            },
            'High Volatility Range': {
                'allowed_directions': ['long', 'short'],
                'position_size_multiplier': 0.5,  # 50% size due to chop
                'confidence_threshold': 0.75,     # Higher confidence required
                'stop_loss_multiplier': 1.5,      # Wider stops for volatility
                'strategy': RegimeStrategy.MEAN_REVERSION,
                'description': 'Choppy market - reduce size, widen stops'
            },
            'Low Volatility Range': {
                'allowed_directions': ['hold'],   # Wait for breakout
                'position_size_multiplier': 0.3,
                'confidence_threshold': 0.80,     # Very high confidence only
                'stop_loss_multiplier': 0.8,      # Tighter stops
                'strategy': RegimeStrategy.WAIT_FOR_BREAKOUT,
                'description': 'Low volatility - wait for clear breakout'
            },
            'Breakout': {
                'allowed_directions': ['long', 'short'],
                'position_size_multiplier': 1.2,  # 120% size for breakouts
                'confidence_threshold': 0.65,
                'stop_loss_multiplier': 1.2,      # Wider stops for momentum
                'strategy': RegimeStrategy.MOMENTUM_TRADING,
                'description': 'Breakout detected - momentum trading'
            },
            'Consolidation': {
                'allowed_directions': ['long', 'short'],
                'position_size_multiplier': 0.7,
                'confidence_threshold': 0.70,
                'stop_loss_multiplier': 0.9,
                'strategy': RegimeStrategy.RANGE_TRADING,
                'description': 'Consolidation - range-bound trading'
            }
        }

    def get_regime_rules(self, regime_name: str) -> Dict:
        """
        Get trading rules for current regime

        Args:
            regime_name: Name of market regime (from Markov detector)

        Returns:
            Dict with trading rules for this regime
        """
        return self.regime_rules.get(
            regime_name,
            self.regime_rules['High Volatility Range']  # Default to conservative
        )

    def filter_signal(
        self,
        signal_direction: str,
        signal_confidence: float,
        regime_name: str
    ) -> tuple[bool, Optional[str]]:
        """
        Check if signal is appropriate for current regime

        Args:
            signal_direction: 'long', 'short', or 'hold'
            signal_confidence: Signal confidence (0-1)
            regime_name: Current market regime

        Returns:
            Tuple of (is_allowed, rejection_reason)
        """
        rules = self.get_regime_rules(regime_name)

        # Check if direction is allowed in this regime
        if signal_direction not in rules['allowed_directions']:
            reason = f"Regime '{regime_name}' doesn't allow {signal_direction} signals"
            logger.info(f"Signal filtered: {reason}")
            return False, reason

        # Check if confidence meets regime threshold
        if signal_confidence < rules['confidence_threshold']:
            reason = f"Confidence {signal_confidence:.2f} < threshold {rules['confidence_threshold']:.2f} for regime '{regime_name}'"
            logger.info(f"Signal filtered: {reason}")
            return False, reason

        # Signal passes all filters
        return True, None

    def adjust_position_size(
        self,
        base_position_size: float,
        regime_name: str
    ) -> float:
        """
        Adjust position size based on regime

        Args:
            base_position_size: Base position size (e.g., from Kelly)
            regime_name: Current market regime

        Returns:
            Adjusted position size
        """
        rules = self.get_regime_rules(regime_name)
        multiplier = rules['position_size_multiplier']

        adjusted_size = base_position_size * multiplier

        logger.debug(f"Position size adjusted: {base_position_size:.2%} → {adjusted_size:.2%} (regime: {regime_name})")

        return adjusted_size

    def adjust_stop_loss(
        self,
        base_stop_distance: float,
        regime_name: str
    ) -> float:
        """
        Adjust stop-loss distance based on regime volatility

        Args:
            base_stop_distance: Base stop-loss distance (as % of price)
            regime_name: Current market regime

        Returns:
            Adjusted stop-loss distance
        """
        rules = self.get_regime_rules(regime_name)
        multiplier = rules['stop_loss_multiplier']

        adjusted_stop = base_stop_distance * multiplier

        logger.debug(f"Stop-loss adjusted: {base_stop_distance:.2%} → {adjusted_stop:.2%} (regime: {regime_name})")

        return adjusted_stop

    def get_regime_recommendation(self, regime_name: str) -> str:
        """
        Get human-readable recommendation for current regime

        Args:
            regime_name: Current market regime

        Returns:
            Recommendation string
        """
        rules = self.get_regime_rules(regime_name)
        return rules['description']

    def print_regime_analysis(self, regime_name: str, regime_confidence: float):
        """Print regime analysis and trading recommendations"""
        print("\n" + "="*70)
        print("MARKET REGIME ANALYSIS")
        print("="*70)
        print(f"\nCurrent Regime: {regime_name}")
        print(f"Confidence: {regime_confidence*100:.1f}%")

        rules = self.get_regime_rules(regime_name)

        print(f"\nStrategy: {rules['strategy'].value}")
        print(f"Description: {rules['description']}")

        print(f"\nTrading Rules:")
        print(f"  Allowed Directions: {', '.join(rules['allowed_directions']).upper()}")
        print(f"  Position Size Multiplier: {rules['position_size_multiplier']*100:.0f}%")
        print(f"  Min Confidence Threshold: {rules['confidence_threshold']*100:.0f}%")
        print(f"  Stop-Loss Multiplier: {rules['stop_loss_multiplier']*100:.0f}%")

        print("="*70)


# Example usage
if __name__ == "__main__":
    manager = RegimeStrategyManager()

    # Test different regimes
    test_regimes = [
        ('Bull Trend', 0.85),
        ('Bear Trend', 0.75),
        ('High Volatility Range', 0.65),
        ('Low Volatility Range', 0.70)
    ]

    for regime_name, regime_conf in test_regimes:
        manager.print_regime_analysis(regime_name, regime_conf)

        # Test signal filtering
        print(f"\nTesting signals for {regime_name}:")

        test_signals = [
            ('long', 0.75),
            ('short', 0.70),
            ('long', 0.60)  # Low confidence
        ]

        for direction, confidence in test_signals:
            is_allowed, reason = manager.filter_signal(direction, confidence, regime_name)
            status = "✅ ALLOWED" if is_allowed else "❌ BLOCKED"
            print(f"  {direction.upper()} @ {confidence:.0%}: {status}")
            if reason:
                print(f"    Reason: {reason}")

        # Test position sizing
        base_size = 0.10  # 10% base
        adjusted = manager.adjust_position_size(base_size, regime_name)
        print(f"\nPosition Size: {base_size:.0%} → {adjusted:.0%}")

        # Test stop-loss adjustment
        base_stop = 0.02  # 2% stop
        adjusted_stop = manager.adjust_stop_loss(base_stop, regime_name)
        print(f"Stop-Loss: {base_stop:.1%} → {adjusted_stop:.1%}")

        print("\n")
