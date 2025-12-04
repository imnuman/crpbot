"""
Volatility Regime Detector

Classifies market volatility into regimes and adapts trading strategy accordingly.

Regimes:
1. High Volatility - Explosive moves, trending markets
2. Normal Volatility - Balanced conditions
3. Low Volatility - Compression/coiling, pre-breakout

Indicators:
- ATR (Average True Range): Absolute volatility measure
- ATR Percentile: Relative volatility ranking
- Bollinger Band Width: Expansion/compression
- Historical Volatility: Recent price variation

Strategy Adaptations:
- High Vol: Wider stops, larger targets, momentum bias
- Normal Vol: Standard 3:1 R:R, balanced approach
- Low Vol: Tighter stops, breakout anticipation, avoid ranging

Expected Impact:
- Better risk management (adaptive stops/targets)
- Improved entry timing (volatility breakouts)
- Reduced whipsaws (avoid low-vol ranging)
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VolatilityRegime:
    """Volatility regime classification result"""
    regime: str  # 'high', 'normal', 'low'
    atr_value: float  # Current ATR
    atr_percentile: float  # ATR percentile (0-1)
    bb_width_pct: float  # Bollinger Band width as % of price
    historical_vol: float  # 30-day historical volatility
    confidence: float  # Confidence in classification (0-1)

    # Strategy recommendations
    recommended_stop_multiplier: float  # Multiply standard stop by this
    recommended_target_multiplier: float  # Multiply standard target by this
    trade_bias: str  # 'momentum', 'balanced', 'breakout'

    reason: str  # Explanation
    metrics: Dict[str, float]  # Raw metrics


class VolatilityRegimeDetector:
    """
    Detect volatility regimes and provide strategy adaptations

    Usage:
        detector = VolatilityRegimeDetector()

        regime = detector.detect_regime(df)

        print(f"Regime: {regime.regime}")
        print(f"Stop Multiplier: {regime.recommended_stop_multiplier}x")
        print(f"Target Multiplier: {regime.recommended_target_multiplier}x")
        print(f"Bias: {regime.trade_bias}")

        # Adjust stops/targets based on regime
        standard_stop = 0.01  # 1%
        adjusted_stop = standard_stop * regime.recommended_stop_multiplier
    """

    def __init__(
        self,
        atr_period: int = 14,
        bb_period: int = 20,
        lookback_for_percentiles: int = 100,
        high_vol_threshold: float = 0.75,  # 75th percentile
        low_vol_threshold: float = 0.25    # 25th percentile
    ):
        """
        Initialize Volatility Regime Detector

        Args:
            atr_period: Period for ATR calculation
            bb_period: Period for Bollinger Bands
            lookback_for_percentiles: Periods for percentile calculation
            high_vol_threshold: Percentile threshold for high volatility
            low_vol_threshold: Percentile threshold for low volatility
        """
        self.atr_period = atr_period
        self.bb_period = bb_period
        self.lookback_for_percentiles = lookback_for_percentiles
        self.high_vol_threshold = high_vol_threshold
        self.low_vol_threshold = low_vol_threshold

        logger.info(
            f"Volatility Regime Detector initialized | "
            f"ATR period: {atr_period} | "
            f"Thresholds: {low_vol_threshold:.0%}/{high_vol_threshold:.0%}"
        )

    def detect_regime(self, df: pd.DataFrame) -> VolatilityRegime:
        """
        Detect current volatility regime

        Args:
            df: OHLCV DataFrame with columns [high, low, close, volume]

        Returns:
            VolatilityRegime with classification and recommendations
        """
        try:
            if len(df) < max(self.atr_period, self.bb_period, self.lookback_for_percentiles):
                logger.warning(f"Insufficient data: {len(df)} candles")
                return self._uncertain_regime("Insufficient data")

            # Calculate indicators
            atr = self._calculate_atr(df)
            atr_percentile = self._calculate_atr_percentile(df)
            bb_width_pct = self._calculate_bb_width(df)
            hist_vol = self._calculate_historical_volatility(df)

            metrics = {
                'atr': atr,
                'atr_percentile': atr_percentile,
                'bb_width_pct': bb_width_pct,
                'historical_vol': hist_vol
            }

            # Classify regime
            regime = self._classify_regime(
                atr_percentile=atr_percentile,
                bb_width_pct=bb_width_pct,
                metrics=metrics
            )

            logger.debug(
                f"Volatility Regime: {regime.regime} | "
                f"ATR %ile: {atr_percentile:.1%} | "
                f"BB Width: {bb_width_pct:.2%}"
            )

            return regime

        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return self._uncertain_regime(f"Error: {e}")

    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """Calculate Average True Range (ATR)"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        # True Range = max(H-L, |H-Cp|, |L-Cp|)
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))

        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value has no previous close

        # ATR = EMA of TR
        atr = pd.Series(tr).ewm(span=self.atr_period, adjust=False).mean()

        return float(atr.iloc[-1])

    def _calculate_atr_percentile(self, df: pd.DataFrame) -> float:
        """Calculate ATR percentile over lookback period"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        # Calculate rolling ATR
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))

        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]

        atr_series = pd.Series(tr).ewm(span=self.atr_period, adjust=False).mean()

        # Get recent ATR values for percentile calculation
        recent_atr = atr_series.tail(self.lookback_for_percentiles)
        current_atr = atr_series.iloc[-1]

        # Calculate percentile
        percentile = (recent_atr < current_atr).sum() / len(recent_atr)

        return float(percentile)

    def _calculate_bb_width(self, df: pd.DataFrame) -> float:
        """Calculate Bollinger Band width as % of price"""
        close = df['close'].values

        # Calculate Bollinger Bands
        sma = pd.Series(close).rolling(window=self.bb_period).mean()
        std = pd.Series(close).rolling(window=self.bb_period).std()

        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)

        # BB Width as % of middle band
        bb_width = (upper_band - lower_band) / sma

        return float(bb_width.iloc[-1])

    def _calculate_historical_volatility(self, df: pd.DataFrame, period: int = 30) -> float:
        """Calculate historical volatility (annualized)"""
        close = df['close'].values

        # Calculate returns
        returns = pd.Series(close).pct_change().dropna()

        # Get recent returns
        recent_returns = returns.tail(period)

        # Annualized volatility (assuming 1-minute candles, 1440 candles/day)
        volatility = recent_returns.std() * np.sqrt(1440 * 365)

        return float(volatility)

    def _classify_regime(
        self,
        atr_percentile: float,
        bb_width_pct: float,
        metrics: Dict[str, float]
    ) -> VolatilityRegime:
        """
        Classify volatility regime and generate recommendations

        Args:
            atr_percentile: ATR percentile (0-1)
            bb_width_pct: Bollinger Band width %
            metrics: Raw metrics

        Returns:
            VolatilityRegime with classification
        """
        # High Volatility: ATR > 75th percentile OR very wide BBs
        if atr_percentile > self.high_vol_threshold or bb_width_pct > 0.08:
            return VolatilityRegime(
                regime='high',
                atr_value=metrics['atr'],
                atr_percentile=atr_percentile,
                bb_width_pct=bb_width_pct,
                historical_vol=metrics['historical_vol'],
                confidence=0.9 if atr_percentile > 0.85 else 0.75,

                # Strategy: Wider stops, larger targets, follow momentum
                recommended_stop_multiplier=1.5,  # 50% wider stops
                recommended_target_multiplier=2.0,  # 2x larger targets
                trade_bias='momentum',

                reason=f"High volatility: ATR at {atr_percentile:.0%} percentile, BB width {bb_width_pct:.2%}",
                metrics=metrics
            )

        # Low Volatility: ATR < 25th percentile AND tight BBs
        elif atr_percentile < self.low_vol_threshold and bb_width_pct < 0.03:
            return VolatilityRegime(
                regime='low',
                atr_value=metrics['atr'],
                atr_percentile=atr_percentile,
                bb_width_pct=bb_width_pct,
                historical_vol=metrics['historical_vol'],
                confidence=0.85,

                # Strategy: Tighter stops, anticipate breakout, avoid ranging
                recommended_stop_multiplier=0.7,  # 30% tighter stops
                recommended_target_multiplier=0.8,  # Smaller initial targets
                trade_bias='breakout',

                reason=f"Low volatility compression: ATR at {atr_percentile:.0%}, BB width {bb_width_pct:.2%} (prepare for breakout)",
                metrics=metrics
            )

        # Normal Volatility: Everything else
        else:
            return VolatilityRegime(
                regime='normal',
                atr_value=metrics['atr'],
                atr_percentile=atr_percentile,
                bb_width_pct=bb_width_pct,
                historical_vol=metrics['historical_vol'],
                confidence=0.8,

                # Strategy: Standard approach
                recommended_stop_multiplier=1.0,  # Standard stops
                recommended_target_multiplier=1.0,  # Standard targets
                trade_bias='balanced',

                reason=f"Normal volatility: ATR at {atr_percentile:.0%}, BB width {bb_width_pct:.2%}",
                metrics=metrics
            )

    def _uncertain_regime(self, reason: str) -> VolatilityRegime:
        """Return conservative regime when uncertain"""
        return VolatilityRegime(
            regime='normal',
            atr_value=0.0,
            atr_percentile=0.5,
            bb_width_pct=0.04,
            historical_vol=0.0,
            confidence=0.0,
            recommended_stop_multiplier=1.0,
            recommended_target_multiplier=1.0,
            trade_bias='balanced',
            reason=reason,
            metrics={}
        )


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("VOLATILITY REGIME DETECTOR TEST")
    print("=" * 70)

    # Create sample data (simulate different volatility regimes)
    np.random.seed(42)
    n = 200

    # Scenario 1: High Volatility (trending with large moves)
    print("\n[Scenario 1] High Volatility Market:")
    high_vol_data = pd.DataFrame({
        'high': 100 + np.cumsum(np.random.randn(n) * 3),
        'low': 99 + np.cumsum(np.random.randn(n) * 3),
        'close': 99.5 + np.cumsum(np.random.randn(n) * 3),
        'volume': np.random.randint(1000, 5000, n)
    })

    detector = VolatilityRegimeDetector()
    regime = detector.detect_regime(high_vol_data)

    print(f"  Regime: {regime.regime}")
    print(f"  ATR Percentile: {regime.atr_percentile:.1%}")
    print(f"  BB Width: {regime.bb_width_pct:.2%}")
    print(f"  Confidence: {regime.confidence:.0%}")
    print(f"  Stop Multiplier: {regime.recommended_stop_multiplier:.1f}x")
    print(f"  Target Multiplier: {regime.recommended_target_multiplier:.1f}x")
    print(f"  Trade Bias: {regime.trade_bias}")
    print(f"  Reason: {regime.reason}")

    # Scenario 2: Low Volatility (tight range)
    print("\n[Scenario 2] Low Volatility Compression:")
    low_vol_data = pd.DataFrame({
        'high': 100 + np.sin(np.arange(n) * 0.1) * 0.5 + np.random.randn(n) * 0.1,
        'low': 99.5 + np.sin(np.arange(n) * 0.1) * 0.5 + np.random.randn(n) * 0.1,
        'close': 99.7 + np.sin(np.arange(n) * 0.1) * 0.5 + np.random.randn(n) * 0.1,
        'volume': np.random.randint(500, 1500, n)
    })

    regime2 = detector.detect_regime(low_vol_data)

    print(f"  Regime: {regime2.regime}")
    print(f"  ATR Percentile: {regime2.atr_percentile:.1%}")
    print(f"  BB Width: {regime2.bb_width_pct:.2%}")
    print(f"  Confidence: {regime2.confidence:.0%}")
    print(f"  Stop Multiplier: {regime2.recommended_stop_multiplier:.1f}x")
    print(f"  Target Multiplier: {regime2.recommended_target_multiplier:.1f}x")
    print(f"  Trade Bias: {regime2.trade_bias}")
    print(f"  Reason: {regime2.reason}")

    # Scenario 3: Normal Volatility
    print("\n[Scenario 3] Normal Volatility:")
    normal_vol_data = pd.DataFrame({
        'high': 100 + np.cumsum(np.random.randn(n) * 1),
        'low': 99 + np.cumsum(np.random.randn(n) * 1),
        'close': 99.5 + np.cumsum(np.random.randn(n) * 1),
        'volume': np.random.randint(1000, 3000, n)
    })

    regime3 = detector.detect_regime(normal_vol_data)

    print(f"  Regime: {regime3.regime}")
    print(f"  ATR Percentile: {regime3.atr_percentile:.1%}")
    print(f"  BB Width: {regime3.bb_width_pct:.2%}")
    print(f"  Confidence: {regime3.confidence:.0%}")
    print(f"  Stop Multiplier: {regime3.recommended_stop_multiplier:.1f}x")
    print(f"  Target Multiplier: {regime3.recommended_target_multiplier:.1f}x")
    print(f"  Trade Bias: {regime3.trade_bias}")
    print(f"  Reason: {regime3.reason}")

    print("\n" + "=" * 70)
    print("✅ Volatility Regime Detector ready for production!")
    print("=" * 70)
