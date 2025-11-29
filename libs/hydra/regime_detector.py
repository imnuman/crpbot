"""
HYDRA 3.0 - Regime Detector (Layer 1)

Classifies market into 5 regimes:
- TRENDING_UP: Deploy trend-following strategies (long bias)
- TRENDING_DOWN: Deploy trend-following strategies (short bias)
- RANGING: Deploy mean-reversion strategies
- VOLATILE: Deploy breakout strategies
- CHOPPY: STAY CASH (Guardian override)

Uses ADX, ATR, and Bollinger Band width to determine regime.
"""

from typing import Dict, List, Tuple
from loguru import logger
import numpy as np
from datetime import datetime, timezone


class RegimeDetector:
    """
    Detects market regime using technical indicators.

    ADX > 25 = Trending
    ADX < 20 + Tight BBands = Ranging
    ATR spike > 2x average = Volatile
    Mixed signals = Choppy → Guardian forces CASH
    """

    # Regime thresholds
    ADX_TRENDING_THRESHOLD = 25.0
    ADX_RANGING_THRESHOLD = 20.0
    ATR_SPIKE_MULTIPLIER = 2.0
    BB_WIDTH_RANGING_THRESHOLD = 0.02  # 2% of price

    # Regime persistence (avoid whipsaw)
    MIN_HOURS_IN_REGIME = 2  # Must stay in regime 2+ hours before switching

    def __init__(self):
        self.regime_history = {}  # symbol -> [(timestamp, regime), ...]
        logger.info("Regime Detector initialized")

    def detect_regime(
        self,
        symbol: str,
        candles: List[Dict],
        lookback_adx: int = 14,
        lookback_atr: int = 14,
        lookback_bb: int = 20
    ) -> Tuple[str, Dict]:
        """
        Detect current market regime.

        Args:
            symbol: Symbol (e.g., "BTC-USD", "USD/TRY")
            candles: List of OHLCV candles (newest last)
            lookback_adx: ADX period
            lookback_atr: ATR period
            lookback_bb: Bollinger Band period

        Returns:
            (regime: str, metrics: Dict)
        """
        if len(candles) < max(lookback_adx, lookback_atr, lookback_bb) + 1:
            logger.warning(f"Insufficient candles for {symbol}: {len(candles)}")
            return "CHOPPY", {}

        # Calculate indicators
        adx = self._calculate_adx(candles, lookback_adx)
        atr = self._calculate_atr(candles, lookback_atr)
        atr_sma = self._calculate_sma([c.get('atr', 0) for c in candles[-lookback_atr:]], lookback_atr)
        bb_width = self._calculate_bb_width(candles, lookback_bb)

        metrics = {
            "adx": adx,
            "atr": atr,
            "atr_sma": atr_sma,
            "bb_width": bb_width,
            "current_price": candles[-1]['close']
        }

        # Regime decision tree
        regime = self._classify_regime(adx, atr, atr_sma, bb_width, candles)

        # Apply regime persistence (avoid whipsaw)
        regime = self._apply_persistence(symbol, regime)

        # Store in history
        if symbol not in self.regime_history:
            self.regime_history[symbol] = []
        self.regime_history[symbol].append((datetime.now(timezone.utc), regime))

        # Keep only last 48 hours
        cutoff = datetime.now(timezone.utc).timestamp() - (48 * 3600)
        self.regime_history[symbol] = [
            (ts, r) for ts, r in self.regime_history[symbol]
            if ts.timestamp() > cutoff
        ]

        logger.info(f"{symbol} regime: {regime} (ADX: {adx:.1f}, ATR: {atr:.4f}, BB: {bb_width:.4f})")
        return regime, metrics

    def _classify_regime(
        self,
        adx: float,
        atr: float,
        atr_sma: float,
        bb_width: float,
        candles: List[Dict]
    ) -> str:
        """
        Decision tree for regime classification.
        """
        # Check 1: Trending (ADX > 25)
        if adx > self.ADX_TRENDING_THRESHOLD:
            # Determine trend direction
            if self._is_uptrend(candles):
                return "TRENDING_UP"
            else:
                return "TRENDING_DOWN"

        # Check 2: Ranging (ADX < 20 + Tight BBands)
        elif adx < self.ADX_RANGING_THRESHOLD and bb_width < self.BB_WIDTH_RANGING_THRESHOLD:
            return "RANGING"

        # Check 3: Volatile (ATR spike)
        elif atr_sma > 0 and atr > self.ATR_SPIKE_MULTIPLIER * atr_sma:
            return "VOLATILE"

        # Check 4: Mixed signals (unclear)
        else:
            return "CHOPPY"

    def _apply_persistence(self, symbol: str, new_regime: str) -> str:
        """
        Apply regime persistence to avoid whipsaw.

        Regime must hold for MIN_HOURS_IN_REGIME before switching.
        """
        if symbol not in self.regime_history or not self.regime_history[symbol]:
            return new_regime

        # Get recent regime
        recent_regimes = self.regime_history[symbol][-10:]  # Last 10 readings
        if not recent_regimes:
            return new_regime

        current_regime = recent_regimes[-1][1]

        # If regime is same, keep it
        if new_regime == current_regime:
            return new_regime

        # Check how long we've been in current regime
        hours_in_current = 0
        for i in range(len(recent_regimes) - 1, -1, -1):
            if recent_regimes[i][1] == current_regime:
                hours_in_current += 1
            else:
                break

        # If we've been in current regime < MIN_HOURS, don't switch yet
        if hours_in_current < self.MIN_HOURS_IN_REGIME:
            logger.debug(f"Regime persistence: staying in {current_regime} ({hours_in_current}hrs < {self.MIN_HOURS_IN_REGIME}hrs)")
            return current_regime

        # OK to switch
        logger.info(f"Regime switch: {current_regime} → {new_regime}")
        return new_regime

    # ==================== INDICATOR CALCULATIONS ====================

    def _calculate_adx(self, candles: List[Dict], period: int = 14) -> float:
        """
        Calculate Average Directional Index (ADX).

        ADX measures trend strength (0-100).
        >25 = Strong trend
        <20 = Weak trend / Ranging
        """
        if len(candles) < period + 1:
            return 0.0

        # Calculate True Range (TR)
        tr_values = []
        for i in range(1, len(candles)):
            high = candles[i]['high']
            low = candles[i]['low']
            close_prev = candles[i-1]['close']

            tr = max(
                high - low,
                abs(high - close_prev),
                abs(low - close_prev)
            )
            tr_values.append(tr)

        # Calculate Directional Movement (+DM, -DM)
        plus_dm = []
        minus_dm = []

        for i in range(1, len(candles)):
            high = candles[i]['high']
            low = candles[i]['low']
            high_prev = candles[i-1]['high']
            low_prev = candles[i-1]['low']

            up_move = high - high_prev
            down_move = low_prev - low

            if up_move > down_move and up_move > 0:
                plus_dm.append(up_move)
            else:
                plus_dm.append(0)

            if down_move > up_move and down_move > 0:
                minus_dm.append(down_move)
            else:
                minus_dm.append(0)

        # Smooth with EMA
        plus_di = self._ema(plus_dm, period)
        minus_di = self._ema(minus_dm, period)
        atr = self._ema(tr_values, period)

        if not plus_di or not minus_di or not atr or atr[-1] == 0:
            return 0.0

        # Calculate DI+, DI-
        plus_di_val = 100 * plus_di[-1] / atr[-1]
        minus_di_val = 100 * minus_di[-1] / atr[-1]

        # Calculate DX
        di_sum = plus_di_val + minus_di_val
        if di_sum == 0:
            return 0.0

        dx = 100 * abs(plus_di_val - minus_di_val) / di_sum

        # ADX is EMA of DX (simplified - using last DX value)
        return dx

    def _calculate_atr(self, candles: List[Dict], period: int = 14) -> float:
        """
        Calculate Average True Range (ATR).

        Measures volatility.
        """
        if len(candles) < period + 1:
            return 0.0

        tr_values = []
        for i in range(1, len(candles)):
            high = candles[i]['high']
            low = candles[i]['low']
            close_prev = candles[i-1]['close']

            tr = max(
                high - low,
                abs(high - close_prev),
                abs(low - close_prev)
            )
            tr_values.append(tr)

        # ATR is EMA of TR
        atr_values = self._ema(tr_values, period)
        return atr_values[-1] if atr_values else 0.0

    def _calculate_bb_width(self, candles: List[Dict], period: int = 20) -> float:
        """
        Calculate Bollinger Band width.

        (upper - lower) / middle

        Narrow bands = Low volatility = Ranging
        Wide bands = High volatility = Trending/Volatile
        """
        if len(candles) < period:
            return 0.0

        closes = [c['close'] for c in candles[-period:]]
        sma = np.mean(closes)
        std = np.std(closes)

        upper = sma + (2 * std)
        lower = sma - (2 * std)

        if sma == 0:
            return 0.0

        return (upper - lower) / sma

    def _calculate_sma(self, values: List[float], period: int) -> float:
        """Simple Moving Average."""
        if len(values) < period:
            return 0.0
        return np.mean(values[-period:])

    def _ema(self, values: List[float], period: int) -> List[float]:
        """
        Exponential Moving Average.
        """
        if len(values) < period:
            return []

        ema_values = []
        multiplier = 2 / (period + 1)

        # Start with SMA
        sma = np.mean(values[:period])
        ema_values.append(sma)

        # Calculate EMA
        for i in range(period, len(values)):
            ema = (values[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
            ema_values.append(ema)

        return ema_values

    def _is_uptrend(self, candles: List[Dict], lookback: int = 20) -> bool:
        """
        Determine if market is in uptrend.

        Simple: Price > SMA(20)
        """
        if len(candles) < lookback:
            return True  # Default to long bias

        closes = [c['close'] for c in candles[-lookback:]]
        sma = np.mean(closes)
        current_price = candles[-1]['close']

        return current_price > sma

    # ==================== UTILITY METHODS ====================

    def get_regime_confidence(self, symbol: str) -> float:
        """
        Calculate confidence in current regime (0.0-1.0).

        Based on how long we've been in this regime.
        """
        if symbol not in self.regime_history or not self.regime_history[symbol]:
            return 0.5

        recent = self.regime_history[symbol][-20:]  # Last 20 readings
        if not recent:
            return 0.5

        current_regime = recent[-1][1]

        # Count consecutive readings in same regime
        consecutive = 0
        for i in range(len(recent) - 1, -1, -1):
            if recent[i][1] == current_regime:
                consecutive += 1
            else:
                break

        # Confidence = % of recent readings in this regime
        confidence = consecutive / len(recent)
        return min(confidence, 1.0)

    def get_regime_duration_hours(self, symbol: str) -> float:
        """
        Get how many hours we've been in current regime.
        """
        if symbol not in self.regime_history or not self.regime_history[symbol]:
            return 0.0

        recent = self.regime_history[symbol]
        if not recent:
            return 0.0

        current_regime = recent[-1][1]
        first_timestamp = None

        # Find when this regime started
        for i in range(len(recent) - 1, -1, -1):
            if recent[i][1] == current_regime:
                first_timestamp = recent[i][0]
            else:
                break

        if first_timestamp:
            duration = (datetime.now(timezone.utc) - first_timestamp).seconds / 3600
            return duration

        return 0.0
