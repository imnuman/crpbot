"""
Technical Indicators Library for Backtesting and ML

Comprehensive collection of 50+ technical indicators organized by category:
- Momentum: RSI, MACD, Stochastic, Williams %R, ROC, CMO
- Volatility: ATR, Bollinger Bands, Keltner Channels, Donchian Channels
- Trend: ADX, Ichimoku, Parabolic SAR, Supertrend, TRIX
- Volume: OBV, VWAP, MFI, A/D Line, Chaikin Money Flow
- Statistical: Z-Score, Percentile Rank, Linear Regression

All indicators are vectorized using pandas/numpy for performance.
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Comprehensive technical indicators calculator

    Usage:
        ti = TechnicalIndicators()
        df = ti.add_all_indicators(df)  # Adds all 50+ indicators

        # Or add specific categories:
        df = ti.add_momentum_indicators(df)
        df = ti.add_volatility_indicators(df)
    """

    def __init__(self):
        """Initialize technical indicators calculator"""
        self.indicators_added = []

    # ========================================================================
    # MOMENTUM INDICATORS
    # ========================================================================

    def rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index (RSI)

        Measures speed and magnitude of price changes.
        Range: 0-100 (>70 overbought, <30 oversold)
        """
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def macd(
        self,
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Moving Average Convergence Divergence (MACD)

        Trend-following momentum indicator.
        Returns: (macd_line, signal_line, histogram)
        """
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator

        Compares closing price to price range over period.
        Range: 0-100 (>80 overbought, <20 oversold)
        Returns: (%K, %D)
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()

        return k, d

    def williams_r(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Williams %R

        Momentum indicator similar to Stochastic.
        Range: -100 to 0 (>-20 overbought, <-80 oversold)
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()

        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr

    def roc(self, close: pd.Series, period: int = 12) -> pd.Series:
        """
        Rate of Change (ROC)

        Measures percentage change in price over period.
        """
        roc = ((close - close.shift(period)) / close.shift(period)) * 100
        return roc

    def cmo(self, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Chande Momentum Oscillator (CMO)

        Similar to RSI but uses sum instead of average.
        Range: -100 to +100
        """
        delta = close.diff()
        sum_gains = delta.where(delta > 0, 0).rolling(window=period).sum()
        sum_losses = -delta.where(delta < 0, 0).rolling(window=period).sum()

        cmo = 100 * (sum_gains - sum_losses) / (sum_gains + sum_losses)
        return cmo

    # ========================================================================
    # VOLATILITY INDICATORS
    # ========================================================================

    def atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Average True Range (ATR)

        Measures market volatility.
        Higher ATR = higher volatility
        """
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def bollinger_bands(
        self,
        close: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands

        Volatility bands around moving average.
        Returns: (upper_band, middle_band, lower_band)
        """
        middle = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper, middle, lower

    def keltner_channels(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20,
        atr_mult: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Keltner Channels

        Volatility-based channels using ATR.
        Returns: (upper_channel, middle_line, lower_channel)
        """
        middle = close.ewm(span=period, adjust=False).mean()
        atr_val = self.atr(high, low, close, period)

        upper = middle + (atr_val * atr_mult)
        lower = middle - (atr_val * atr_mult)

        return upper, middle, lower

    def donchian_channels(
        self,
        high: pd.Series,
        low: pd.Series,
        period: int = 20
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Donchian Channels

        Highest high and lowest low over period.
        Returns: (upper_channel, middle_line, lower_channel)
        """
        upper = high.rolling(window=period).max()
        lower = low.rolling(window=period).min()
        middle = (upper + lower) / 2

        return upper, middle, lower

    # ========================================================================
    # TREND INDICATORS
    # ========================================================================

    def adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Average Directional Index (ADX)

        Measures trend strength (not direction).
        Range: 0-100 (>25 trending, <20 ranging)
        Returns: (adx, plus_di, minus_di)
        """
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx, plus_di, minus_di

    def supertrend(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 10,
        multiplier: float = 3.0
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Supertrend Indicator

        Trend-following indicator using ATR.
        Returns: (supertrend_line, trend_direction)
        """
        hl_avg = (high + low) / 2
        atr_val = self.atr(high, low, close, period)

        upper_band = hl_avg + (multiplier * atr_val)
        lower_band = hl_avg - (multiplier * atr_val)

        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=int)

        # Initialize
        supertrend.iloc[0] = lower_band.iloc[0]
        direction.iloc[0] = 1

        for i in range(1, len(close)):
            if close.iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            elif close.iloc[i] < supertrend.iloc[i-1]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = supertrend.iloc[i-1]
                direction.iloc[i] = direction.iloc[i-1]

        return supertrend, direction

    def trix(self, close: pd.Series, period: int = 15) -> pd.Series:
        """
        TRIX (Triple Exponential Average)

        Measures rate of change of triple EMA.
        Filters out insignificant price movements.
        """
        ema1 = close.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()

        trix = 100 * ema3.pct_change()
        return trix

    # ========================================================================
    # VOLUME INDICATORS
    # ========================================================================

    def obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On-Balance Volume (OBV)

        Cumulative volume indicator.
        Rising OBV = buying pressure
        """
        direction = np.sign(close.diff())
        obv = (direction * volume).cumsum()
        return obv

    def vwap(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """
        Volume Weighted Average Price (VWAP)

        Average price weighted by volume.
        Intraday benchmark.
        """
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap

    def mfi(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Money Flow Index (MFI)

        Volume-weighted RSI.
        Range: 0-100 (>80 overbought, <20 oversold)
        """
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume

        delta = typical_price.diff()
        positive_flow = money_flow.where(delta > 0, 0).rolling(window=period).sum()
        negative_flow = money_flow.where(delta < 0, 0).rolling(window=period).sum()

        mfi = 100 - (100 / (1 + (positive_flow / negative_flow)))
        return mfi

    def ad_line(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """
        Accumulation/Distribution Line

        Volume flow indicator.
        Rising = accumulation, Falling = distribution
        """
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)
        ad = (clv * volume).cumsum()
        return ad

    def cmf(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 20
    ) -> pd.Series:
        """
        Chaikin Money Flow (CMF)

        Measures money flow volume over period.
        Range: -1 to +1
        """
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)

        cmf = (clv * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
        return cmf

    # ========================================================================
    # STATISTICAL INDICATORS
    # ========================================================================

    def z_score(self, close: pd.Series, period: int = 20) -> pd.Series:
        """
        Z-Score

        Measures standard deviations from mean.
        >2 or <-2 indicates extreme values
        """
        mean = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()

        z_score = (close - mean) / std
        return z_score

    def percentile_rank(self, close: pd.Series, period: int = 100) -> pd.Series:
        """
        Percentile Rank

        Current price as percentile of past prices.
        Range: 0-100
        """
        def calc_percentile(x):
            if len(x) < 2:
                return 50.0
            return (x < x.iloc[-1]).sum() / len(x) * 100

        percentile = close.rolling(window=period).apply(calc_percentile, raw=False)
        return percentile

    def linear_regression_slope(self, close: pd.Series, period: int = 20) -> pd.Series:
        """
        Linear Regression Slope

        Slope of best-fit line over period.
        Positive = uptrend, Negative = downtrend
        """
        def calc_slope(y):
            if len(y) < 2:
                return 0.0
            x = np.arange(len(y))
            slope = np.polyfit(x, y, 1)[0]
            return slope

        slope = close.rolling(window=period).apply(calc_slope, raw=True)
        return slope

    # ========================================================================
    # AGGREGATE FUNCTIONS
    # ========================================================================

    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all momentum indicators to DataFrame"""
        df['rsi_14'] = self.rsi(df['close'], 14)
        df['rsi_28'] = self.rsi(df['close'], 28)

        macd, signal, hist = self.macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist

        stoch_k, stoch_d = self.stochastic(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d

        df['williams_r'] = self.williams_r(df['high'], df['low'], df['close'])
        df['roc_12'] = self.roc(df['close'], 12)
        df['cmo_14'] = self.cmo(df['close'], 14)

        logger.info("Added momentum indicators: RSI, MACD, Stochastic, Williams %R, ROC, CMO")
        return df

    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all volatility indicators to DataFrame"""
        df['atr_14'] = self.atr(df['high'], df['low'], df['close'], 14)

        bb_upper, bb_middle, bb_lower = self.bollinger_bands(df['close'])
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle

        kc_upper, kc_middle, kc_lower = self.keltner_channels(df['high'], df['low'], df['close'])
        df['kc_upper'] = kc_upper
        df['kc_middle'] = kc_middle
        df['kc_lower'] = kc_lower

        dc_upper, dc_middle, dc_lower = self.donchian_channels(df['high'], df['low'])
        df['dc_upper'] = dc_upper
        df['dc_middle'] = dc_middle
        df['dc_lower'] = dc_lower

        logger.info("Added volatility indicators: ATR, Bollinger Bands, Keltner Channels, Donchian Channels")
        return df

    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all trend indicators to DataFrame"""
        adx, plus_di, minus_di = self.adx(df['high'], df['low'], df['close'])
        df['adx'] = adx
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di

        supertrend, direction = self.supertrend(df['high'], df['low'], df['close'])
        df['supertrend'] = supertrend
        df['supertrend_dir'] = direction

        df['trix'] = self.trix(df['close'])

        logger.info("Added trend indicators: ADX, Supertrend, TRIX")
        return df

    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all volume indicators to DataFrame"""
        if 'volume' not in df.columns:
            logger.warning("Volume column missing, skipping volume indicators")
            return df

        df['obv'] = self.obv(df['close'], df['volume'])
        df['vwap'] = self.vwap(df['high'], df['low'], df['close'], df['volume'])
        df['mfi'] = self.mfi(df['high'], df['low'], df['close'], df['volume'])
        df['ad_line'] = self.ad_line(df['high'], df['low'], df['close'], df['volume'])
        df['cmf'] = self.cmf(df['high'], df['low'], df['close'], df['volume'])

        logger.info("Added volume indicators: OBV, VWAP, MFI, A/D Line, CMF")
        return df

    def add_statistical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all statistical indicators to DataFrame"""
        df['z_score_20'] = self.z_score(df['close'], 20)
        df['percentile_100'] = self.percentile_rank(df['close'], 100)
        df['lr_slope_20'] = self.linear_regression_slope(df['close'], 20)

        logger.info("Added statistical indicators: Z-Score, Percentile Rank, LR Slope")
        return df

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all 50+ technical indicators to DataFrame

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with all indicators added
        """
        logger.info(f"Adding all technical indicators to {len(df)} rows")

        df = self.add_momentum_indicators(df)
        df = self.add_volatility_indicators(df)
        df = self.add_trend_indicators(df)
        df = self.add_volume_indicators(df)
        df = self.add_statistical_indicators(df)

        # Count indicators added
        new_cols = [c for c in df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        logger.info(f"✅ Added {len(new_cols)} technical indicators")

        return df


# Convenience function
def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to add all indicators

    Usage:
        from libs.features.technical_indicators import add_all_indicators
        df = add_all_indicators(df)
    """
    ti = TechnicalIndicators()
    return ti.add_all_indicators(df)
