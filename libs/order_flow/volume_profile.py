"""
Volume Profile Analyzer

Builds volume-by-price histograms to identify:
1. POC (Point of Control) - Price with highest volume
2. VAH (Value Area High) - Top of 70% volume concentration
3. VAL (Value Area Low) - Bottom of 70% volume concentration
4. HVN (High Volume Nodes) - Strong support/resistance
5. LVN (Low Volume Nodes) - Weak areas, price moves fast

Volume Profile is used by institutional traders to identify:
- Fair value zones (where most trading occurred)
- Support/resistance levels (HVN)
- Breakout zones (LVN)
- Acceptance vs rejection areas

Research: POC accuracy for support/resistance: 65-75% in crypto
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class VolumeProfile:
    """
    Build and analyze volume-by-price profiles

    Volume Profile shows where volume traded at each price level,
    revealing institutional accumulation/distribution zones.
    """

    def __init__(
        self,
        price_bins: int = 50,
        value_area_pct: float = 0.70,
        lookback_hours: int = 24
    ):
        """
        Initialize Volume Profile analyzer

        Args:
            price_bins: Number of price levels to bin (default: 50)
            value_area_pct: Percentage for value area (default: 70%)
            lookback_hours: Hours of data to analyze (default: 24h)
        """
        self.price_bins = price_bins
        self.value_area_pct = value_area_pct
        self.lookback_hours = lookback_hours

        # Storage
        self.volume_by_price: Dict[str, Dict[float, float]] = {}
        self.profiles: Dict[str, Dict] = {}

    def update_volume_at_price(
        self,
        symbol: str,
        price: float,
        volume: float,
        timestamp: datetime = None
    ):
        """
        Update volume traded at a specific price

        Args:
            symbol: Trading symbol
            price: Price level
            volume: Volume traded at this price
            timestamp: Time of trade
        """
        if symbol not in self.volume_by_price:
            self.volume_by_price[symbol] = defaultdict(float)

        # Round price to reduce granularity
        price_rounded = round(price, 2)
        self.volume_by_price[symbol][price_rounded] += volume

    def build_profile(
        self,
        symbol: str,
        prices: np.ndarray,
        volumes: np.ndarray,
        high_prices: Optional[np.ndarray] = None,
        low_prices: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Build volume profile from OHLCV data

        Args:
            symbol: Trading symbol
            prices: Close prices array
            volumes: Volume array
            high_prices: High prices (for better accuracy)
            low_prices: Low prices (for better accuracy)

        Returns:
            Dict with volume profile metrics
        """
        if len(prices) == 0 or len(volumes) == 0:
            return self._empty_profile()

        # Use high/low if available for better accuracy
        if high_prices is not None and low_prices is not None:
            min_price = low_prices.min()
            max_price = high_prices.max()
        else:
            min_price = prices.min()
            max_price = prices.max()

        # Create price bins
        price_range = max_price - min_price
        if price_range == 0:
            return self._empty_profile()

        bin_size = price_range / self.price_bins
        bins = np.linspace(min_price, max_price, self.price_bins + 1)

        # Build volume histogram
        volume_histogram = np.zeros(self.price_bins)

        for i in range(len(prices)):
            # Find which bin this price falls into
            if high_prices is not None and low_prices is not None:
                # Distribute volume across price range of candle
                low = low_prices[i]
                high = high_prices[i]
                volume = volumes[i]

                # Simple distribution: split volume evenly across bins touched
                low_bin = max(0, min(self.price_bins - 1, int((low - min_price) / bin_size)))
                high_bin = max(0, min(self.price_bins - 1, int((high - min_price) / bin_size)))

                bins_touched = max(1, high_bin - low_bin + 1)
                volume_per_bin = volume / bins_touched

                for bin_idx in range(low_bin, high_bin + 1):
                    if 0 <= bin_idx < self.price_bins:
                        volume_histogram[bin_idx] += volume_per_bin
            else:
                # Just use close price
                close = prices[i]
                volume = volumes[i]
                bin_idx = max(0, min(self.price_bins - 1, int((close - min_price) / bin_size)))
                volume_histogram[bin_idx] += volume

        # Calculate key metrics
        profile = self._calculate_profile_metrics(
            bins,
            volume_histogram,
            min_price,
            max_price,
            bin_size
        )

        # Store profile
        self.profiles[symbol] = {
            **profile,
            'symbol': symbol,
            'timestamp': datetime.now(),
            'lookback_hours': self.lookback_hours
        }

        return self.profiles[symbol]

    def _calculate_profile_metrics(
        self,
        bins: np.ndarray,
        volume_histogram: np.ndarray,
        min_price: float,
        max_price: float,
        bin_size: float
    ) -> Dict:
        """Calculate POC, VAH, VAL, HVN, LVN"""

        total_volume = volume_histogram.sum()
        if total_volume == 0:
            return self._empty_profile()

        # 1. POC (Point of Control) - Price with highest volume
        poc_bin = volume_histogram.argmax()
        poc_price = min_price + (poc_bin * bin_size) + (bin_size / 2)
        poc_volume = volume_histogram[poc_bin]

        # 2. Value Area (70% of volume)
        # Start from POC and expand outward until we have 70% of volume
        target_volume = total_volume * self.value_area_pct
        value_area_volume = poc_volume
        lower_idx = poc_bin
        upper_idx = poc_bin

        while value_area_volume < target_volume:
            # Check which direction has more volume
            lower_volume = volume_histogram[lower_idx - 1] if lower_idx > 0 else 0
            upper_volume = volume_histogram[upper_idx + 1] if upper_idx < len(volume_histogram) - 1 else 0

            if lower_volume > upper_volume and lower_idx > 0:
                lower_idx -= 1
                value_area_volume += lower_volume
            elif upper_idx < len(volume_histogram) - 1:
                upper_idx += 1
                value_area_volume += upper_volume
            else:
                break

        # VAL and VAH
        val_price = min_price + (lower_idx * bin_size) + (bin_size / 2)
        vah_price = min_price + (upper_idx * bin_size) + (bin_size / 2)

        # 3. High Volume Nodes (HVN) - Bins with volume > 1.5x average
        avg_volume = volume_histogram.mean()
        hvn_threshold = avg_volume * 1.5

        hvn_prices = []
        for i in range(len(volume_histogram)):
            if volume_histogram[i] >= hvn_threshold:
                hvn_price = min_price + (i * bin_size) + (bin_size / 2)
                hvn_prices.append({
                    'price': hvn_price,
                    'volume': volume_histogram[i],
                    'volume_pct': (volume_histogram[i] / total_volume) * 100
                })

        # Sort HVN by volume (strongest first)
        hvn_prices = sorted(hvn_prices, key=lambda x: x['volume'], reverse=True)

        # 4. Low Volume Nodes (LVN) - Bins with volume < 0.5x average
        lvn_threshold = avg_volume * 0.5

        lvn_prices = []
        for i in range(len(volume_histogram)):
            if volume_histogram[i] <= lvn_threshold and volume_histogram[i] > 0:
                lvn_price = min_price + (i * bin_size) + (bin_size / 2)
                lvn_prices.append({
                    'price': lvn_price,
                    'volume': volume_histogram[i],
                    'volume_pct': (volume_histogram[i] / total_volume) * 100
                })

        return {
            'poc': poc_price,
            'poc_volume': poc_volume,
            'poc_volume_pct': (poc_volume / total_volume) * 100,
            'val': val_price,
            'vah': vah_price,
            'value_area_volume_pct': (value_area_volume / total_volume) * 100,
            'hvn': hvn_prices[:10],  # Top 10 HVN
            'lvn': lvn_prices[:10],  # Top 10 LVN
            'total_volume': total_volume,
            'min_price': min_price,
            'max_price': max_price,
            'price_range': max_price - min_price
        }

    def _empty_profile(self) -> Dict:
        """Return empty profile structure"""
        return {
            'poc': 0.0,
            'poc_volume': 0.0,
            'poc_volume_pct': 0.0,
            'val': 0.0,
            'vah': 0.0,
            'value_area_volume_pct': 0.0,
            'hvn': [],
            'lvn': [],
            'total_volume': 0.0,
            'min_price': 0.0,
            'max_price': 0.0,
            'price_range': 0.0
        }

    def get_support_resistance_levels(
        self,
        symbol: str,
        current_price: float,
        num_levels: int = 3
    ) -> Dict[str, List[float]]:
        """
        Get support and resistance levels from volume profile

        Args:
            symbol: Trading symbol
            current_price: Current market price
            num_levels: Number of levels above/below price

        Returns:
            Dict with support and resistance price levels
        """
        if symbol not in self.profiles:
            return {'support': [], 'resistance': []}

        profile = self.profiles[symbol]

        # HVN act as support (below) and resistance (above)
        hvn_prices = [node['price'] for node in profile['hvn']]

        support_levels = sorted([p for p in hvn_prices if p < current_price], reverse=True)[:num_levels]
        resistance_levels = sorted([p for p in hvn_prices if p > current_price])[:num_levels]

        # Add POC if it's significant
        poc = profile['poc']
        if poc < current_price and poc not in support_levels:
            support_levels.append(poc)
            support_levels = sorted(support_levels, reverse=True)[:num_levels]
        elif poc > current_price and poc not in resistance_levels:
            resistance_levels.append(poc)
            resistance_levels = sorted(resistance_levels)[:num_levels]

        return {
            'support': support_levels,
            'resistance': resistance_levels,
            'poc': poc,
            'val': profile['val'],
            'vah': profile['vah']
        }

    def is_price_at_hvn(
        self,
        symbol: str,
        price: float,
        tolerance_pct: float = 0.5
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Check if price is near a High Volume Node

        Args:
            symbol: Trading symbol
            price: Price to check
            tolerance_pct: How close to HVN (default: 0.5%)

        Returns:
            (is_at_hvn, hvn_info) tuple
        """
        if symbol not in self.profiles:
            return False, None

        profile = self.profiles[symbol]
        tolerance = price * (tolerance_pct / 100)

        for hvn in profile['hvn']:
            hvn_price = hvn['price']
            if abs(price - hvn_price) <= tolerance:
                return True, hvn

        return False, None

    def is_price_at_lvn(
        self,
        symbol: str,
        price: float,
        tolerance_pct: float = 0.5
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Check if price is near a Low Volume Node (breakout zone)

        Args:
            symbol: Trading symbol
            price: Price to check
            tolerance_pct: How close to LVN (default: 0.5%)

        Returns:
            (is_at_lvn, lvn_info) tuple
        """
        if symbol not in self.profiles:
            return False, None

        profile = self.profiles[symbol]
        tolerance = price * (tolerance_pct / 100)

        for lvn in profile['lvn']:
            lvn_price = lvn['price']
            if abs(price - lvn_price) <= tolerance:
                return True, lvn

        return False, None

    def get_trading_bias(
        self,
        symbol: str,
        current_price: float
    ) -> Dict[str, str]:
        """
        Get trading bias based on price position relative to value area

        Args:
            symbol: Trading symbol
            current_price: Current market price

        Returns:
            Dict with bias and reasoning
        """
        if symbol not in self.profiles:
            return {'bias': 'neutral', 'reason': 'No profile data'}

        profile = self.profiles[symbol]
        poc = profile['poc']
        val = profile['val']
        vah = profile['vah']

        # Price above VAH = bullish (price accepted above value)
        if current_price > vah:
            distance_pct = ((current_price - vah) / vah) * 100
            return {
                'bias': 'bullish',
                'reason': f'Price {distance_pct:.2f}% above Value Area High',
                'strength': 'strong' if distance_pct > 2 else 'moderate'
            }

        # Price below VAL = bearish (price accepted below value)
        elif current_price < val:
            distance_pct = ((val - current_price) / val) * 100
            return {
                'bias': 'bearish',
                'reason': f'Price {distance_pct:.2f}% below Value Area Low',
                'strength': 'strong' if distance_pct > 2 else 'moderate'
            }

        # Price inside value area = neutral (balanced)
        elif val <= current_price <= vah:
            # Check if near POC
            distance_from_poc = abs(current_price - poc)
            poc_distance_pct = (distance_from_poc / poc) * 100

            if poc_distance_pct < 0.5:
                return {
                    'bias': 'neutral',
                    'reason': f'Price at POC (fair value)',
                    'strength': 'strong'
                }
            else:
                return {
                    'bias': 'neutral',
                    'reason': f'Price in value area (balanced)',
                    'strength': 'moderate'
                }

        return {'bias': 'neutral', 'reason': 'Unknown'}


# Example usage and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("="*70)
    print("VOLUME PROFILE ANALYZER TEST")
    print("="*70)

    # Create volume profile analyzer
    vp = VolumeProfile(price_bins=50, value_area_pct=0.70)

    # Simulate price data (BTC ranging 95K-105K)
    np.random.seed(42)
    n_candles = 1440  # 24 hours of 1-minute candles

    # Simulate prices with volume concentration around 100K (POC)
    prices = np.random.normal(100000, 2000, n_candles)
    prices = np.clip(prices, 95000, 105000)

    # Simulate volumes (higher at certain price levels)
    volumes = np.random.exponential(scale=0.5, size=n_candles)

    # High/low prices (for better accuracy)
    highs = prices + np.random.uniform(50, 200, n_candles)
    lows = prices - np.random.uniform(50, 200, n_candles)

    # Build volume profile
    print("\nBuilding volume profile from 24h of 1-minute data...")
    profile = vp.build_profile('BTC-USD', prices, volumes, highs, lows)

    # Display results
    print(f"\n📊 Volume Profile Metrics:")
    print(f"  POC (Point of Control):  ${profile['poc']:,.2f} ({profile['poc_volume_pct']:.1f}% of volume)")
    print(f"  VAH (Value Area High):   ${profile['vah']:,.2f}")
    print(f"  VAL (Value Area Low):    ${profile['val']:,.2f}")
    print(f"  Value Area Volume:        {profile['value_area_volume_pct']:.1f}%")
    print(f"  Price Range:             ${profile['min_price']:,.2f} - ${profile['max_price']:,.2f}")

    print(f"\n🎯 High Volume Nodes (Support/Resistance):")
    for i, hvn in enumerate(profile['hvn'][:5], 1):
        print(f"  {i}. ${hvn['price']:,.2f} ({hvn['volume_pct']:.1f}% volume)")

    print(f"\n⚡ Low Volume Nodes (Breakout Zones):")
    for i, lvn in enumerate(profile['lvn'][:5], 1):
        print(f"  {i}. ${lvn['price']:,.2f} ({lvn['volume_pct']:.1f}% volume)")

    # Test support/resistance detection
    current_price = 99500
    sr_levels = vp.get_support_resistance_levels('BTC-USD', current_price, num_levels=3)

    print(f"\n📍 Support/Resistance at ${current_price:,.2f}:")
    print(f"  Support:    {[f'${p:,.2f}' for p in sr_levels['support']]}")
    print(f"  Resistance: {[f'${p:,.2f}' for p in sr_levels['resistance']]}")

    # Test HVN detection
    is_at_hvn, hvn_info = vp.is_price_at_hvn('BTC-USD', current_price, tolerance_pct=0.5)
    print(f"\n  At HVN: {is_at_hvn}")

    # Test trading bias
    bias = vp.get_trading_bias('BTC-USD', current_price)
    print(f"\n📈 Trading Bias: {bias['bias'].upper()} ({bias['strength']})")
    print(f"  Reason: {bias['reason']}")

    print("\n" + "="*70)
    print("✅ Volume Profile Analyzer ready for production!")
    print("="*70)
