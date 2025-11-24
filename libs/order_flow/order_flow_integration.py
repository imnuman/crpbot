"""
Order Flow Integration for V7 Ultimate

Integrates all order flow components into the V7 signal generation pipeline:
1. Order Flow Imbalance (OFI)
2. Volume Profile (VP)
3. Market Microstructure (MS)

These features add the "missing 80%" of market data that candles don't capture.

Usage:
    analyzer = OrderFlowAnalyzer()
    features = analyzer.analyze(symbol, market_data, order_book)
    # Returns comprehensive order flow features for signal generation
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from libs.order_flow.order_flow_imbalance import OrderFlowImbalance
from libs.order_flow.volume_profile import VolumeProfile
from libs.order_flow.market_microstructure import MarketMicrostructure

logger = logging.getLogger(__name__)


class OrderFlowAnalyzer:
    """
    Unified Order Flow analyzer for V7 Ultimate

    Combines OFI, Volume Profile, and Microstructure into a single interface
    """

    def __init__(
        self,
        depth_levels: int = 10,
        lookback_periods: int = 20,
        price_bins: int = 50
    ):
        """
        Initialize Order Flow analyzer

        Args:
            depth_levels: Number of order book levels to analyze
            lookback_periods: Historical periods to keep
            price_bins: Number of bins for volume profile
        """
        self.ofi = OrderFlowImbalance(
            depth_levels=depth_levels,
            lookback_periods=lookback_periods
        )

        self.volume_profile = VolumeProfile(
            price_bins=price_bins,
            lookback_hours=24  # 24 hours of data
        )

        self.microstructure = MarketMicrostructure(
            lookback_periods=lookback_periods,
            depth_levels=depth_levels
        )

        # Store previous order book for OFI calculation
        self.previous_order_books: Dict[str, Dict] = {}

    def analyze(
        self,
        symbol: str,
        candles_df: pd.DataFrame,
        current_order_book: Optional[Dict] = None,
        recent_trades: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive order flow analysis

        Args:
            symbol: Trading symbol (e.g., 'BTC-USD')
            candles_df: DataFrame with OHLCV data (last 60+ minutes)
            current_order_book: Dict with 'bids' and 'asks' lists
            recent_trades: Optional list of recent trades

        Returns:
            Dict with all order flow features
        """
        features = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'has_order_book': current_order_book is not None,
            'has_trades': recent_trades is not None
        }

        try:
            # 1. Volume Profile Analysis (always available from candles)
            vp_features = self._analyze_volume_profile(symbol, candles_df)
            features.update(vp_features)

            # 2. Order Book Analysis (if available)
            if current_order_book:
                ob_features = self._analyze_order_book(
                    symbol, current_order_book
                )
                features.update(ob_features)

                # 3. Microstructure Analysis (needs both candles and order book)
                ms_features = self._analyze_microstructure(
                    symbol, candles_df, current_order_book, recent_trades
                )
                features.update(ms_features)
            else:
                logger.warning(f"{symbol}: No order book data, using volume profile only")
                features['order_flow_quality'] = 'limited'

            # 4. Generate trading signals from order flow
            signals = self._generate_order_flow_signals(features)
            features['signals'] = signals

        except Exception as e:
            logger.error(f"Order flow analysis failed for {symbol}: {e}")
            features['error'] = str(e)

        return features

    def _analyze_volume_profile(
        self,
        symbol: str,
        candles_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze volume profile from OHLCV data

        Returns features:
        - poc_price: Point of Control
        - vah/val: Value Area High/Low
        - price_position: Where current price is relative to value area
        - support/resistance levels
        - trading_bias: BULLISH/BEARISH/NEUTRAL
        """
        try:
            # Extract OHLCV data
            closes = candles_df['close'].values
            volumes = candles_df['volume'].values
            highs = candles_df['high'].values
            lows = candles_df['low'].values

            # Build volume profile
            self.volume_profile.build_profile(
                symbol, closes, volumes, highs, lows
            )

            # Get current price
            current_price = closes[-1]

            # Get support/resistance levels
            levels = self.volume_profile.get_support_resistance_levels(
                symbol, current_price
            )

            # Get value area metrics
            profile = self.volume_profile.profiles.get(symbol)
            if profile:
                poc = profile['poc_price']
                vah = profile['vah']
                val = profile['val']
                trading_bias = profile['bias']

                # Calculate distance from key levels
                distance_from_poc = (current_price - poc) / poc
                distance_from_vah = (current_price - vah) / vah
                distance_from_val = (current_price - val) / val

                return {
                    'vp_poc': poc,
                    'vp_vah': vah,
                    'vp_val': val,
                    'vp_value_area_volume': profile['value_area_volume'],
                    'vp_distance_from_poc': distance_from_poc,
                    'vp_distance_from_vah': distance_from_vah,
                    'vp_distance_from_val': distance_from_val,
                    'vp_trading_bias': trading_bias,
                    'vp_bias_strength': profile['bias_strength'],
                    'vp_at_hvn': levels['at_hvn'],
                    'vp_support_levels': levels['support'],
                    'vp_resistance_levels': levels['resistance'],
                    'vp_nearest_support': levels['nearest_support'],
                    'vp_nearest_resistance': levels['nearest_resistance']
                }
            else:
                return {'vp_available': False}

        except Exception as e:
            logger.error(f"Volume profile analysis failed: {e}")
            return {'vp_error': str(e)}

    def _analyze_order_book(
        self,
        symbol: str,
        order_book: Dict
    ) -> Dict[str, Any]:
        """
        Analyze order book for OFI and imbalance

        Returns features:
        - order_imbalance: Bid/ask volume ratio
        - ofi: Order Flow Imbalance
        - whale_detected: Large orders present
        """
        try:
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])

            if not bids or not asks:
                return {'order_book_available': False}

            # Get previous order book for OFI
            previous_book = self.previous_order_books.get(symbol)
            previous_bids = previous_book['bids'] if previous_book else None
            previous_asks = previous_book['asks'] if previous_book else None

            # Calculate comprehensive metrics
            metrics = self.ofi.get_comprehensive_metrics(
                symbol,
                bids,
                asks,
                previous_bids,
                previous_asks
            )

            # Store current book for next iteration
            self.previous_order_books[symbol] = {
                'bids': bids,
                'asks': asks,
                'timestamp': datetime.now()
            }

            return {
                'ofi_imbalance': metrics['imbalance'],
                'ofi_bid_volume': metrics['bid_volume'],
                'ofi_ask_volume': metrics['ask_volume'],
                'ofi_ratio': metrics['ratio'],
                'ofi_net': metrics['ofi'],
                'ofi_momentum': metrics['ofi_momentum'],
                'ofi_whale_detected': metrics['whale_detected'],
                'ofi_large_bids': len(metrics.get('large_bids', [])),
                'ofi_large_asks': len(metrics.get('large_asks', []))
            }

        except Exception as e:
            logger.error(f"Order book analysis failed: {e}")
            return {'order_book_error': str(e)}

    def _analyze_microstructure(
        self,
        symbol: str,
        candles_df: pd.DataFrame,
        order_book: Dict,
        recent_trades: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Analyze market microstructure

        Returns features:
        - vwap_deviation: How far from fair value
        - spread metrics: Liquidity quality
        - depth metrics: Order book depth
        - trade aggressiveness: Buyer/seller urgency
        """
        try:
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])

            if not bids or not asks:
                return {'microstructure_available': False}

            # Extract price/volume for VWAP
            prices = candles_df['close'].values[-20:]  # Last 20 bars
            volumes = candles_df['volume'].values[-20:]

            # Calculate comprehensive microstructure
            ms_metrics = self.microstructure.get_comprehensive_microstructure(
                symbol,
                prices.tolist(),
                volumes.tolist(),
                bids,
                asks,
                recent_trades,
                hypothetical_trade_size=1.0
            )

            # Extract key metrics
            vwap = ms_metrics['vwap']
            spread = ms_metrics['spread']
            depth = ms_metrics['depth']
            impact_buy = ms_metrics['price_impact_buy']
            impact_sell = ms_metrics['price_impact_sell']
            aggression = ms_metrics['trade_aggression']

            return {
                'ms_vwap': vwap['vwap'],
                'ms_vwap_deviation': vwap['deviation'],
                'ms_vwap_deviation_pct': vwap['deviation_pct'],
                'ms_spread_bps': spread['relative_spread_bps'],
                'ms_spread_quality': spread['liquidity_quality'],
                'ms_spread_percentile': spread['spread_percentile'],
                'ms_depth_imbalance': depth['depth_imbalance'],
                'ms_depth_trend': depth['depth_trend'],
                'ms_weighted_mid': depth['weighted_mid'],
                'ms_impact_buy_bps': impact_buy['price_impact_bps'],
                'ms_impact_sell_bps': impact_sell['price_impact_bps'],
                'ms_buy_pressure': aggression['buy_pressure'],
                'ms_aggression_trend': aggression['aggression_trend']
            }

        except Exception as e:
            logger.error(f"Microstructure analysis failed: {e}")
            return {'microstructure_error': str(e)}

    def _generate_order_flow_signals(
        self,
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate trading signals from order flow features

        Returns:
            Dict with signal recommendations:
            - direction: 'LONG', 'SHORT', 'HOLD'
            - strength: 0.0 to 1.0
            - reasons: List of supporting factors
        """
        signals = {
            'direction': 'HOLD',
            'strength': 0.0,
            'reasons': [],
            'warnings': []
        }

        bullish_score = 0.0
        bearish_score = 0.0

        # 1. Volume Profile Signals
        if 'vp_trading_bias' in features:
            bias = features['vp_trading_bias']
            bias_strength = features.get('vp_bias_strength', 'weak')

            if bias == 'BULLISH':
                bullish_score += 0.3 if bias_strength == 'strong' else 0.15
                signals['reasons'].append(f"Volume profile: {bias} ({bias_strength})")
            elif bias == 'BEARISH':
                bearish_score += 0.3 if bias_strength == 'strong' else 0.15
                signals['reasons'].append(f"Volume profile: {bias} ({bias_strength})")

            # Support/resistance proximity
            at_hvn = features.get('vp_at_hvn', False)
            if at_hvn:
                signals['warnings'].append("Price at high volume node (support/resistance)")

        # 2. Order Flow Imbalance Signals
        if 'ofi_imbalance' in features:
            imbalance = features['ofi_imbalance']
            ofi_momentum = features.get('ofi_momentum', 0.0)

            # Strong bid imbalance = bullish
            if imbalance > 0.2:  # 20%+ more bids than asks
                bullish_score += 0.2
                signals['reasons'].append(f"Strong bid imbalance: {imbalance:+.2f}")
            elif imbalance < -0.2:  # 20%+ more asks than bids
                bearish_score += 0.2
                signals['reasons'].append(f"Strong ask imbalance: {imbalance:+.2f}")

            # OFI momentum (sustained pressure)
            if ofi_momentum > 0.1:
                bullish_score += 0.15
                signals['reasons'].append("Sustained buying pressure (OFI+)")
            elif ofi_momentum < -0.1:
                bearish_score += 0.15
                signals['reasons'].append("Sustained selling pressure (OFI-)")

            # Whale detection
            if features.get('ofi_whale_detected', False):
                large_bids = features.get('ofi_large_bids', 0)
                large_asks = features.get('ofi_large_asks', 0)

                if large_bids > large_asks:
                    bullish_score += 0.1
                    signals['reasons'].append(f"Large buy orders detected ({large_bids})")
                elif large_asks > large_bids:
                    bearish_score += 0.1
                    signals['reasons'].append(f"Large sell orders detected ({large_asks})")

        # 3. Microstructure Signals
        if 'ms_vwap_deviation_pct' in features:
            vwap_dev = features['ms_vwap_deviation_pct']

            # Price far below VWAP = oversold, potential bounce
            if vwap_dev < -1.0:  # 1%+ below VWAP
                bullish_score += 0.15
                signals['reasons'].append(f"Price {abs(vwap_dev):.1f}% below VWAP (cheap)")
            elif vwap_dev > 1.0:  # 1%+ above VWAP
                bearish_score += 0.15
                signals['reasons'].append(f"Price {vwap_dev:.1f}% above VWAP (expensive)")

            # Depth imbalance
            depth_imbalance = features.get('ms_depth_imbalance', 0.0)
            if depth_imbalance > 0.15:  # More bid depth
                bullish_score += 0.1
                signals['reasons'].append("Strong bid depth support")
            elif depth_imbalance < -0.15:  # More ask depth
                bearish_score += 0.1
                signals['reasons'].append("Heavy ask depth resistance")

            # Trade aggressiveness
            buy_pressure = features.get('ms_buy_pressure', 0.5)
            if buy_pressure > 0.65:  # 65%+ aggressive buying
                bullish_score += 0.1
                signals['reasons'].append(f"Aggressive buying: {buy_pressure:.0%}")
            elif buy_pressure < 0.35:  # 65%+ aggressive selling
                bearish_score += 0.1
                signals['reasons'].append(f"Aggressive selling: {1-buy_pressure:.0%}")

        # 4. Liquidity Quality Check
        if 'ms_spread_quality' in features:
            spread_quality = features['ms_spread_quality']
            if spread_quality == 'poor':
                signals['warnings'].append("Poor liquidity - wide spreads detected")

        # 5. Determine final signal
        net_score = bullish_score - bearish_score

        if net_score > 0.4:
            signals['direction'] = 'LONG'
            signals['strength'] = min(1.0, net_score)
        elif net_score < -0.4:
            signals['direction'] = 'SHORT'
            signals['strength'] = min(1.0, abs(net_score))
        else:
            signals['direction'] = 'HOLD'
            signals['strength'] = 0.0
            signals['reasons'].append(f"No clear order flow edge (net: {net_score:+.2f})")

        return signals

    def get_feature_summary(self, features: Dict[str, Any]) -> str:
        """
        Generate human-readable summary of order flow features

        Args:
            features: Order flow features dict

        Returns:
            Formatted string summary
        """
        lines = []
        lines.append(f"Order Flow Analysis - {features['symbol']}")
        lines.append("=" * 60)

        # Volume Profile
        if 'vp_trading_bias' in features:
            lines.append(f"\nVolume Profile:")
            lines.append(f"  Bias:       {features['vp_trading_bias']} ({features.get('vp_bias_strength', 'weak')})")
            lines.append(f"  POC:        ${features['vp_poc']:.2f}")
            lines.append(f"  Value Area: ${features['vp_val']:.2f} - ${features['vp_vah']:.2f}")

        # Order Flow Imbalance
        if 'ofi_imbalance' in features:
            lines.append(f"\nOrder Flow:")
            lines.append(f"  Imbalance:  {features['ofi_imbalance']:+.3f}")
            lines.append(f"  OFI:        {features['ofi_net']:+.3f}")
            lines.append(f"  Momentum:   {features['ofi_momentum']:+.3f}")

        # Microstructure
        if 'ms_vwap_deviation_pct' in features:
            lines.append(f"\nMicrostructure:")
            lines.append(f"  VWAP Dev:   {features['ms_vwap_deviation_pct']:+.2f}%")
            lines.append(f"  Spread:     {features['ms_spread_bps']:.1f} bps ({features['ms_spread_quality']})")
            lines.append(f"  Depth Imb:  {features['ms_depth_imbalance']:+.3f}")
            lines.append(f"  Buy Press:  {features['ms_buy_pressure']:.1%}")

        # Signals
        signals = features.get('signals', {})
        lines.append(f"\nSignal:")
        lines.append(f"  Direction:  {signals.get('direction', 'N/A')}")
        lines.append(f"  Strength:   {signals.get('strength', 0):.2f}")
        lines.append(f"  Reasons:    {len(signals.get('reasons', []))}")

        for reason in signals.get('reasons', [])[:5]:
            lines.append(f"    - {reason}")

        for warning in signals.get('warnings', []):
            lines.append(f"  ⚠️  {warning}")

        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("ORDER FLOW INTEGRATION TEST")
    print("=" * 70)

    # Create analyzer
    analyzer = OrderFlowAnalyzer(
        depth_levels=10,
        lookback_periods=20,
        price_bins=50
    )

    # Simulate market data
    # 1. OHLCV candles (last 60 minutes)
    np.random.seed(42)
    base_price = 100000.0
    candles_data = []

    for i in range(60):
        price = base_price + np.random.randn() * 100
        candles_data.append({
            'timestamp': pd.Timestamp.now() - pd.Timedelta(minutes=60-i),
            'open': price,
            'high': price + np.random.rand() * 50,
            'low': price - np.random.rand() * 50,
            'close': price + np.random.randn() * 20,
            'volume': np.random.rand() * 10
        })

    candles_df = pd.DataFrame(candles_data)

    # 2. Current order book
    current_price = candles_df['close'].iloc[-1]
    order_book = {
        'bids': [
            [current_price - 5, 2.0],
            [current_price - 10, 3.5],
            [current_price - 15, 1.5],
            [current_price - 20, 2.5],
            [current_price - 25, 4.0]
        ],
        'asks': [
            [current_price + 5, 1.5],
            [current_price + 10, 2.5],
            [current_price + 15, 3.0],
            [current_price + 20, 1.0],
            [current_price + 25, 2.0]
        ]
    }

    # 3. Recent trades (optional)
    recent_trades = [
        {'price': current_price, 'size': 0.5, 'side': 'buy', 'timestamp': datetime.now()}
    ]

    # Analyze
    print("\n🔍 Analyzing BTC-USD order flow...")
    features = analyzer.analyze(
        'BTC-USD',
        candles_df,
        order_book,
        recent_trades
    )

    # Print summary
    print("\n")
    print(analyzer.get_feature_summary(features))

    print("\n" + "=" * 70)
    print("✅ Order Flow Integration ready for V7 Ultimate!")
    print("=" * 70)
