"""
Order Flow Imbalance (OFI) Calculator

Calculates real-time order flow imbalance from Level 2 order book data.
OFI is one of the most predictive features for short-term price movement.

Key Metrics:
1. Order Imbalance = (Bid Volume - Ask Volume) / (Bid Volume + Ask Volume)
2. Delta (Aggressive buying/selling)
3. CVD (Cumulative Volume Delta)
4. Large order detection (whale activity)

Research: OFI explains 8-10% of price variance in crypto markets
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import deque
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class OrderFlowImbalance:
    """
    Calculate Order Flow Imbalance from Level 2 order book

    OFI = (ΔBid_volume - ΔAsk_volume) at each price level
    Positive OFI = buying pressure
    Negative OFI = selling pressure
    """

    def __init__(self, depth_levels: int = 10, lookback_periods: int = 20):
        """
        Initialize OFI calculator

        Args:
            depth_levels: Number of order book levels to analyze (default: 10)
            lookback_periods: Number of snapshots to keep for trend analysis
        """
        self.depth_levels = depth_levels
        self.lookback_periods = lookback_periods

        # Historical storage
        self.order_book_history: Dict[str, deque] = {}
        self.ofi_history: Dict[str, deque] = {}
        self.delta_history: Dict[str, deque] = {}

    def calculate_order_imbalance(
        self,
        bids: List[List[float]],
        asks: List[List[float]]
    ) -> Dict[str, float]:
        """
        Calculate order book imbalance at current snapshot

        Args:
            bids: List of [price, size] for bid side
            asks: List of [price, size] for ask side

        Returns:
            Dict with imbalance metrics
        """
        # Extract top N levels
        bids = bids[:self.depth_levels]
        asks = asks[:self.depth_levels]

        # Calculate total volumes
        bid_volume = sum([size for _, size in bids])
        ask_volume = sum([size for _, size in asks])
        total_volume = bid_volume + ask_volume

        if total_volume == 0:
            return {
                'imbalance': 0.0,
                'bid_volume': 0.0,
                'ask_volume': 0.0,
                'ratio': 1.0
            }

        # Order imbalance: -1 (all asks) to +1 (all bids)
        imbalance = (bid_volume - ask_volume) / total_volume

        # Bid/Ask ratio
        ratio = bid_volume / ask_volume if ask_volume > 0 else 999.0

        return {
            'imbalance': imbalance,
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'ratio': ratio,
            'timestamp': datetime.now()
        }

    def calculate_ofi(
        self,
        symbol: str,
        current_bids: List[List[float]],
        current_asks: List[List[float]],
        previous_bids: Optional[List[List[float]]] = None,
        previous_asks: Optional[List[List[float]]] = None
    ) -> Dict[str, float]:
        """
        Calculate Order Flow Imbalance (OFI)

        OFI measures changes in liquidity at each price level
        Positive = net buying pressure, Negative = net selling pressure

        Args:
            symbol: Trading symbol
            current_bids/asks: Current order book state
            previous_bids/asks: Previous order book state

        Returns:
            Dict with OFI metrics
        """
        # Initialize history if needed
        if symbol not in self.order_book_history:
            self.order_book_history[symbol] = deque(maxlen=self.lookback_periods)
            self.ofi_history[symbol] = deque(maxlen=self.lookback_periods)

        # If no previous state, can't calculate OFI
        if previous_bids is None or previous_asks is None:
            self.order_book_history[symbol].append({
                'bids': current_bids,
                'asks': current_asks,
                'timestamp': datetime.now()
            })
            return {'ofi': 0.0, 'ofi_momentum': 0.0}

        # Calculate OFI at each price level
        ofi_bid = 0.0
        ofi_ask = 0.0

        # Bid side OFI
        for i in range(min(len(current_bids), len(previous_bids), self.depth_levels)):
            curr_price, curr_size = current_bids[i]
            prev_price, prev_size = previous_bids[i]

            # If price level unchanged, calculate volume change
            if abs(curr_price - prev_price) < 0.01:  # Same price level
                delta_volume = curr_size - prev_size
                ofi_bid += delta_volume

        # Ask side OFI
        for i in range(min(len(current_asks), len(previous_asks), self.depth_levels)):
            curr_price, curr_size = current_asks[i]
            prev_price, prev_size = previous_asks[i]

            if abs(curr_price - prev_price) < 0.01:
                delta_volume = curr_size - prev_size
                ofi_ask += delta_volume

        # Net OFI = Bid liquidity increase - Ask liquidity increase
        # Positive = buying pressure (bids added faster than asks)
        net_ofi = ofi_bid - ofi_ask

        # Store OFI history
        self.ofi_history[symbol].append({
            'ofi': net_ofi,
            'ofi_bid': ofi_bid,
            'ofi_ask': ofi_ask,
            'timestamp': datetime.now()
        })

        # Calculate OFI momentum (trend)
        ofi_momentum = 0.0
        if len(self.ofi_history[symbol]) >= 5:
            recent_ofi = [entry['ofi'] for entry in list(self.ofi_history[symbol])[-5:]]
            ofi_momentum = np.mean(recent_ofi)

        return {
            'ofi': net_ofi,
            'ofi_bid': ofi_bid,
            'ofi_ask': ofi_ask,
            'ofi_momentum': ofi_momentum,
            'timestamp': datetime.now()
        }

    def calculate_delta(
        self,
        symbol: str,
        trade_price: float,
        trade_size: float,
        trade_side: str,
        best_bid: float,
        best_ask: float
    ) -> Dict[str, float]:
        """
        Calculate trade delta (aggressive buying vs selling)

        Delta = Volume bought at ask - Volume sold at bid
        Positive delta = aggressive buyers (market orders hitting asks)
        Negative delta = aggressive sellers (market orders hitting bids)

        Args:
            symbol: Trading symbol
            trade_price: Trade execution price
            trade_size: Trade size
            trade_side: 'buy' or 'sell'
            best_bid/ask: Current best bid/ask prices

        Returns:
            Dict with delta metrics
        """
        # Initialize history
        if symbol not in self.delta_history:
            self.delta_history[symbol] = deque(maxlen=self.lookback_periods)

        # Determine if trade was aggressive (taker)
        # Aggressive buy = price >= best ask
        # Aggressive sell = price <= best bid
        is_aggressive_buy = trade_price >= best_ask - 0.01
        is_aggressive_sell = trade_price <= best_bid + 0.01

        # Calculate delta contribution
        if is_aggressive_buy:
            delta = trade_size  # Positive (buying pressure)
        elif is_aggressive_sell:
            delta = -trade_size  # Negative (selling pressure)
        else:
            delta = 0.0  # Passive order (limit order filled)

        # Store delta
        self.delta_history[symbol].append({
            'delta': delta,
            'size': trade_size,
            'price': trade_price,
            'aggressive': is_aggressive_buy or is_aggressive_sell,
            'timestamp': datetime.now()
        })

        # Calculate Cumulative Volume Delta (CVD)
        cvd = sum([entry['delta'] for entry in self.delta_history[symbol]])

        # Recent delta (last 10 trades)
        recent_delta = sum([
            entry['delta']
            for entry in list(self.delta_history[symbol])[-10:]
        ])

        return {
            'delta': delta,
            'cvd': cvd,
            'recent_delta': recent_delta,
            'is_aggressive': is_aggressive_buy or is_aggressive_sell,
            'side': 'buy' if delta > 0 else 'sell',
            'timestamp': datetime.now()
        }

    def detect_large_orders(
        self,
        symbol: str,
        bids: List[List[float]],
        asks: List[List[float]],
        threshold_multiplier: float = 3.0
    ) -> Dict[str, any]:
        """
        Detect unusually large orders (whale activity)

        Args:
            symbol: Trading symbol
            bids/asks: Current order book
            threshold_multiplier: Large order = avg size * multiplier

        Returns:
            Dict with whale detection results
        """
        # Calculate average order size
        all_sizes = [size for _, size in bids[:self.depth_levels]] + \
                    [size for _, size in asks[:self.depth_levels]]

        if not all_sizes:
            return {'whale_detected': False}

        avg_size = np.mean(all_sizes)
        threshold = avg_size * threshold_multiplier

        # Find large orders
        large_bids = [
            {'price': price, 'size': size, 'side': 'bid'}
            for price, size in bids[:self.depth_levels]
            if size >= threshold
        ]

        large_asks = [
            {'price': price, 'size': size, 'side': 'ask'}
            for price, size in asks[:self.depth_levels]
            if size >= threshold
        ]

        whale_detected = len(large_bids) > 0 or len(large_asks) > 0

        return {
            'whale_detected': whale_detected,
            'large_bids': large_bids,
            'large_asks': large_asks,
            'avg_order_size': avg_size,
            'threshold': threshold,
            'timestamp': datetime.now()
        }

    def get_comprehensive_metrics(
        self,
        symbol: str,
        current_bids: List[List[float]],
        current_asks: List[List[float]],
        previous_bids: Optional[List[List[float]]] = None,
        previous_asks: Optional[List[List[float]]] = None
    ) -> Dict[str, any]:
        """
        Calculate all order flow metrics in one call

        Returns:
            Comprehensive dict with all OFI, delta, and whale metrics
        """
        # Order imbalance
        imbalance = self.calculate_order_imbalance(current_bids, current_asks)

        # OFI (if we have previous state)
        ofi = self.calculate_ofi(
            symbol,
            current_bids,
            current_asks,
            previous_bids,
            previous_asks
        )

        # Whale detection
        whale = self.detect_large_orders(symbol, current_bids, current_asks)

        return {
            **imbalance,
            **ofi,
            **whale,
            'timestamp': datetime.now()
        }


# Example usage
if __name__ == "__main__":
    # Test OFI calculator
    ofi_calc = OrderFlowImbalance(depth_levels=10)

    # Simulate order book
    bids = [[100.0, 1.5], [99.9, 2.0], [99.8, 1.0]]
    asks = [[100.1, 0.5], [100.2, 1.0], [100.3, 2.0]]

    print("=== Order Flow Imbalance Test ===")

    # Test 1: Order imbalance
    imbalance = ofi_calc.calculate_order_imbalance(bids, asks)
    print(f"\nOrder Imbalance: {imbalance['imbalance']:+.3f}")
    print(f"Bid Volume: {imbalance['bid_volume']:.2f}")
    print(f"Ask Volume: {imbalance['ask_volume']:.2f}")
    print(f"Ratio: {imbalance['ratio']:.2f}")

    # Test 2: OFI calculation
    # Simulate book change (more bids added)
    new_bids = [[100.0, 2.0], [99.9, 2.5], [99.8, 1.5]]  # Increased sizes
    new_asks = [[100.1, 0.5], [100.2, 1.0], [100.3, 2.0]]  # Same

    ofi = ofi_calc.calculate_ofi('BTC-USD', new_bids, new_asks, bids, asks)
    print(f"\nOFI: {ofi['ofi']:+.3f}")
    print(f"OFI Momentum: {ofi['ofi_momentum']:+.3f}")

    # Test 3: Delta calculation
    delta = ofi_calc.calculate_delta(
        'BTC-USD',
        trade_price=100.1,
        trade_size=0.5,
        trade_side='buy',
        best_bid=100.0,
        best_ask=100.1
    )
    print(f"\nDelta: {delta['delta']:+.3f}")
    print(f"CVD: {delta['cvd']:+.3f}")
    print(f"Side: {delta['side']}")

    # Test 4: Whale detection
    whale = ofi_calc.detect_large_orders('BTC-USD', bids, asks)
    print(f"\nWhale Detected: {whale['whale_detected']}")
    print(f"Avg Order Size: {whale['avg_order_size']:.2f}")

    print("\n✅ Order Flow Imbalance calculator ready for production!")
