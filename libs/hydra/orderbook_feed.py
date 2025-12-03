"""
HYDRA 3.0 - Order Book Data Feed

Provides real-time order book analysis from Coinbase Advanced Trade API:
- Bid/Ask spread analysis
- Order book imbalance detection
- Large order (whale) tracking
- Market maker behavior analysis
- Liquidity depth measurement

Enables gladiators to detect:
- Spoofing attempts
- Whale accumulation/distribution
- Support/resistance levels
- Liquidity zones
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from loguru import logger
import json
from pathlib import Path
from libs.data.coinbase_client import get_coinbase_client


class OrderBookAnalyzer:
    """
    Analyzes order book data for market microstructure insights.
    """

    def __init__(self, coinbase_client = None):
        self.client = coinbase_client or get_coinbase_client()

        # Thresholds for detection
        self.whale_order_threshold = 100000  # $100k+ orders
        self.imbalance_threshold = 0.65  # 65% buy or sell pressure
        self.spread_alert_threshold = 0.002  # 0.2% spread = concerning

        logger.info("Order Book Analyzer initialized")

    def analyze_orderbook(self, symbol: str, depth: int = 50) -> Dict:
        """
        Analyze order book for a symbol.

        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            depth: Number of levels to analyze (default: 50)

        Returns:
            Dict with order book analysis
        """
        try:
            # Get order book from Coinbase
            orderbook = self.client.get_product_book(symbol, limit=depth)

            if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
                logger.warning(f"Invalid order book data for {symbol}")
                return self._get_empty_analysis(symbol)

            # Parse bids and asks
            bids = self._parse_orders(orderbook['bids'])
            asks = self._parse_orders(orderbook['asks'])

            if not bids or not asks:
                return self._get_empty_analysis(symbol)

            # Calculate metrics
            spread_analysis = self._analyze_spread(bids, asks)
            imbalance = self._calculate_imbalance(bids, asks)
            whale_orders = self._detect_whale_orders(bids, asks)
            liquidity = self._measure_liquidity(bids, asks)
            support_resistance = self._find_levels(bids, asks)

            analysis = {
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "spread": spread_analysis,
                "imbalance": imbalance,
                "whale_orders": whale_orders,
                "liquidity": liquidity,
                "levels": support_resistance,
                "summary": self._generate_summary(spread_analysis, imbalance, whale_orders, liquidity)
            }

            logger.debug(f"Order book analyzed for {symbol}: {analysis['summary']}")
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing order book for {symbol}: {e}")
            return self._get_empty_analysis(symbol)

    def _parse_orders(self, orders: List) -> List[Dict]:
        """Parse order book orders into structured format."""
        parsed = []
        for order in orders:
            try:
                # Coinbase format: [price, size, num_orders]
                parsed.append({
                    "price": float(order[0]),
                    "size": float(order[1]),
                    "value": float(order[0]) * float(order[1]),
                    "num_orders": int(order[2]) if len(order) > 2 else 1
                })
            except (ValueError, IndexError) as e:
                logger.debug(f"Error parsing order: {e}")
                continue

        return parsed

    def _analyze_spread(self, bids: List[Dict], asks: List[Dict]) -> Dict:
        """Analyze bid-ask spread."""
        if not bids or not asks:
            return {"spread_pct": 0.0, "spread_usd": 0.0, "status": "unknown"}

        best_bid = bids[0]["price"]
        best_ask = asks[0]["price"]

        spread_usd = best_ask - best_bid
        spread_pct = (spread_usd / best_bid) * 100

        # Determine status
        if spread_pct < 0.05:  # < 0.05%
            status = "tight"
        elif spread_pct < 0.1:  # < 0.1%
            status = "normal"
        elif spread_pct < self.spread_alert_threshold * 100:  # < 0.2%
            status = "wide"
        else:
            status = "concerning"

        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread_usd": spread_usd,
            "spread_pct": spread_pct,
            "status": status
        }

    def _calculate_imbalance(self, bids: List[Dict], asks: List[Dict]) -> Dict:
        """Calculate order book imbalance (buy vs sell pressure)."""
        if not bids or not asks:
            return {"ratio": 0.5, "direction": "neutral", "strength": "none"}

        # Sum total value on each side
        bid_value = sum(o["value"] for o in bids)
        ask_value = sum(o["value"] for o in asks)
        total_value = bid_value + ask_value

        if total_value == 0:
            return {"ratio": 0.5, "direction": "neutral", "strength": "none"}

        # Buy pressure ratio
        buy_ratio = bid_value / total_value

        # Determine direction and strength
        if buy_ratio > self.imbalance_threshold:
            direction = "buy"
            strength = "strong" if buy_ratio > 0.75 else "moderate"
        elif buy_ratio < (1 - self.imbalance_threshold):
            direction = "sell"
            strength = "strong" if buy_ratio < 0.25 else "moderate"
        else:
            direction = "neutral"
            strength = "balanced"

        return {
            "bid_value": bid_value,
            "ask_value": ask_value,
            "ratio": buy_ratio,
            "direction": direction,
            "strength": strength
        }

    def _detect_whale_orders(self, bids: List[Dict], asks: List[Dict]) -> Dict:
        """Detect large (whale) orders."""
        whale_bids = [o for o in bids if o["value"] >= self.whale_order_threshold]
        whale_asks = [o for o in asks if o["value"] >= self.whale_order_threshold]

        total_whale_value = sum(o["value"] for o in whale_bids + whale_asks)

        return {
            "whale_bids": len(whale_bids),
            "whale_asks": len(whale_asks),
            "total_whale_value": total_whale_value,
            "largest_bid": max([o["value"] for o in whale_bids], default=0),
            "largest_ask": max([o["value"] for o in whale_asks], default=0),
            "whale_direction": "buy" if len(whale_bids) > len(whale_asks) else "sell" if len(whale_asks) > len(whale_bids) else "neutral"
        }

    def _measure_liquidity(self, bids: List[Dict], asks: List[Dict]) -> Dict:
        """Measure liquidity depth."""
        if not bids or not asks:
            return {"depth_usd": 0, "status": "low"}

        # Calculate total liquidity within 1% of mid price
        mid_price = (bids[0]["price"] + asks[0]["price"]) / 2
        one_percent_range = mid_price * 0.01

        near_bids = [o for o in bids if mid_price - o["price"] <= one_percent_range]
        near_asks = [o for o in asks if o["price"] - mid_price <= one_percent_range]

        depth_usd = sum(o["value"] for o in near_bids + near_asks)

        # Determine liquidity status
        if depth_usd > 10_000_000:  # $10M+
            status = "excellent"
        elif depth_usd > 5_000_000:  # $5M+
            status = "good"
        elif depth_usd > 1_000_000:  # $1M+
            status = "moderate"
        elif depth_usd > 500_000:  # $500k+
            status = "low"
        else:
            status = "very_low"

        return {
            "depth_usd": depth_usd,
            "near_bids": len(near_bids),
            "near_asks": len(near_asks),
            "status": status
        }

    def _find_levels(self, bids: List[Dict], asks: List[Dict]) -> Dict:
        """Find significant support/resistance levels."""
        # Identify price levels with large orders (potential support/resistance)
        support_levels = []
        resistance_levels = []

        # Support: large bid walls
        for bid in bids[:10]:  # Top 10 bids
            if bid["value"] > self.whale_order_threshold:
                support_levels.append({
                    "price": bid["price"],
                    "value": bid["value"]
                })

        # Resistance: large ask walls
        for ask in asks[:10]:  # Top 10 asks
            if ask["value"] > self.whale_order_threshold:
                resistance_levels.append({
                    "price": ask["price"],
                    "value": ask["value"]
                })

        return {
            "support_levels": support_levels[:3],  # Top 3
            "resistance_levels": resistance_levels[:3],  # Top 3
            "strongest_support": support_levels[0]["price"] if support_levels else None,
            "strongest_resistance": resistance_levels[0]["price"] if resistance_levels else None
        }

    def _generate_summary(
        self,
        spread: Dict,
        imbalance: Dict,
        whale_orders: Dict,
        liquidity: Dict
    ) -> str:
        """Generate human-readable summary."""
        parts = []

        # Spread
        parts.append(f"Spread: {spread['status']} ({spread['spread_pct']:.3f}%)")

        # Imbalance
        if imbalance['direction'] != 'neutral':
            parts.append(f"{imbalance['strength']} {imbalance['direction']} pressure")

        # Whale activity
        if whale_orders['whale_bids'] + whale_orders['whale_asks'] > 0:
            parts.append(f"{whale_orders['whale_bids'] + whale_orders['whale_asks']} whale orders detected")

        # Liquidity
        parts.append(f"Liquidity: {liquidity['status']}")

        return " | ".join(parts)

    def _get_empty_analysis(self, symbol: str) -> Dict:
        """Return empty analysis structure."""
        return {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "spread": {"spread_pct": 0.0, "status": "unknown"},
            "imbalance": {"ratio": 0.5, "direction": "neutral", "strength": "none"},
            "whale_orders": {"whale_bids": 0, "whale_asks": 0, "total_whale_value": 0},
            "liquidity": {"depth_usd": 0, "status": "unknown"},
            "levels": {"support_levels": [], "resistance_levels": []},
            "summary": "No data available"
        }

    def get_orderbook_summary_for_prompt(self, symbol: str) -> str:
        """
        Get formatted order book summary for gladiator prompts.

        Args:
            symbol: Trading symbol

        Returns:
            Formatted string for prompt injection
        """
        analysis = self.analyze_orderbook(symbol)

        summary = f"""
ORDER BOOK ANALYSIS ({symbol}):

Spread: {analysis['spread']['status']} ({analysis['spread']['spread_pct']:.3f}%)
Best Bid: ${analysis['spread'].get('best_bid', 0):,.2f}
Best Ask: ${analysis['spread'].get('best_ask', 0):,.2f}

Market Pressure:
- Direction: {analysis['imbalance']['direction']}
- Strength: {analysis['imbalance']['strength']}
- Buy/Sell Ratio: {analysis['imbalance']['ratio']:.1%}

Whale Activity:
- Large Bids: {analysis['whale_orders']['whale_bids']}
- Large Asks: {analysis['whale_orders']['whale_asks']}
- Direction: {analysis['whale_orders']['whale_direction']}

Liquidity:
- Depth (±1%): ${analysis['liquidity']['depth_usd']:,.0f}
- Status: {analysis['liquidity']['status']}

Key Levels:
- Support: {self._format_levels(analysis['levels']['support_levels'])}
- Resistance: {self._format_levels(analysis['levels']['resistance_levels'])}

Summary: {analysis['summary']}
"""
        return summary.strip()

    def _format_levels(self, levels: List[Dict]) -> str:
        """Format support/resistance levels."""
        if not levels:
            return "None detected"

        return ", ".join([f"${level['price']:,.2f}" for level in levels[:3]])


# ==================== SINGLETON PATTERN ====================

_orderbook_analyzer = None

def get_orderbook_analyzer() -> OrderBookAnalyzer:
    """Get singleton instance of OrderBookAnalyzer."""
    global _orderbook_analyzer
    if _orderbook_analyzer is None:
        _orderbook_analyzer = OrderBookAnalyzer()
    return _orderbook_analyzer
