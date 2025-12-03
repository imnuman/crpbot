"""
HYDRA 3.0 - Order Book Data Feed

Provides real-time order book analysis:
- Bid/Ask spread and depth
- Order book imbalance (buy vs sell pressure)
- Support/Resistance detection from order clusters
- Whale order detection
- Liquidity analysis

Uses Coinbase API for order book data.
"""

import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import httpx

logger = logging.getLogger(__name__)


# Configuration
COINBASE_API_URL = "https://api.exchange.coinbase.com"
ORDER_BOOK_LEVELS = 50  # Depth levels to fetch
WHALE_ORDER_THRESHOLD_BTC = 1.0  # Orders > 1 BTC are whale orders
WHALE_ORDER_THRESHOLD_USD = 50000  # Orders > $50k are whale orders
CLUSTER_THRESHOLD_PCT = 0.5  # Price levels within 0.5% grouped as cluster


@dataclass
class OrderLevel:
    """A single price level in the order book."""
    price: float
    size: float
    num_orders: int = 1

    @property
    def value_usd(self) -> float:
        return self.price * self.size


@dataclass
class OrderBookMetrics:
    """Computed metrics from order book."""
    symbol: str
    timestamp: datetime

    # Spread
    best_bid: float
    best_ask: float
    spread: float
    spread_pct: float
    mid_price: float

    # Depth (total volume at top N levels)
    bid_depth_5: float  # Total bid volume in top 5 levels
    ask_depth_5: float
    bid_depth_20: float
    ask_depth_20: float

    # Imbalance (-1 to +1, positive = more bids/buying pressure)
    imbalance_5: float
    imbalance_20: float

    # Whale orders
    whale_bids: int
    whale_asks: int
    whale_bid_volume: float
    whale_ask_volume: float

    # Support/Resistance clusters
    support_levels: list[float]
    resistance_levels: list[float]

    # Liquidity score (0-100)
    liquidity_score: float

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "timestamp": self.timestamp.isoformat(),
        }

    def format_for_llm(self) -> str:
        """Format metrics for LLM consumption."""
        # Determine market bias
        if self.imbalance_20 > 0.2:
            bias = "BULLISH (strong buying pressure)"
        elif self.imbalance_20 > 0.05:
            bias = "SLIGHTLY BULLISH"
        elif self.imbalance_20 < -0.2:
            bias = "BEARISH (strong selling pressure)"
        elif self.imbalance_20 < -0.05:
            bias = "SLIGHTLY BEARISH"
        else:
            bias = "NEUTRAL"

        lines = [
            f"=== Order Book: {self.symbol} ===",
            f"Mid Price: ${self.mid_price:,.2f}",
            f"Spread: ${self.spread:.2f} ({self.spread_pct:.3f}%)",
            f"",
            f"Order Book Imbalance: {self.imbalance_20:+.2f} ({bias})",
            f"  Top 5 levels: Bids {self.bid_depth_5:.4f} / Asks {self.ask_depth_5:.4f}",
            f"  Top 20 levels: Bids {self.bid_depth_20:.4f} / Asks {self.ask_depth_20:.4f}",
            f"",
            f"Whale Activity:",
            f"  Bid whales: {self.whale_bids} orders ({self.whale_bid_volume:.2f} units)",
            f"  Ask whales: {self.whale_asks} orders ({self.whale_ask_volume:.2f} units)",
            f"",
            f"Liquidity Score: {self.liquidity_score:.0f}/100",
        ]

        if self.support_levels:
            lines.append(f"Support Clusters: {', '.join(f'${p:,.0f}' for p in self.support_levels[:3])}")
        if self.resistance_levels:
            lines.append(f"Resistance Clusters: {', '.join(f'${p:,.0f}' for p in self.resistance_levels[:3])}")

        return "\n".join(lines)


class OrderBookFeed:
    """
    Order book data feed using Coinbase API.

    Features:
    - Real-time order book snapshots
    - Bid/ask spread analysis
    - Order book imbalance calculation
    - Support/resistance detection
    - Whale order identification
    """

    _instance: Optional["OrderBookFeed"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, data_dir: Optional[Path] = None):
        if self._initialized:
            return

        # Auto-detect data directory
        if data_dir is None:
            if os.path.exists("/root/crpbot"):
                data_dir = Path("/root/crpbot/data/hydra")
            else:
                data_dir = Path.home() / "crpbot" / "data" / "hydra"

        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Cache for recent snapshots
        self.cache: dict[str, tuple[datetime, OrderBookMetrics]] = {}
        self.cache_ttl_seconds = 5  # Cache for 5 seconds

        self._initialized = True
        logger.info("OrderBookFeed initialized")

    async def fetch_order_book(self, symbol: str = "BTC-USD") -> Optional[dict]:
        """
        Fetch raw order book from Coinbase.

        Args:
            symbol: Trading pair (e.g., "BTC-USD")

        Returns:
            Raw order book data or None on error
        """
        url = f"{COINBASE_API_URL}/products/{symbol}/book"
        params = {"level": 2}  # Level 2 = aggregated by price

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch order book for {symbol}: {e}")
            return None

    def _parse_levels(self, raw_levels: list) -> list[OrderLevel]:
        """Parse raw order book levels."""
        levels = []
        for level in raw_levels:
            try:
                price = float(level[0])
                size = float(level[1])
                num_orders = int(level[2]) if len(level) > 2 else 1
                levels.append(OrderLevel(price, size, num_orders))
            except (ValueError, IndexError):
                continue
        return levels

    def _calculate_imbalance(self, bids: list[OrderLevel], asks: list[OrderLevel], n: int) -> float:
        """
        Calculate order book imbalance.

        Returns:
            Value from -1 (all asks) to +1 (all bids)
        """
        bid_vol = sum(b.size for b in bids[:n])
        ask_vol = sum(a.size for a in asks[:n])
        total = bid_vol + ask_vol

        if total == 0:
            return 0.0

        return (bid_vol - ask_vol) / total

    def _find_clusters(
        self,
        levels: list[OrderLevel],
        mid_price: float,
        threshold_pct: float = CLUSTER_THRESHOLD_PCT
    ) -> list[float]:
        """
        Find price clusters where orders are concentrated.

        Returns:
            List of cluster center prices
        """
        if not levels:
            return []

        # Group levels into clusters
        clusters: list[list[OrderLevel]] = []
        current_cluster: list[OrderLevel] = []

        for level in sorted(levels, key=lambda x: x.price):
            if not current_cluster:
                current_cluster.append(level)
            else:
                # Check if this level is within threshold of cluster
                cluster_price = sum(l.price for l in current_cluster) / len(current_cluster)
                if abs(level.price - cluster_price) / mid_price * 100 < threshold_pct:
                    current_cluster.append(level)
                else:
                    if len(current_cluster) >= 2:  # Meaningful cluster
                        clusters.append(current_cluster)
                    current_cluster = [level]

        if len(current_cluster) >= 2:
            clusters.append(current_cluster)

        # Sort by total volume and return center prices
        cluster_volumes = []
        for cluster in clusters:
            total_vol = sum(l.size for l in cluster)
            center_price = sum(l.price * l.size for l in cluster) / total_vol
            cluster_volumes.append((center_price, total_vol))

        cluster_volumes.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in cluster_volumes[:5]]  # Top 5 clusters

    def _detect_whales(
        self,
        levels: list[OrderLevel],
        mid_price: float
    ) -> tuple[int, float]:
        """
        Detect whale orders.

        Returns:
            (count, total_volume)
        """
        # Dynamic threshold based on price
        threshold = max(
            WHALE_ORDER_THRESHOLD_BTC,
            WHALE_ORDER_THRESHOLD_USD / mid_price
        )

        whales = [l for l in levels if l.size >= threshold]
        return len(whales), sum(w.size for w in whales)

    def _calculate_liquidity_score(
        self,
        spread_pct: float,
        bid_depth: float,
        ask_depth: float,
        mid_price: float
    ) -> float:
        """
        Calculate liquidity score (0-100).

        Based on:
        - Tight spread = good
        - High depth = good
        - Balanced book = good
        """
        # Spread score (0-40 points)
        # < 0.01% = 40, > 0.5% = 0
        spread_score = max(0, min(40, 40 * (1 - spread_pct / 0.5)))

        # Depth score (0-40 points)
        # Based on USD value of top 20 levels
        total_depth_usd = (bid_depth + ask_depth) * mid_price
        # > $10M = 40, < $100k = 0
        depth_score = max(0, min(40, 40 * min(1, total_depth_usd / 10_000_000)))

        # Balance score (0-20 points)
        # More balanced = better
        if bid_depth + ask_depth > 0:
            balance = 1 - abs(bid_depth - ask_depth) / (bid_depth + ask_depth)
        else:
            balance = 0
        balance_score = balance * 20

        return spread_score + depth_score + balance_score

    async def get_metrics(self, symbol: str = "BTC-USD") -> Optional[OrderBookMetrics]:
        """
        Get computed order book metrics.

        Args:
            symbol: Trading pair

        Returns:
            OrderBookMetrics or None on error
        """
        # Check cache
        if symbol in self.cache:
            cached_time, cached_metrics = self.cache[symbol]
            age = (datetime.now() - cached_time).total_seconds()
            if age < self.cache_ttl_seconds:
                return cached_metrics

        # Fetch fresh data
        raw = await self.fetch_order_book(symbol)
        if not raw:
            return None

        # Parse levels
        bids = self._parse_levels(raw.get("bids", []))
        asks = self._parse_levels(raw.get("asks", []))

        if not bids or not asks:
            logger.warning(f"Empty order book for {symbol}")
            return None

        # Basic metrics
        best_bid = bids[0].price
        best_ask = asks[0].price
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        spread_pct = (spread / mid_price) * 100

        # Depth calculations
        bid_depth_5 = sum(b.size for b in bids[:5])
        ask_depth_5 = sum(a.size for a in asks[:5])
        bid_depth_20 = sum(b.size for b in bids[:20])
        ask_depth_20 = sum(a.size for a in asks[:20])

        # Imbalance
        imbalance_5 = self._calculate_imbalance(bids, asks, 5)
        imbalance_20 = self._calculate_imbalance(bids, asks, 20)

        # Whale detection
        whale_bids, whale_bid_vol = self._detect_whales(bids, mid_price)
        whale_asks, whale_ask_vol = self._detect_whales(asks, mid_price)

        # Support/Resistance clusters
        support_levels = self._find_clusters(bids, mid_price)
        resistance_levels = self._find_clusters(asks, mid_price)

        # Liquidity score
        liquidity_score = self._calculate_liquidity_score(
            spread_pct, bid_depth_20, ask_depth_20, mid_price
        )

        metrics = OrderBookMetrics(
            symbol=symbol,
            timestamp=datetime.now(),
            best_bid=best_bid,
            best_ask=best_ask,
            spread=spread,
            spread_pct=spread_pct,
            mid_price=mid_price,
            bid_depth_5=bid_depth_5,
            ask_depth_5=ask_depth_5,
            bid_depth_20=bid_depth_20,
            ask_depth_20=ask_depth_20,
            imbalance_5=imbalance_5,
            imbalance_20=imbalance_20,
            whale_bids=whale_bids,
            whale_asks=whale_asks,
            whale_bid_volume=whale_bid_vol,
            whale_ask_volume=whale_ask_vol,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            liquidity_score=liquidity_score,
        )

        # Cache
        self.cache[symbol] = (datetime.now(), metrics)

        return metrics

    async def get_multi_symbol_metrics(
        self,
        symbols: Optional[list[str]] = None
    ) -> dict[str, OrderBookMetrics]:
        """
        Get metrics for multiple symbols.

        Args:
            symbols: List of trading pairs (default: BTC, ETH, SOL)

        Returns:
            Dict of symbol -> metrics
        """
        symbols = symbols or ["BTC-USD", "ETH-USD", "SOL-USD"]
        results = {}

        for symbol in symbols:
            metrics = await self.get_metrics(symbol)
            if metrics:
                results[symbol] = metrics

        return results

    def format_multi_for_llm(self, metrics: dict[str, OrderBookMetrics]) -> str:
        """Format multiple order books for LLM."""
        lines = ["=== Order Book Analysis ===\n"]

        for symbol, m in metrics.items():
            lines.append(m.format_for_llm())
            lines.append("")

        return "\n".join(lines)

    def get_trading_signal(self, metrics: OrderBookMetrics) -> dict:
        """
        Generate a simple trading signal from order book.

        Returns:
            Dict with signal info
        """
        signal = {
            "symbol": metrics.symbol,
            "direction": "NEUTRAL",
            "strength": 0.0,
            "reasons": [],
        }

        reasons = []
        strength = 0.0

        # Imbalance signal
        if metrics.imbalance_20 > 0.3:
            strength += 0.3
            reasons.append(f"Strong bid imbalance: {metrics.imbalance_20:.2f}")
        elif metrics.imbalance_20 < -0.3:
            strength -= 0.3
            reasons.append(f"Strong ask imbalance: {metrics.imbalance_20:.2f}")

        # Whale activity
        if metrics.whale_bid_volume > metrics.whale_ask_volume * 1.5:
            strength += 0.2
            reasons.append("Whale buying pressure detected")
        elif metrics.whale_ask_volume > metrics.whale_bid_volume * 1.5:
            strength -= 0.2
            reasons.append("Whale selling pressure detected")

        # Tight spread = more reliable
        if metrics.spread_pct < 0.05:
            reasons.append("Very tight spread - high liquidity")
        elif metrics.spread_pct > 0.2:
            reasons.append("Wide spread - caution advised")

        # Determine direction
        if strength > 0.2:
            signal["direction"] = "LONG"
        elif strength < -0.2:
            signal["direction"] = "SHORT"

        signal["strength"] = abs(strength)
        signal["reasons"] = reasons

        return signal


# Singleton accessor
_orderbook_instance: Optional[OrderBookFeed] = None


def get_order_book_feed() -> OrderBookFeed:
    """Get or create the order book feed singleton."""
    global _orderbook_instance
    if _orderbook_instance is None:
        _orderbook_instance = OrderBookFeed()
    return _orderbook_instance
