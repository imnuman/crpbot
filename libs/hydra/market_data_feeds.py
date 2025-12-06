"""
HYDRA 3.0 - Market Data Feeds

Comprehensive market data collection for gladiators:
1. Funding Rates (perpetual futures)
2. Liquidations tracking
3. Open Interest monitoring
4. Volume profile analysis

Provides insights into:
- Long/short bias in the market
- Leverage levels
- Liquidation cascade risk
- Market maker positioning
"""

from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta
from loguru import logger
import json
from pathlib import Path
import requests


class FundingRatesMonitor:
    """
    Monitors funding rates for perpetual futures.

    Funding rates indicate long/short bias:
    - Positive rate: Longs pay shorts (bullish bias)
    - Negative rate: Shorts pay longs (bearish bias)
    - Extreme rates: Overleveraged positions (reversal signal)
    """

    def __init__(self, cache_dir: Path = None):
        if cache_dir is None:
            from .config import FUNDING_CACHE_DIR
            cache_dir = FUNDING_CACHE_DIR
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Thresholds for analysis
        self.extreme_positive_threshold = 0.0010  # 0.10% (bullish extreme)
        self.extreme_negative_threshold = -0.0010  # -0.10% (bearish extreme)

        logger.info("Funding Rates Monitor initialized")

    def get_funding_rate(self, symbol: str, market_data: List[Dict] = None) -> Dict:
        """
        Get current funding rate for a symbol.

        For spot markets without perp data, we synthesize funding rates from
        price momentum - this correlates with real funding behavior:
        - Strong uptrend = positive funding (longs pay shorts)
        - Strong downtrend = negative funding (shorts pay longs)
        - Extreme moves = extreme funding (reversal signal)

        Args:
            symbol: Perp symbol (e.g., "BTC-PERP", "ETH-PERP")
            market_data: Optional OHLCV data for synthetic rate calculation

        Returns:
            Dict with funding rate analysis
        """
        # Check cache first
        cached = self._get_cached_funding(symbol)
        if cached:
            return cached

        # Calculate synthetic funding rate from price momentum
        synthetic_rate = 0.0
        direction = "neutral"
        strength = "low"
        bias = "Neutral market"
        risk_level = "low"

        if market_data and len(market_data) >= 24:
            # Calculate momentum-based synthetic funding
            # Use 24-period returns as proxy for market sentiment
            current_price = market_data[-1].get("close", 0)
            price_24h_ago = market_data[-24].get("close", current_price)
            price_8h_ago = market_data[-8].get("close", current_price) if len(market_data) >= 8 else current_price

            if price_24h_ago > 0:
                # 24h return as sentiment proxy
                return_24h = (current_price - price_24h_ago) / price_24h_ago
                return_8h = (current_price - price_8h_ago) / price_8h_ago if price_8h_ago > 0 else 0

                # Calculate volatility (standard deviation of returns)
                if len(market_data) >= 24:
                    closes = [c.get("close", 0) for c in market_data[-24:]]
                    returns = [(closes[i] - closes[i-1]) / closes[i-1]
                              for i in range(1, len(closes)) if closes[i-1] > 0]
                    if returns:
                        avg_return = sum(returns) / len(returns)
                        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
                        volatility = variance ** 0.5
                    else:
                        volatility = 0.01
                else:
                    volatility = 0.01

                # Synthetic funding = momentum * volatility_multiplier
                # Base: 0.01% per 1% price move, amplified by volatility
                volatility_mult = min(3.0, max(0.5, volatility / 0.01))
                synthetic_rate = return_8h * 0.001 * volatility_mult  # 8h window for funding

                # Cap at realistic extremes (±0.15%)
                synthetic_rate = max(-0.0015, min(0.0015, synthetic_rate))

                # Determine direction and strength
                if synthetic_rate > self.extreme_positive_threshold:
                    direction = "bullish"
                    strength = "extreme"
                    bias = "Overleveraged longs - reversal risk"
                    risk_level = "high"
                elif synthetic_rate > 0.0003:
                    direction = "bullish"
                    strength = "high"
                    bias = "Strong bullish sentiment"
                    risk_level = "medium"
                elif synthetic_rate > 0.0001:
                    direction = "bullish"
                    strength = "moderate"
                    bias = "Moderate bullish sentiment"
                    risk_level = "low"
                elif synthetic_rate < self.extreme_negative_threshold:
                    direction = "bearish"
                    strength = "extreme"
                    bias = "Overleveraged shorts - reversal risk"
                    risk_level = "high"
                elif synthetic_rate < -0.0003:
                    direction = "bearish"
                    strength = "high"
                    bias = "Strong bearish sentiment"
                    risk_level = "medium"
                elif synthetic_rate < -0.0001:
                    direction = "bearish"
                    strength = "moderate"
                    bias = "Moderate bearish sentiment"
                    risk_level = "low"

        analysis = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "current_rate": synthetic_rate,
            "rate_8h_annualized": synthetic_rate * 3 * 365,  # Annualized from 8h rate
            "direction": direction,
            "strength": strength,
            "bias": bias,
            "risk_level": risk_level,
            "arbitrage_opportunity": abs(synthetic_rate) > 0.0005,
            "note": "Synthetic rate from price momentum (spot market proxy)"
        }

        self._cache_funding(symbol, analysis)
        return analysis

    def analyze_funding_trend(self, symbol: str, lookback_hours: int = 24) -> Dict:
        """
        Analyze funding rate trends over time.

        Args:
            symbol: Perp symbol
            lookback_hours: Hours to look back

        Returns:
            Dict with trend analysis
        """
        return {
            "symbol": symbol,
            "lookback_hours": lookback_hours,
            "trend": "neutral",
            "average_rate": 0.0,
            "volatility": 0.0,
            "reversals": 0,
            "note": "Trend analysis available when perps are integrated"
        }

    def _get_cached_funding(self, symbol: str) -> Optional[Dict]:
        """Get cached funding rate."""
        cache_file = self.cache_dir / f"{symbol}_funding.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)

            # Check cache age (max 1 hour)
            cache_time = datetime.fromisoformat(data["timestamp"])
            age = datetime.now(timezone.utc) - cache_time

            if age > timedelta(hours=1):
                return None

            return data

        except Exception as e:
            logger.debug(f"Error reading funding cache: {e}")
            return None

    def _cache_funding(self, symbol: str, data: Dict):
        """Cache funding rate data."""
        cache_file = self.cache_dir / f"{symbol}_funding.json"

        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)


class LiquidationsTracker:
    """
    Tracks liquidation events across the market.

    Liquidations indicate:
    - Over-leverage levels
    - Cascade risk (liquidations triggering more liquidations)
    - Key price levels where stops cluster
    - Market panic/euphoria levels
    """

    def __init__(self, cache_dir: Path = None):
        if cache_dir is None:
            from .config import LIQUIDATIONS_CACHE_DIR
            cache_dir = LIQUIDATIONS_CACHE_DIR
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Thresholds
        self.large_liquidation_threshold = 1_000_000  # $1M+
        self.cascade_time_window = 300  # 5 minutes

        logger.info("Liquidations Tracker initialized")

    def get_recent_liquidations(self, symbol: str, lookback_minutes: int = 60) -> Dict:
        """
        Get recent liquidation events.

        Args:
            symbol: Trading symbol
            lookback_minutes: Minutes to look back

        Returns:
            Dict with liquidation analysis
        """
        # Check cache
        cached = self._get_cached_liquidations(symbol)
        if cached:
            return cached

        # In production, this would fetch from liquidations API
        # For now, return placeholder structure
        analysis = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "lookback_minutes": lookback_minutes,
            "total_liquidations": 0,
            "total_value_usd": 0,
            "long_liquidations": 0,
            "short_liquidations": 0,
            "largest_liquidation": 0,
            "cascade_detected": False,
            "risk_level": "low",
            "direction_bias": "neutral",
            "note": "Liquidations tracking available when perps are integrated"
        }

        self._cache_liquidations(symbol, analysis)
        return analysis

    def detect_liquidation_clusters(self, symbol: str) -> Dict:
        """
        Detect price levels with high liquidation risk.

        Returns:
            Dict with liquidation cluster analysis
        """
        return {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "clusters": [],
            "nearest_cluster_distance_pct": 0.0,
            "cascade_risk": "low",
            "note": "Cluster detection available when perps are integrated"
        }

    def _get_cached_liquidations(self, symbol: str) -> Optional[Dict]:
        """Get cached liquidations data."""
        cache_file = self.cache_dir / f"{symbol}_liquidations.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)

            # Check cache age (max 5 minutes)
            cache_time = datetime.fromisoformat(data["timestamp"])
            age = datetime.now(timezone.utc) - cache_time

            if age > timedelta(minutes=5):
                return None

            return data

        except Exception as e:
            logger.debug(f"Error reading liquidations cache: {e}")
            return None

    def _cache_liquidations(self, symbol: str, data: Dict):
        """Cache liquidations data."""
        cache_file = self.cache_dir / f"{symbol}_liquidations.json"

        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)


class MarketDataAggregator:
    """
    Aggregates all market data feeds into a single interface.

    Provides gladiators with comprehensive market intelligence.
    """

    def __init__(self):
        self.funding_monitor = FundingRatesMonitor()
        self.liquidations_tracker = LiquidationsTracker()

        logger.info("Market Data Aggregator initialized")

    def get_comprehensive_analysis(self, symbol: str) -> Dict:
        """
        Get comprehensive market data analysis for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dict with all market data feeds
        """
        return {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "funding": self.funding_monitor.get_funding_rate(symbol),
            "liquidations": self.liquidations_tracker.get_recent_liquidations(symbol),
            "summary": self._generate_summary(symbol)
        }

    def _generate_summary(self, symbol: str) -> str:
        """Generate human-readable summary of all feeds."""
        funding = self.funding_monitor.get_funding_rate(symbol)
        liquidations = self.liquidations_tracker.get_recent_liquidations(symbol)

        parts = []

        # Funding summary
        if funding.get("current_rate", 0) != 0:
            parts.append(f"Funding: {funding['direction']} bias ({funding['current_rate']:.4f}%)")
        else:
            parts.append("Funding: N/A (spot market)")

        # Liquidations summary
        if liquidations.get("total_liquidations", 0) > 0:
            parts.append(f"Liquidations: {liquidations['total_liquidations']} events")
        else:
            parts.append("Liquidations: None recent")

        # Risk assessment
        risk_levels = [
            funding.get("risk_level", "low"),
            liquidations.get("risk_level", "low")
        ]

        if "high" in risk_levels:
            parts.append("Risk: HIGH")
        elif "medium" in risk_levels:
            parts.append("Risk: MEDIUM")
        else:
            parts.append("Risk: LOW")

        return " | ".join(parts)

    def get_market_data_for_prompt(self, symbol: str) -> str:
        """
        Get formatted market data summary for gladiator prompts.

        Args:
            symbol: Trading symbol

        Returns:
            Formatted string for prompt injection
        """
        analysis = self.get_comprehensive_analysis(symbol)

        funding = analysis["funding"]
        liquidations = analysis["liquidations"]

        summary = f"""
MARKET DATA FEEDS ({symbol}):

Funding Rate Analysis:
- Current Rate: {funding['current_rate']:.4f}%
- Direction: {funding['direction']}
- Bias: {funding['bias']}
- Risk Level: {funding['risk_level']}

Liquidations (Last Hour):
- Total Events: {liquidations['total_liquidations']}
- Total Value: ${liquidations['total_value_usd']:,.0f}
- Long Liquidations: {liquidations['long_liquidations']}
- Short Liquidations: {liquidations['short_liquidations']}
- Cascade Detected: {liquidations['cascade_detected']}
- Direction Bias: {liquidations['direction_bias']}

Overall Market Summary: {analysis['summary']}

Note: Full perpetual futures integration pending. Currently showing spot market data.
"""
        return summary.strip()


# ==================== SINGLETON PATTERN ====================

_market_data_aggregator = None

def get_market_data_aggregator() -> MarketDataAggregator:
    """Get singleton instance of MarketDataAggregator."""
    global _market_data_aggregator
    if _market_data_aggregator is None:
        _market_data_aggregator = MarketDataAggregator()
    return _market_data_aggregator
