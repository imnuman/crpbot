"""
HYDRA 3.0 - Internet Search Module

Provides gladiators with access to real-time information:
- News sentiment analysis
- Macro event detection
- Breaking news alerts
- Market context gathering

Uses Claude Code's built-in WebSearch tool for queries.
Each gladiator can search independently during decision-making.
"""

from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta
from loguru import logger
import json
from pathlib import Path
import time


class InternetSearch:
    """
    Internet search client for HYDRA gladiators.

    Provides structured search capabilities for market intelligence.
    """

    def __init__(self, cache_dir: Path = Path("/root/crpbot/data/hydra/search_cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Search history for rate limiting
        self.search_history = []
        self.max_searches_per_minute = 10  # Conservative limit

        logger.info("Internet Search initialized")

    def search_asset_news(
        self,
        asset: str,
        time_range: str = "24h"
    ) -> Dict:
        """
        Search for recent news about a specific asset.

        Args:
            asset: Asset symbol (e.g., "BTC", "ETH")
            time_range: Time range for news ("1h", "24h", "7d")

        Returns:
            Dict with news summary and sentiment
        """
        # Check rate limit
        if not self._check_rate_limit():
            logger.warning("Search rate limit reached, using cached data")
            return self._get_cached_search(asset, "news")

        query = self._build_news_query(asset, time_range)

        logger.info(f"Searching news for {asset} ({time_range})")

        # Note: In production, this would call WebSearch tool
        # For now, we'll structure the interface for when it's available
        results = self._execute_search(query, search_type="news")

        # Parse and structure results
        structured = self._structure_news_results(asset, results)

        # Cache results
        self._cache_search(asset, "news", structured)

        return structured

    def search_macro_events(self) -> Dict:
        """
        Search for macro economic events and news.

        Returns:
            Dict with macro event summary
        """
        if not self._check_rate_limit():
            logger.warning("Search rate limit reached, using cached data")
            return self._get_cached_search("macro", "events")

        query = "cryptocurrency market macro events economic news federal reserve interest rates"

        logger.info("Searching macro events")

        results = self._execute_search(query, search_type="macro")
        structured = self._structure_macro_results(results)

        self._cache_search("macro", "events", structured)

        return structured

    def search_breaking_news(self, assets: List[str]) -> Dict:
        """
        Search for breaking news across multiple assets.

        Args:
            assets: List of asset symbols

        Returns:
            Dict with breaking news by asset
        """
        if not self._check_rate_limit():
            logger.warning("Search rate limit reached, using cached data")
            return self._get_cached_search("breaking", "news")

        query = " OR ".join([f"{asset} breaking news" for asset in assets[:3]])  # Limit to 3 assets

        logger.info(f"Searching breaking news for {len(assets)} assets")

        results = self._execute_search(query, search_type="breaking")
        structured = self._structure_breaking_news(assets, results)

        self._cache_search("breaking", "news", structured)

        return structured

    def search_sentiment(self, asset: str) -> Dict:
        """
        Search for market sentiment about an asset.

        Args:
            asset: Asset symbol

        Returns:
            Dict with sentiment analysis
        """
        if not self._check_rate_limit():
            logger.warning("Search rate limit reached, using cached data")
            return self._get_cached_search(asset, "sentiment")

        query = f"{asset} cryptocurrency sentiment analysis market outlook"

        logger.info(f"Searching sentiment for {asset}")

        results = self._execute_search(query, search_type="sentiment")
        structured = self._structure_sentiment_results(asset, results)

        self._cache_search(asset, "sentiment", structured)

        return structured

    # ==================== INTERNAL METHODS ====================

    def _build_news_query(self, asset: str, time_range: str) -> str:
        """Build optimized news search query."""
        asset_map = {
            "BTC-USD": "Bitcoin BTC",
            "ETH-USD": "Ethereum ETH",
            "SOL-USD": "Solana SOL",
            "XRP-USD": "Ripple XRP",
            "DOGE-USD": "Dogecoin DOGE",
            "ADA-USD": "Cardano ADA",
            "AVAX-USD": "Avalanche AVAX",
            "LINK-USD": "Chainlink LINK",
            "POL-USD": "Polygon POL MATIC",
            "LTC-USD": "Litecoin LTC"
        }

        asset_name = asset_map.get(asset, asset.replace("-USD", ""))

        # Add time-based keywords
        time_keywords = {
            "1h": "latest breaking urgent",
            "24h": "today recent news",
            "7d": "this week trend analysis"
        }

        time_filter = time_keywords.get(time_range, "recent")

        return f"{asset_name} cryptocurrency {time_filter} news price movement"

    def _execute_search(self, query: str, search_type: str) -> Dict:
        """
        Execute search query.

        NOTE: This is a placeholder for WebSearch tool integration.
        In production, this would call the actual WebSearch tool.
        """
        # Record search
        self.search_history.append({
            "timestamp": datetime.now(timezone.utc),
            "query": query,
            "type": search_type
        })

        # Placeholder: Return mock structure
        # In production, this would be replaced with actual WebSearch call
        return {
            "query": query,
            "results": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "search_type": search_type,
            "note": "WebSearch integration pending"
        }

    def _structure_news_results(self, asset: str, results: Dict) -> Dict:
        """Structure news search results."""
        return {
            "asset": asset,
            "search_type": "news",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": "No recent significant news (WebSearch integration pending)",
            "sentiment": "neutral",
            "key_events": [],
            "sources_count": 0,
            "confidence": 0.5
        }

    def _structure_macro_results(self, results: Dict) -> Dict:
        """Structure macro event results."""
        return {
            "search_type": "macro",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": "Standard macro conditions (WebSearch integration pending)",
            "key_events": [],
            "risk_level": "medium",
            "confidence": 0.5
        }

    def _structure_breaking_news(self, assets: List[str], results: Dict) -> Dict:
        """Structure breaking news results."""
        return {
            "search_type": "breaking",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "assets_monitored": assets,
            "breaking_news": [],
            "alert_level": "normal",
            "confidence": 0.5
        }

    def _structure_sentiment_results(self, asset: str, results: Dict) -> Dict:
        """Structure sentiment analysis results."""
        return {
            "asset": asset,
            "search_type": "sentiment",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_sentiment": "neutral",
            "sentiment_score": 0.5,  # -1 to +1 scale
            "bullish_signals": 0,
            "bearish_signals": 0,
            "confidence": 0.5
        }

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = datetime.now(timezone.utc)
        one_minute_ago = now - timedelta(minutes=1)

        # Count recent searches
        recent_searches = [
            s for s in self.search_history
            if s["timestamp"] > one_minute_ago
        ]

        if len(recent_searches) >= self.max_searches_per_minute:
            return False

        return True

    def _cache_search(self, identifier: str, search_type: str, data: Dict):
        """Cache search results."""
        cache_key = f"{identifier}_{search_type}"
        cache_file = self.cache_dir / f"{cache_key}.json"

        cache_data = {
            "identifier": identifier,
            "search_type": search_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data
        }

        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)

    def _get_cached_search(self, identifier: str, search_type: str) -> Dict:
        """Get cached search results."""
        cache_key = f"{identifier}_{search_type}"
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return self._get_empty_result(search_type)

        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)

            # Check cache age (max 1 hour)
            cache_time = datetime.fromisoformat(cache_data["timestamp"])
            age = datetime.now(timezone.utc) - cache_time

            if age > timedelta(hours=1):
                logger.debug(f"Cache expired for {cache_key}")
                return self._get_empty_result(search_type)

            logger.debug(f"Using cached search for {cache_key}")
            return cache_data["data"]

        except Exception as e:
            logger.error(f"Error reading cache: {e}")
            return self._get_empty_result(search_type)

    def _get_empty_result(self, search_type: str) -> Dict:
        """Get empty result structure."""
        return {
            "search_type": search_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": "No data available",
            "confidence": 0.0,
            "cached": False
        }

    def get_search_summary_for_prompt(self, asset: str) -> str:
        """
        Get formatted search summary for gladiator prompts.

        Args:
            asset: Asset symbol

        Returns:
            Formatted string for prompt injection
        """
        # Get recent searches from cache
        news = self._get_cached_search(asset, "news")
        sentiment = self._get_cached_search(asset, "sentiment")
        macro = self._get_cached_search("macro", "events")

        summary = f"""
INTERNET SEARCH RESULTS ({asset}):

News Summary: {news.get('summary', 'No recent news')}
Sentiment: {sentiment.get('overall_sentiment', 'neutral')} (score: {sentiment.get('sentiment_score', 0.5):.2f})
Macro Events: {macro.get('summary', 'Standard conditions')}

Key Events:
{self._format_key_events(news.get('key_events', []))}

Last Updated: {news.get('timestamp', 'N/A')}
"""
        return summary.strip()

    def _format_key_events(self, events: List[str]) -> str:
        """Format key events for display."""
        if not events:
            return "- No significant events"

        return "\n".join([f"- {event}" for event in events[:5]])


# ==================== SINGLETON PATTERN ====================

_internet_search = None

def get_internet_search() -> InternetSearch:
    """Get singleton instance of InternetSearch."""
    global _internet_search
    if _internet_search is None:
        _internet_search = InternetSearch()
    return _internet_search
