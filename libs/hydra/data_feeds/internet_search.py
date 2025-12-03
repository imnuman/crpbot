"""
HYDRA 3.0 - Internet Search API (Serper)

Provides real-time market intelligence via Google Search:
- Breaking news and events
- Market sentiment from headlines
- Regulatory news
- Whale activity reports
- Exchange news

Uses Serper API (~$0.001/search) with caching to minimize costs.
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import httpx

logger = logging.getLogger(__name__)


# Serper API configuration
SERPER_API_URL = "https://google.serper.dev/search"
SERPER_NEWS_URL = "https://google.serper.dev/news"
DEFAULT_NUM_RESULTS = 10
CACHE_TTL_MINUTES = 15  # Cache search results for 15 minutes
MAX_SEARCHES_PER_HOUR = 20  # Cost control


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    snippet: str
    link: str
    source: str = ""
    date: str = ""
    position: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_serper(cls, item: dict, position: int = 0) -> "SearchResult":
        """Create from Serper API response item."""
        return cls(
            title=item.get("title", ""),
            snippet=item.get("snippet", ""),
            link=item.get("link", ""),
            source=item.get("source", ""),
            date=item.get("date", ""),
            position=position,
        )


@dataclass
class SearchResponse:
    """Complete search response with metadata."""
    query: str
    results: list[SearchResult]
    search_type: str  # "web" or "news"
    timestamp: datetime
    cached: bool = False
    credits_used: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "search_type": self.search_type,
            "timestamp": self.timestamp.isoformat(),
            "cached": self.cached,
            "credits_used": self.credits_used,
            "error": self.error,
            "num_results": len(self.results),
        }

    def format_for_llm(self, max_results: int = 5) -> str:
        """Format search results for LLM consumption."""
        if self.error:
            return f"Search failed: {self.error}"

        if not self.results:
            return f"No results found for: {self.query}"

        lines = [f"Search: {self.query}", f"Results ({len(self.results)} found):"]

        for i, result in enumerate(self.results[:max_results], 1):
            date_str = f" [{result.date}]" if result.date else ""
            lines.append(f"\n{i}. {result.title}{date_str}")
            if result.snippet:
                lines.append(f"   {result.snippet[:200]}...")
            if result.source:
                lines.append(f"   Source: {result.source}")

        return "\n".join(lines)


# Preset search queries for crypto market intelligence
PRESET_QUERIES = {
    "btc_news": "Bitcoin BTC breaking news today",
    "eth_news": "Ethereum ETH news today",
    "crypto_regulation": "cryptocurrency regulation news",
    "crypto_whale": "crypto whale large transaction today",
    "btc_etf": "Bitcoin ETF news",
    "fed_rates": "Federal Reserve interest rate decision",
    "crypto_hack": "cryptocurrency exchange hack security",
    "defi_news": "DeFi decentralized finance news",
    "nft_market": "NFT market news today",
    "stablecoin_news": "USDT USDC stablecoin news",
    "binance_news": "Binance exchange news",
    "coinbase_news": "Coinbase exchange news",
    "sec_crypto": "SEC cryptocurrency enforcement",
    "crypto_sentiment": "cryptocurrency market sentiment analysis",
    "bitcoin_halving": "Bitcoin halving countdown",
}


class InternetSearch:
    """
    Internet search client using Serper API.

    Features:
    - Web and news search
    - Result caching (15 min TTL)
    - Rate limiting
    - Cost tracking
    - Search history for dashboard
    """

    _instance: Optional["InternetSearch"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, api_key: Optional[str] = None, data_dir: Optional[Path] = None):
        if self._initialized:
            return

        # Get API key from env if not provided
        self.api_key = api_key or os.getenv("SERPER_API_KEY", "")

        # Auto-detect data directory
        if data_dir is None:
            if os.path.exists("/root/crpbot"):
                data_dir = Path("/root/crpbot/data/hydra")
            else:
                data_dir = Path.home() / "crpbot" / "data" / "hydra"

        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Cache and history files
        self.cache_file = self.data_dir / "search_cache.json"
        self.history_file = self.data_dir / "search_history.jsonl"

        # In-memory cache
        self.cache: dict[str, dict] = {}
        self._load_cache()

        # Rate limiting
        self.search_timestamps: list[datetime] = []

        # Cost tracking
        self.total_credits_used = 0.0
        self.searches_today = 0

        self._initialized = True
        logger.info(f"InternetSearch initialized (API key: {'set' if self.api_key else 'NOT SET'})")

    def _load_cache(self):
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    data = json.load(f)
                    # Only load non-expired entries
                    now = datetime.now()
                    for key, entry in data.items():
                        expires = datetime.fromisoformat(entry.get("expires", "2000-01-01"))
                        if expires > now:
                            self.cache[key] = entry
                logger.info(f"Loaded {len(self.cache)} cached searches")
            except Exception as e:
                logger.error(f"Failed to load search cache: {e}")

    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save search cache: {e}")

    def _cache_key(self, query: str, search_type: str, num_results: int) -> str:
        """Generate cache key for a search."""
        key_str = f"{query}:{search_type}:{num_results}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)

        # Remove old timestamps
        self.search_timestamps = [ts for ts in self.search_timestamps if ts > hour_ago]

        return len(self.search_timestamps) < MAX_SEARCHES_PER_HOUR

    def _log_search(self, response: SearchResponse):
        """Log search to history file."""
        try:
            with open(self.history_file, 'a') as f:
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "query": response.query,
                    "type": response.search_type,
                    "num_results": len(response.results),
                    "cached": response.cached,
                    "credits": response.credits_used,
                }
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to log search: {e}")

    async def search(
        self,
        query: str,
        search_type: str = "news",
        num_results: int = DEFAULT_NUM_RESULTS,
        use_cache: bool = True
    ) -> SearchResponse:
        """
        Perform a search using Serper API.

        Args:
            query: Search query
            search_type: "web" or "news"
            num_results: Number of results to return
            use_cache: Whether to use cached results

        Returns:
            SearchResponse with results
        """
        # Check cache first
        cache_key = self._cache_key(query, search_type, num_results)
        if use_cache and cache_key in self.cache:
            cached = self.cache[cache_key]
            expires = datetime.fromisoformat(cached["expires"])
            if expires > datetime.now():
                logger.info(f"Cache hit for: {query}")
                results = [SearchResult(**r) for r in cached["results"]]
                response = SearchResponse(
                    query=query,
                    results=results,
                    search_type=search_type,
                    timestamp=datetime.fromisoformat(cached["timestamp"]),
                    cached=True,
                    credits_used=0.0,
                )
                self._log_search(response)
                return response

        # Check API key
        if not self.api_key:
            return SearchResponse(
                query=query,
                results=[],
                search_type=search_type,
                timestamp=datetime.now(),
                error="SERPER_API_KEY not set",
            )

        # Check rate limit
        if not self._check_rate_limit():
            return SearchResponse(
                query=query,
                results=[],
                search_type=search_type,
                timestamp=datetime.now(),
                error=f"Rate limit exceeded ({MAX_SEARCHES_PER_HOUR}/hour)",
            )

        # Perform search
        url = SERPER_NEWS_URL if search_type == "news" else SERPER_API_URL
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "q": query,
            "num": num_results,
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()

            # Parse results
            results = []
            items = data.get("news", data.get("organic", []))
            for i, item in enumerate(items):
                results.append(SearchResult.from_serper(item, i + 1))

            # Track rate limiting
            self.search_timestamps.append(datetime.now())
            self.total_credits_used += 0.001  # ~$0.001 per search
            self.searches_today += 1

            # Create response
            response = SearchResponse(
                query=query,
                results=results,
                search_type=search_type,
                timestamp=datetime.now(),
                cached=False,
                credits_used=0.001,
            )

            # Cache results
            self.cache[cache_key] = {
                "results": [r.to_dict() for r in results],
                "timestamp": datetime.now().isoformat(),
                "expires": (datetime.now() + timedelta(minutes=CACHE_TTL_MINUTES)).isoformat(),
            }
            self._save_cache()

            self._log_search(response)
            logger.info(f"Search completed: {query} ({len(results)} results)")

            return response

        except httpx.HTTPStatusError as e:
            logger.error(f"Serper API error: {e}")
            return SearchResponse(
                query=query,
                results=[],
                search_type=search_type,
                timestamp=datetime.now(),
                error=f"API error: {e.response.status_code}",
            )
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return SearchResponse(
                query=query,
                results=[],
                search_type=search_type,
                timestamp=datetime.now(),
                error=str(e),
            )

    async def search_preset(self, preset_name: str, use_cache: bool = True) -> SearchResponse:
        """
        Search using a preset query.

        Args:
            preset_name: Name of preset (e.g., "btc_news", "crypto_regulation")
            use_cache: Whether to use cache

        Returns:
            SearchResponse
        """
        if preset_name not in PRESET_QUERIES:
            return SearchResponse(
                query=preset_name,
                results=[],
                search_type="news",
                timestamp=datetime.now(),
                error=f"Unknown preset: {preset_name}. Available: {list(PRESET_QUERIES.keys())}",
            )

        query = PRESET_QUERIES[preset_name]
        return await self.search(query, search_type="news", use_cache=use_cache)

    async def get_market_intelligence(
        self,
        symbols: Optional[list[str]] = None
    ) -> dict[str, SearchResponse]:
        """
        Get comprehensive market intelligence.

        Args:
            symbols: Specific symbols to search (default: BTC, ETH)

        Returns:
            Dict of search responses by category
        """
        symbols = symbols or ["BTC", "ETH"]
        results = {}

        # Search for each symbol
        for symbol in symbols[:3]:  # Limit to 3 symbols
            query = f"{symbol} cryptocurrency news today"
            results[f"{symbol.lower()}_news"] = await self.search(query)

        # General market searches
        results["regulation"] = await self.search_preset("crypto_regulation")
        results["sentiment"] = await self.search_preset("crypto_sentiment")

        return results

    def format_intelligence_for_llm(self, intelligence: dict[str, SearchResponse]) -> str:
        """Format market intelligence for LLM consumption."""
        lines = ["=== Market Intelligence ===\n"]

        for category, response in intelligence.items():
            lines.append(f"\n### {category.upper().replace('_', ' ')}")
            lines.append(response.format_for_llm(max_results=3))

        return "\n".join(lines)

    def get_search_stats(self) -> dict:
        """Get search statistics."""
        # Load history
        history = []
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    for line in f:
                        if line.strip():
                            history.append(json.loads(line))
            except Exception:
                pass

        # Calculate stats
        today = datetime.now().date()
        today_searches = [
            h for h in history
            if datetime.fromisoformat(h["timestamp"]).date() == today
        ]

        return {
            "total_searches": len(history),
            "searches_today": len(today_searches),
            "cached_hits": sum(1 for h in history if h.get("cached", False)),
            "total_credits": round(self.total_credits_used, 4),
            "rate_limit_remaining": MAX_SEARCHES_PER_HOUR - len(self.search_timestamps),
            "cache_size": len(self.cache),
        }

    def get_recent_searches(self, limit: int = 10) -> list[dict]:
        """Get recent search history for dashboard."""
        history = []
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    for line in f:
                        if line.strip():
                            history.append(json.loads(line))
            except Exception:
                pass

        return history[-limit:][::-1]  # Most recent first


# Singleton accessor
_search_instance: Optional[InternetSearch] = None


def get_internet_search(api_key: Optional[str] = None) -> InternetSearch:
    """Get or create the internet search singleton."""
    global _search_instance
    if _search_instance is None:
        _search_instance = InternetSearch(api_key)
    return _search_instance


# Synchronous wrapper for convenience
def search_sync(query: str, search_type: str = "news") -> SearchResponse:
    """Synchronous search wrapper."""
    import asyncio
    search = get_internet_search()
    return asyncio.run(search.search(query, search_type))
