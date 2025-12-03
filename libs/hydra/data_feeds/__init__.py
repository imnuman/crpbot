"""
HYDRA 3.0 - Data Feeds

Real-time data sources for market intelligence:
- Internet Search (Serper) - News and sentiment
- Order Book (Coinbase) - Depth, imbalance, whale detection
- Funding Rates (Binance) - Perpetual futures sentiment
- Liquidations (upcoming)
"""

from .internet_search import (
    InternetSearch,
    get_internet_search,
    search_sync,
    SearchResult,
    SearchResponse,
    PRESET_QUERIES,
)

from .order_book import (
    OrderBookFeed,
    get_order_book_feed,
    OrderBookMetrics,
    OrderLevel,
)

from .funding_rates import (
    FundingRatesFeed,
    get_funding_rates_feed,
    FundingRate,
    FundingSnapshot,
)

__all__ = [
    # Internet Search
    "InternetSearch",
    "get_internet_search",
    "search_sync",
    "SearchResult",
    "SearchResponse",
    "PRESET_QUERIES",
    # Order Book
    "OrderBookFeed",
    "get_order_book_feed",
    "OrderBookMetrics",
    "OrderLevel",
    # Funding Rates
    "FundingRatesFeed",
    "get_funding_rates_feed",
    "FundingRate",
    "FundingSnapshot",
]
