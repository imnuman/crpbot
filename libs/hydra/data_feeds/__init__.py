"""
HYDRA 3.0 - Data Feeds

Real-time data sources for market intelligence:
- Internet Search (Serper) - News and sentiment
- Order Book (Coinbase) - Depth, imbalance, whale detection
- Funding Rates (Binance) - Perpetual futures sentiment
- Liquidations (Binance/Coinglass) - Forced position closures
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

from .liquidations import (
    LiquidationsFeed,
    get_liquidations_feed,
    LiquidationEvent,
    LiquidationStats,
    LiquidationSnapshot,
)

from .historical_storage import (
    HistoricalStorage,
    get_historical_storage,
    DataType,
    HistoricalRecord,
    store_order_book,
    store_funding_rate,
    store_liquidation,
    store_price,
    store_engine_signal,
    get_price_history,
    get_funding_history,
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
    # Liquidations
    "LiquidationsFeed",
    "get_liquidations_feed",
    "LiquidationEvent",
    "LiquidationStats",
    "LiquidationSnapshot",
    # Historical Storage
    "HistoricalStorage",
    "get_historical_storage",
    "DataType",
    "HistoricalRecord",
    "store_order_book",
    "store_funding_rate",
    "store_liquidation",
    "store_price",
    "store_engine_signal",
    "get_price_history",
    "get_funding_history",
]
