"""
HYDRA 3.0 - Data Feeds

Real-time data sources for market intelligence:
- Internet Search (Serper) - News and sentiment
- Order Book (upcoming)
- Funding Rates (upcoming)
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

__all__ = [
    "InternetSearch",
    "get_internet_search",
    "search_sync",
    "SearchResult",
    "SearchResponse",
    "PRESET_QUERIES",
]
