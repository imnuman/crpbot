"""
Knowledge collectors from various sources.
"""

from typing import List, Type, Dict

from ..base import BaseCollector

# Import collectors (graceful if deps missing)
_COLLECTORS: Dict[str, Type[BaseCollector]] = {}

try:
    from .mql5 import MQL5Collector
    _COLLECTORS["mql5"] = MQL5Collector
except ImportError:
    pass

try:
    from .github import GitHubCollector
    _COLLECTORS["github"] = GitHubCollector
except ImportError:
    pass

try:
    from .economic_calendar import EconomicCalendarCollector
    _COLLECTORS["calendar"] = EconomicCalendarCollector
except ImportError:
    pass

try:
    from .tradingview import TradingViewCollector
    _COLLECTORS["tradingview"] = TradingViewCollector
except ImportError:
    pass

try:
    from .reddit import RedditCollector
    _COLLECTORS["reddit"] = RedditCollector
except ImportError:
    pass  # Reddit requires API approval

# NEW: COT (Commitment of Traders) - Institutional positioning
try:
    from .cot import COTCollector
    _COLLECTORS["cot"] = COTCollector
except ImportError:
    pass

# NEW: Fear/Greed Index - Market sentiment
try:
    from .fear_greed import FearGreedCollector
    _COLLECTORS["fear_greed"] = FearGreedCollector
except ImportError:
    pass

# NEW: Broker Sentiment - Retail positioning (contrarian)
try:
    from .broker_sentiment import BrokerSentimentCollector
    _COLLECTORS["broker_sentiment"] = BrokerSentimentCollector
except ImportError:
    pass

# NEW: Central Bank Calendar - FOMC, ECB, BOE meetings
try:
    from .central_banks import CentralBankCollector
    _COLLECTORS["central_banks"] = CentralBankCollector
except ImportError:
    pass

# NEW: Options Flow - Unusual options activity
try:
    from .options_flow import OptionsFlowCollector
    _COLLECTORS["options_flow"] = OptionsFlowCollector
except ImportError:
    pass

# NEW: News Sentiment - Market news with sentiment analysis
try:
    from .news_sentiment import NewsSentimentCollector
    _COLLECTORS["news_sentiment"] = NewsSentimentCollector
except ImportError:
    pass


def get_all_collectors() -> Dict[str, Type[BaseCollector]]:
    """Get all available collector classes."""
    return _COLLECTORS.copy()


def get_collector(name: str) -> Type[BaseCollector]:
    """Get a specific collector by name."""
    if name not in _COLLECTORS:
        raise ValueError(f"Unknown collector: {name}. Available: {list(_COLLECTORS.keys())}")
    return _COLLECTORS[name]


def register_collector(name: str, collector_cls: Type[BaseCollector]):
    """Register a collector class."""
    _COLLECTORS[name] = collector_cls
