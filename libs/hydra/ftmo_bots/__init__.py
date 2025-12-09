"""
FTMO Challenge Bots

Specialized trading bots designed for consistent daily profits:
- Gold London Reversal: Asian session reversal at London open
- EUR/USD Breakout: Daily S/R level breakouts
- US30 ORB: Opening range breakout fade
- NAS100 Gap Fill: Gap fill strategy at open
- Gold NY Mean Reversion: VWAP reversion during NY session

Expected Combined Daily Performance (~$868/day):
- Engine D (ATR): $66/day (when active)
- Gold London: $184/day
- EUR/USD Breakout: $172/day
- US30 ORB: $148/day
- NAS100 Gap: $150/day
- Gold NY: $148/day
"""

from .base_ftmo_bot import BaseFTMOBot, BotConfig, TradeSignal
from .gold_london_reversal import GoldLondonReversalBot, get_gold_london_bot
from .eurusd_breakout import EURUSDBreakoutBot, get_eurusd_bot
from .us30_orb import US30ORBBot, get_us30_bot
from .nas100_gap import NAS100GapBot, get_nas100_bot
from .gold_ny_reversion import GoldNYReversionBot, get_gold_ny_bot
from .orchestrator import FTMOOrchestrator, get_ftmo_orchestrator
from .metalearning import (
    FTMOMetalearner,
    get_ftmo_metalearner,
    AdaptivePositionSizer,
    VolatilityRegimeDetector,
    TradeResult,
)

__all__ = [
    "BaseFTMOBot",
    "BotConfig",
    "TradeSignal",
    "GoldLondonReversalBot",
    "get_gold_london_bot",
    "EURUSDBreakoutBot",
    "get_eurusd_bot",
    "US30ORBBot",
    "get_us30_bot",
    "NAS100GapBot",
    "get_nas100_bot",
    "GoldNYReversionBot",
    "get_gold_ny_bot",
    "FTMOOrchestrator",
    "get_ftmo_orchestrator",
    "FTMOMetalearner",
    "get_ftmo_metalearner",
    "AdaptivePositionSizer",
    "VolatilityRegimeDetector",
    "TradeResult",
]
