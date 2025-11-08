"""Trading constants package."""
from libs.constants.trading_constants import *

__all__ = [
    # Risk & Position Management
    "RISK_PER_TRADE",
    "RISK_REWARD_RATIO",
    "INITIAL_BALANCE",

    # Confidence & Tiers
    "CONFIDENCE_THRESHOLD",
    "TIER_HIGH_CONFIDENCE",
    "TIER_MEDIUM_CONFIDENCE",
    "TIER_LOW_CONFIDENCE",
    "EXPECTED_WIN_RATE_HIGH",
    "EXPECTED_WIN_RATE_MEDIUM",
    "EXPECTED_WIN_RATE_LOW",

    # Execution
    "LATENCY_BUDGET_MS",
    "DEFAULT_SPREAD_MEAN_BPS",
    "DEFAULT_SPREAD_P90_BPS",
    "DEFAULT_SLIPPAGE_MEAN_BPS",
    "DEFAULT_SLIPPAGE_P90_BPS",

    # Sessions
    "TOKYO_SESSION_START",
    "TOKYO_SESSION_END",
    "LONDON_SESSION_START",
    "LONDON_SESSION_END",
    "NEW_YORK_SESSION_START",
    "NEW_YORK_SESSION_END",

    # Technical Indicators
    "ATR_PERIOD",
    "VOLUME_MA_PERIOD",
    "SMA_PERIODS",

    # Numerical
    "EPSILON_DIV_BY_ZERO",
    "BASIS_POINTS_CONVERSION",

    # Model
    "MIN_ACCURACY_GATE",
    "MAX_CALIBRATION_ERROR_GATE",
]
