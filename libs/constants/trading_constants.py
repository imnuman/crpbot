"""Trading configuration constants - centralized magic numbers.

This module contains all hardcoded constants used throughout the trading system.
Centralizing these values makes them easier to tune and maintain.
"""

# ============================================================================
# RISK & POSITION MANAGEMENT
# ============================================================================
RISK_PER_TRADE = 0.01  # 1% risk per trade
RISK_REWARD_RATIO = 2.0  # 2:1 risk:reward ratio
INITIAL_BALANCE = 10000.0  # Default account balance

# ============================================================================
# CONFIDENCE THRESHOLDS & TIERS
# ============================================================================
CONFIDENCE_THRESHOLD = 0.75  # Default confidence threshold for signals
CONFIDENCE_THRESHOLD_DEFAULT = 0.50  # Conservative default for evaluation
CONFIDENCE_THRESHOLD_MIN = 0.50  # Minimum allowed threshold

# Tier boundaries (for signal classification)
TIER_HIGH_CONFIDENCE = 0.75  # >= 75% confidence
TIER_MEDIUM_CONFIDENCE = 0.65  # >= 65% confidence
TIER_LOW_CONFIDENCE = 0.55  # >= 55% confidence

# Expected win rates by tier (for calibration metrics)
EXPECTED_WIN_RATE_HIGH = 0.75  # 75% expected for high-confidence signals
EXPECTED_WIN_RATE_MEDIUM = 0.65  # 65% expected for medium-confidence
EXPECTED_WIN_RATE_LOW = 0.55  # 55% expected for low-confidence
EXPECTED_WIN_RATE_DEFAULT = 0.60  # 60% fallback expected win rate

# ============================================================================
# LATENCY & EXECUTION
# ============================================================================
LATENCY_BUDGET_MS = 500.0  # Maximum allowed latency in milliseconds
LATENCY_PENALTY_MULTIPLIER = 0.9  # Penalty for exceeding latency budget
P90_PERCENTILE = 90  # 90th percentile for latency metrics

# ============================================================================
# FINANCIAL VALUES & CONVERSIONS
# ============================================================================
BASIS_POINTS_CONVERSION = 10000  # 1 bp = 1/10000
PERCENT_CONVERSION = 100  # Percentage conversion factor
TRADING_DAYS_PER_YEAR = 252  # For Sharpe ratio calculation

# Default execution metrics (spreads & slippage in basis points)
DEFAULT_SPREAD_MEAN_BPS = 12.0  # Average spread cost
DEFAULT_SPREAD_P50_BPS = 12.0  # Median spread
DEFAULT_SPREAD_P90_BPS = 18.0  # 90th percentile spread
DEFAULT_SLIPPAGE_MEAN_BPS = 3.0  # Average slippage
DEFAULT_SLIPPAGE_P50_BPS = 3.0  # Median slippage
DEFAULT_SLIPPAGE_P90_BPS = 6.0  # 90th percentile slippage

# ============================================================================
# TRADING SESSIONS (UTC HOURS)
# ============================================================================
TOKYO_SESSION_START = 0  # Tokyo opens at midnight UTC
TOKYO_SESSION_END = 8  # Tokyo closes at 8am UTC
LONDON_SESSION_START = 8  # London opens at 8am UTC
LONDON_SESSION_END = 16  # London closes at 4pm UTC
NEW_YORK_SESSION_START = 16  # New York opens at 4pm UTC
NEW_YORK_SESSION_END = 24  # New York closes at midnight UTC

# Day of week constants
WEEKEND_THRESHOLD = 5  # Saturday = 5, Sunday = 6

# Session names
TRADING_SESSIONS = ["tokyo", "london", "new_york"]

# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================
ATR_PERIOD = 14  # Average True Range period (standard)
VOLUME_MA_PERIOD = 20  # Volume moving average period
VOLATILITY_WINDOW = 20  # Rolling window for volatility calculation
VOLATILITY_LOW_PERCENTILE = 33.0  # Low volatility threshold (33rd percentile)
VOLATILITY_HIGH_PERCENTILE = 67.0  # High volatility threshold (67th percentile)

# Moving average periods for feature engineering
SMA_PERIODS = [7, 14, 21, 50]  # Short, medium, long-term trends

# ============================================================================
# NUMERICAL STABILITY
# ============================================================================
EPSILON_DIV_BY_ZERO = 1e-8  # Small value to prevent division by zero
ROBUST_NORM_Q25 = 0.25  # 25th percentile for robust normalization
ROBUST_NORM_Q75 = 0.75  # 75th percentile for robust normalization

# ============================================================================
# MODEL TRAINING HYPERPARAMETERS
# ============================================================================
# LSTM Model
LSTM_BATCH_SIZE = 32
LSTM_SEQUENCE_LENGTH = 60  # 60-minute lookback window
LSTM_PREDICTION_HORIZON = 15  # Predict 15 minutes ahead

# Transformer Model
TRANSFORMER_BATCH_SIZE = 16
TRANSFORMER_SEQUENCE_LENGTH = 100  # 100-minute lookback window
TRANSFORMER_PREDICTION_HORIZON = 15  # Predict 15 minutes ahead

# Reinforcement Learning
RL_DEFAULT_STEPS = 1000  # Default training steps
MAX_SYNTHETIC_DATA_RATIO = 0.2  # Maximum 20% synthetic data

# ============================================================================
# ENSEMBLE MODEL WEIGHTS
# ============================================================================
ENSEMBLE_WEIGHT_LSTM = 0.35  # LSTM contributes 35%
ENSEMBLE_WEIGHT_TRANSFORMER = 0.40  # Transformer contributes 40%
ENSEMBLE_WEIGHT_RL = 0.25  # RL contributes 25%

# Fallback weights when RL model is unavailable
ENSEMBLE_FALLBACK_LSTM = 0.50  # LSTM gets 50%
ENSEMBLE_FALLBACK_TRANSFORMER = 0.50  # Transformer gets 50%
ENSEMBLE_FALLBACK_RL = 0.0  # RL gets 0%

# ============================================================================
# MODEL PROMOTION GATES
# ============================================================================
MIN_ACCURACY_GATE = 0.68  # Minimum 68% win rate required for promotion
MAX_CALIBRATION_ERROR_GATE = 0.05  # Maximum 5% calibration error allowed

# ============================================================================
# SAFETY RAILS
# ============================================================================
MAX_SIGNALS_PER_HOUR = 10  # Maximum total signals per hour
MAX_HIGH_TIER_SIGNALS_PER_HOUR = 5  # Maximum high-confidence signals per hour

# ============================================================================
# CONFIGURATION DEFAULTS
# ============================================================================
DEFAULT_DATA_PROVIDER = "coinbase"  # Default data source
DEFAULT_DATABASE_URL = "sqlite:///tradingai.db"  # Default SQLite database
DEFAULT_LOG_FORMAT = "json"  # Default logging format
DEFAULT_MODEL_VERSION = "v1.0.0"  # Default model version

# Default symbols for execution metrics and testing
DEFAULT_SYMBOLS = ["BTC-USD", "ETH-USD", "BNB-USD"]
