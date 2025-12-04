"""
HYDRA 3.0 - Validators

Strategy validation systems:
- Walk-Forward Validator: Rolling window out-of-sample testing
- Monte Carlo Validator: Fast bootstrap simulation (< 1 sec)
"""

from .walk_forward import (
    WalkForwardValidator,
    get_walk_forward_validator,
    WalkForwardResult,
    ValidationStatus,
    TradeRecord,
    WindowMetrics,
    OverfitMetrics
)

from .monte_carlo import (
    MonteCarloValidator,
    get_monte_carlo_validator,
    MonteCarloResult
)

__all__ = [
    # Walk-Forward
    "WalkForwardValidator",
    "get_walk_forward_validator",
    "WalkForwardResult",
    "ValidationStatus",
    "TradeRecord",
    "WindowMetrics",
    "OverfitMetrics",
    # Monte Carlo
    "MonteCarloValidator",
    "get_monte_carlo_validator",
    "MonteCarloResult",
]
