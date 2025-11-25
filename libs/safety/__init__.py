"""
Safety Guards Module

Production-grade risk mitigation for V7 Ultimate trading system.

Modules:
1. MarketRegimeDetector - Filters choppy/ranging markets
2. DrawdownCircuitBreaker - Emergency stops on excessive losses
3. CorrelationManager - Prevents correlated position stacking
4. RejectionLogger - Tracks rejected signals for learning

Purpose:
- Win Rate: +5-10 points (avoiding bad conditions)
- Max Drawdown: -30-50% reduction
- System Intelligence: Learn from rejections
"""

from .market_regime_detector import MarketRegimeDetector, RegimeResult
from .drawdown_circuit_breaker import DrawdownCircuitBreaker, DrawdownStatus
from .correlation_manager import CorrelationManager, CorrelationResult
from .rejection_logger import RejectionLogger

__all__ = [
    'MarketRegimeDetector',
    'RegimeResult',
    'DrawdownCircuitBreaker',
    'DrawdownStatus',
    'CorrelationManager',
    'CorrelationResult',
    'RejectionLogger',
]
