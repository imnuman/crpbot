"""
LLM Integration Modules for V7 Ultimate

This package contains DeepSeek LLM integration for signal synthesis.
"""

from .deepseek_client import DeepSeekClient, DeepSeekResponse
from .signal_synthesizer import (
    SignalSynthesizer,
    MarketContext,
    TheoryAnalysis,
    build_signal_prompt
)

__all__ = [
    'DeepSeekClient',
    'DeepSeekResponse',
    'SignalSynthesizer',
    'MarketContext',
    'TheoryAnalysis',
    'build_signal_prompt',
]
