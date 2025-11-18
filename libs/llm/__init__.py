"""
LLM Integration Modules for V7 Ultimate

This package contains DeepSeek LLM integration for signal synthesis.

Main Entry Point:
- SignalGenerator: Complete signal generation orchestrator

Components:
- DeepSeekClient: API client for DeepSeek LLM
- SignalSynthesizer: Formats 6 theories into LLM prompts
- SignalParser: Parses LLM responses into structured signals
"""

from .deepseek_client import DeepSeekClient, DeepSeekResponse
from .signal_synthesizer import (
    SignalSynthesizer,
    MarketContext,
    TheoryAnalysis,
    build_signal_prompt
)
from .signal_parser import (
    SignalParser,
    ParsedSignal,
    SignalType,
    parse_llm_response
)
from .signal_generator import SignalGenerator, SignalGenerationResult

__all__ = [
    # Main entry point
    'SignalGenerator',
    'SignalGenerationResult',

    # DeepSeek client
    'DeepSeekClient',
    'DeepSeekResponse',

    # Signal synthesizer
    'SignalSynthesizer',
    'MarketContext',
    'TheoryAnalysis',
    'build_signal_prompt',

    # Signal parser
    'SignalParser',
    'ParsedSignal',
    'SignalType',
    'parse_llm_response',
]
