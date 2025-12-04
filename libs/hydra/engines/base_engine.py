"""
HYDRA 3.0 - Base Gladiator (Abstract Class)

All gladiators inherit from this base class.

Each gladiator must implement:
- generate_strategy(): Create a trading strategy
- vote_on_trade(): Vote BUY/SELL/HOLD on a specific trade
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from loguru import logger

from ..engine_specialization import get_specialty_validator, SpecialtyConfig


class BaseGladiator(ABC):
    """
    Abstract base class for all gladiators.

    Each gladiator is an LLM with a specialized role and prompt.
    """

    def __init__(self, name: str, role: str, api_key: Optional[str] = None):
        """
        Initialize gladiator.

        Args:
            name: Gladiator identifier (e.g., "A", "B", "C", "D")
            role: Specialized role (e.g., "Structural Edge Generator")
            api_key: LLM API key (optional - can be None for mock/testing)
        """
        self.name = name
        self.role = role
        self.api_key = api_key
        self.strategy_count = 0
        self.vote_history = []
        self.specialty_validator = get_specialty_validator()
        self.specialty_config = self.specialty_validator.get_specialty(name)

        logger.info(f"Gladiator {name} initialized: {role}")

    @abstractmethod
    def generate_strategy(
        self,
        asset: str,
        asset_type: str,
        asset_profile: Dict,
        regime: str,
        regime_confidence: float,
        market_data: Dict,
        existing_strategies: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Generate a new trading strategy for this asset/regime.

        Args:
            asset: Symbol (e.g., "USD/TRY", "BONK")
            asset_type: "exotic_forex" or "meme_perp"
            asset_profile: Full AssetProfile dict
            regime: Current regime (TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE, CHOPPY)
            regime_confidence: Confidence in regime (0.0-1.0)
            market_data: Recent OHLCV + indicators
            existing_strategies: Previously generated strategies (to avoid duplication)

        Returns:
            {
                "strategy_id": "GLADIATOR_A_001",
                "strategy_name": "London Open Volatility - USD/TRY",
                "structural_edge": "Session open volatility spike",
                "entry_rules": "Enter LONG at London open (8AM UTC) if spread < 25 pips",
                "exit_rules": "TP at 1.5R, SL at 2x ATR",
                "filters": ["spread_normal", "no_cb_meeting_24hrs"],
                "risk_per_trade": 0.008,  # 0.8%
                "expected_wr": 0.63,  # 63%
                "expected_rr": 1.5,  # 1.5 R:R
                "why_it_works": "London open creates predictable volatility...",
                "weaknesses": ["Fails during CB surprise announcements"],
                "confidence": 0.75
            }
        """
        pass

    @abstractmethod
    def vote_on_trade(
        self,
        asset: str,
        asset_type: str,
        regime: str,
        strategy: Dict,
        signal: Dict,
        market_data: Dict
    ) -> Dict:
        """
        Vote on a specific trade signal.

        Args:
            asset: Symbol
            asset_type: Type
            regime: Current regime
            strategy: Strategy that generated this signal
            signal: Trade signal details (direction, entry, SL, TP)
            market_data: Current market state

        Returns:
            {
                "vote": "BUY" | "SELL" | "HOLD",
                "confidence": 0.75,  # 0.0-1.0
                "reasoning": "Clear breakout, momentum aligned, low spread",
                "concerns": ["Funding rate elevated at 0.3%"]
            }
        """
        pass

    def check_specialty_trigger(self, market_data: Dict) -> Tuple[bool, str]:
        """
        Check if this engine's specialty trigger is active.

        Each engine has ONE specialty:
        - Engine A: Liquidation cascades ($20M+ trigger)
        - Engine B: Funding rate extremes (>0.5%)
        - Engine C: Orderbook imbalance (>2.5:1)
        - Engine D: Regime transitions (ATR 2× expansion)

        Args:
            market_data: Dict with market conditions

        Returns:
            Tuple of (is_triggered: bool, reason: str)
        """
        if not self.specialty_config:
            return True, "No specialty configured - always active"

        active_triggers = self.specialty_validator.get_active_triggers(market_data)
        is_triggered = active_triggers.get(self.name, False)

        if is_triggered:
            return True, f"Specialty triggered: {self.specialty_config.specialty.value}"
        else:
            return False, f"Specialty NOT triggered: {self.specialty_config.specialty.value}"

    def vote_on_trade_with_specialty_check(
        self,
        asset: str,
        asset_type: str,
        regime: str,
        strategy: Dict,
        signal: Dict,
        market_data: Dict
    ) -> Dict:
        """
        Vote on a trade with specialty enforcement.

        If engine's specialty is not triggered, returns HOLD vote.
        Otherwise, calls the underlying vote_on_trade() method.

        Args:
            Same as vote_on_trade()

        Returns:
            Vote dict with "vote", "confidence", "reasoning", "concerns"
        """
        # Check specialty trigger
        is_triggered, reason = self.check_specialty_trigger(market_data)

        if not is_triggered:
            logger.info(f"Gladiator {self.name}: {reason} - voting HOLD")
            return {
                "vote": "HOLD",
                "confidence": 0.0,
                "reasoning": f"Specialty not triggered: {reason}",
                "concerns": ["Engine specialty condition not met"],
                "specialty_blocked": True
            }

        # Specialty triggered - proceed with normal vote
        logger.info(f"Gladiator {self.name}: {reason} - proceeding with vote")
        return self.vote_on_trade(asset, asset_type, regime, strategy, signal, market_data)

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """
        Call LLM API (to be implemented by subclasses).

        Args:
            system_prompt: System instructions
            user_prompt: User query
            temperature: Sampling temperature
            max_tokens: Max response tokens

        Returns:
            LLM response as string
        """
        raise NotImplementedError("Subclass must implement _call_llm()")

    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """
        Parse JSON from LLM response.

        LLMs often wrap JSON in markdown code blocks, so we handle that.
        """
        import json
        import re

        # Try to extract JSON from markdown code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                logger.error(f"Could not find JSON in LLM response: {response[:200]}")
                return None

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}\nResponse: {json_str[:200]}")
            return None

    def get_stats(self) -> Dict:
        """Get gladiator performance statistics."""
        return {
            "name": self.name,
            "role": self.role,
            "strategies_generated": self.strategy_count,
            "votes_cast": len(self.vote_history)
        }

    def _log_vote(self, vote: Dict):
        """Log vote to history."""
        self.vote_history.append({
            "timestamp": datetime.now(timezone.utc),
            **vote
        })

        # Keep only last 1000 votes
        if len(self.vote_history) > 1000:
            self.vote_history = self.vote_history[-1000:]
