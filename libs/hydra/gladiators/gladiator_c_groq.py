"""
HYDRA 3.0 - Gladiator C (Groq)

Role: Fast Backtester

Specialization:
- Rapid mental backtest simulation
- Pattern recognition across historical data
- Identifies similar market conditions
- Fast failure mode detection
- "Have we seen this before?" analysis

Groq is chosen for ultra-fast inference (~500 tokens/sec).
Perfect for quick historical pattern matching.
"""

from typing import Dict, List, Optional
from loguru import logger
import requests
import os

from .base_gladiator import BaseGladiator


class GladiatorC_Groq(BaseGladiator):
    """
    Gladiator C: Fast Backtester using Groq (Llama 3).

    Cost: ~$0.0001 per backtest (extremely cheap + fast)
    Speed: ~500 tokens/sec (vs ~20-50 for others)
    """

    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    MODEL = "llama-3.3-70b-versatile"  # Latest Llama 3

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            name="C",
            role="Fast Backtester",
            api_key=api_key or os.getenv("GROQ_API_KEY")
        )

        if not self.api_key:
            logger.warning("Groq API key not provided - gladiator will be in mock mode")

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
        Backtest existing strategies mentally.

        Gladiator C doesn't create strategies - it BACKTESTS them.
        """
        logger.info(f"Gladiator C backtesting strategies for {asset}")

        if not existing_strategies:
            return self._no_strategy_to_backtest(asset)

        # Backtest the latest strategy
        strategy_to_test = existing_strategies[-1]

        system_prompt = self._build_backtest_system_prompt()
        user_prompt = self._build_backtest_prompt(
            asset=asset,
            asset_type=asset_type,
            regime=regime,
            strategy=strategy_to_test,
            market_data=market_data
        )

        response = self._call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,  # Low temperature for consistent backtesting
            max_tokens=1500
        )

        backtest_result = self._parse_json_response(response)

        if backtest_result:
            backtest_result["strategy_id"] = f"GLADIATOR_C_{self.strategy_count:04d}"
            backtest_result["gladiator"] = "C"
            backtest_result["backtested_strategy"] = strategy_to_test.get("strategy_id", "unknown")
            self.strategy_count += 1

            logger.success(
                f"Gladiator C backtested: {backtest_result.get('strategy_name', 'Unknown')} "
                f"(passed: {backtest_result.get('backtest_passed', False)}, "
                f"win rate: {backtest_result.get('estimated_wr', 0):.1%})"
            )
            return backtest_result
        else:
            return self._no_strategy_to_backtest(asset)

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
        Vote based on historical pattern matching.
        """
        logger.info(f"Gladiator C voting on {asset} {signal.get('direction', 'UNKNOWN')}")

        system_prompt = "You are Gladiator C, a pattern recognition specialist. Have we seen this setup before?"

        user_prompt = self._build_vote_prompt(
            asset=asset,
            regime=regime,
            strategy=strategy,
            signal=signal,
            market_data=market_data
        )

        response = self._call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            max_tokens=400
        )

        vote = self._parse_json_response(response)

        if vote:
            self._log_vote(vote)
            logger.info(f"Gladiator C votes: {vote.get('vote', 'UNKNOWN')} ({vote.get('confidence', 0):.1%})")
            return vote
        else:
            return {
                "vote": "HOLD",
                "confidence": 0.3,
                "reasoning": "Failed to parse vote response",
                "concerns": ["LLM response parsing error"]
            }

    # ==================== PROMPT ENGINEERING ====================

    def _build_backtest_system_prompt(self) -> str:
        """Build system prompt for backtesting."""
        return """You are Gladiator C, a quantitative backtesting specialist.

Your job: MENTALLY SIMULATE this strategy on historical data.

Think through:
1. Similar Setups: When has this setup appeared before?
2. Outcomes: What happened in those cases?
3. Failure Cases: When did similar strategies fail badly?
4. Market Regimes: Does this work in all regimes or just some?
5. Sample Size: How many trades would this generate per month?
6. Drawdowns: What's the max consecutive losses?

You don't have access to actual historical data.
But you KNOW general market patterns from training data.

Use your knowledge to estimate:
- Realistic win rate (not curve-fit optimism)
- Realistic R:R (accounting for slippage)
- Worst-case scenarios

Output JSON:
{
  "backtest_passed": true/false,
  "strategy_name": "Same as input",
  "estimated_wr": 0.58,
  "estimated_rr": 1.4,
  "estimated_trades_per_month": 12,
  "max_consecutive_losses": 5,
  "similar_historical_setups": ["Description of past similar scenarios"],
  "failure_scenarios": ["When this would fail badly"],
  "adjustments_recommended": ["How to improve based on history"],
  "confidence": 0.7
}

Be realistic. Backtests that are too good = overfitted."""

    def _build_backtest_prompt(
        self,
        asset: str,
        asset_type: str,
        regime: str,
        strategy: Dict,
        market_data: Dict
    ) -> str:
        """Build prompt for mental backtesting."""
        return f"""Mentally backtest this strategy:

ASSET: {asset} ({asset_type})
REGIME: {regime}

STRATEGY:
{strategy.get('strategy_name', 'Unknown')}

STRUCTURAL EDGE:
{strategy.get('structural_edge', 'Unknown')}

ENTRY: {strategy.get('entry_rules', 'Unknown')}
EXIT: {strategy.get('exit_rules', 'Unknown')}

CLAIMED PERFORMANCE:
- Win rate: {strategy.get('expected_wr', 0):.1%}
- R:R: {strategy.get('expected_rr', 0):.2f}

Your task:
1. Think of similar strategies you've seen in training data
2. What were their ACTUAL results?
3. Is this claimed performance realistic?
4. What adjustments would improve it?

Be BRUTALLY HONEST. Real backtest data > optimistic claims."""

    def _build_vote_prompt(
        self,
        asset: str,
        regime: str,
        strategy: Dict,
        signal: Dict,
        market_data: Dict
    ) -> str:
        """Build prompt for pattern matching vote."""
        return f"""Have we seen this setup before?

ASSET: {asset}
REGIME: {regime}
DIRECTION: {signal.get('direction', 'UNKNOWN')}

STRATEGY: {strategy.get('strategy_name', 'Unknown')}
CURRENT SETUP: Entry at {signal.get('entry_price', 'N/A')}

Think through your training data:
1. Similar setups in the past?
2. What happened next?
3. Success rate of this pattern?

Vote: BUY, SELL, or HOLD

Output JSON:
{{
  "vote": "BUY|SELL|HOLD",
  "confidence": 0.7,
  "reasoning": "Historical pattern analysis",
  "concerns": ["Deviations from past successful setups"]
}}"""

    def _no_strategy_to_backtest(self, asset: str) -> Dict:
        """Fallback when no strategy to backtest."""
        return {
            "strategy_id": f"GLADIATOR_C_{self.strategy_count:04d}_NO_BACKTEST",
            "backtest_passed": False,
            "strategy_name": f"No Strategy - {asset}",
            "estimated_wr": 0.0,
            "estimated_rr": 0.0,
            "estimated_trades_per_month": 0,
            "max_consecutive_losses": 0,
            "similar_historical_setups": [],
            "failure_scenarios": ["No strategy provided"],
            "adjustments_recommended": ["Provide strategy to backtest"],
            "confidence": 0.0,
            "gladiator": "C"
        }

    # ==================== LLM API INTEGRATION ====================

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1500
    ) -> str:
        """
        Call Groq API.
        """
        if not self.api_key:
            logger.warning("Groq API key not set - using mock response")
            return self._mock_response()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(
                self.GROQ_API_URL,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            return data["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            logger.error(f"Groq API error: {e}")
            return self._mock_response()

    def _mock_response(self) -> str:
        """Mock response for testing."""
        return """{
  "backtest_passed": true,
  "strategy_name": "London Open Volatility",
  "estimated_wr": 0.56,
  "estimated_rr": 1.3,
  "estimated_trades_per_month": 15,
  "max_consecutive_losses": 6,
  "similar_historical_setups": [
    "London fix volatility strategies (common in FX)",
    "Session open momentum plays",
    "Liquidity spike strategies"
  ],
  "failure_scenarios": [
    "Central bank interventions during London open",
    "Major news releases at session start",
    "Extremely wide spreads (> 3x normal)"
  ],
  "adjustments_recommended": [
    "Add news calendar filter",
    "Require tighter spread (< 2x normal)",
    "Reduce size on Fridays (weekend risk)"
  ],
  "confidence": 0.68
}"""
