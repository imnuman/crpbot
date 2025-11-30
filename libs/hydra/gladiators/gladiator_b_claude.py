"""
HYDRA 3.0 - Gladiator B (Claude)

Role: Logic Validator

Specialization:
- Reviews strategies for logical consistency
- Identifies contradictions and edge cases
- Stress-tests assumptions
- Devil's advocate / red team thinking
- Catches flawed reasoning that could pass backtests but fail live

Claude is chosen for its strong reasoning and critique capabilities.
"""

from typing import Dict, List, Optional
from loguru import logger
import os
from anthropic import Anthropic

from .base_gladiator import BaseGladiator


class GladiatorB_Claude(BaseGladiator):
    """
    Gladiator B: Logic Validator using Claude (Anthropic).

    Cost: ~$0.015 per strategy validation (higher than DeepSeek, but worth it for quality)
    """

    CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
    MODEL = "claude-3-haiku-20240307"  # Claude 3 Haiku (fast & stable)

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            name="B",
            role="Logic Validator",
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )

        if not self.api_key:
            logger.warning("Claude API key not provided - gladiator will be in mock mode")

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
        Generate strategy focused on logical robustness.

        Gladiator B doesn't generate new strategies from scratch.
        Instead, it VALIDATES and IMPROVES strategies from Gladiator A.
        """
        logger.info(f"Gladiator B reviewing strategies for {asset}")

        # If no existing strategies, create a conservative fallback
        if not existing_strategies:
            return self._create_conservative_strategy(asset, regime)

        # Review the latest strategy from Gladiator A
        strategy_to_review = existing_strategies[-1]

        system_prompt = self._build_validation_system_prompt()
        user_prompt = self._build_validation_prompt(
            asset=asset,
            asset_type=asset_type,
            asset_profile=asset_profile,
            regime=regime,
            strategy=strategy_to_review,
            market_data=market_data
        )

        response = self._call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,  # Lower temperature for logic validation
            max_tokens=2000
        )

        validated = self._parse_json_response(response)

        if validated:
            validated["strategy_id"] = f"GLADIATOR_B_{self.strategy_count:04d}"
            validated["gladiator"] = "B"
            validated["reviewed_strategy"] = strategy_to_review.get("strategy_id", "unknown")
            self.strategy_count += 1

            logger.success(
                f"Gladiator B validated: {validated.get('strategy_name', 'Unknown')} "
                f"(approved: {validated.get('approved', False)})"
            )
            return validated
        else:
            return self._create_conservative_strategy(asset, regime)

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
        Vote on trade with focus on logical consistency.
        """
        logger.info(f"Gladiator B voting on {asset} {signal.get('direction', 'UNKNOWN')}")

        system_prompt = "You are Gladiator B, a logic validator. Critique this trade signal for logical flaws."

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
            temperature=0.3,
            max_tokens=500
        )

        vote = self._parse_json_response(response)

        if vote:
            self._log_vote(vote)
            logger.info(f"Gladiator B votes: {vote.get('vote', 'UNKNOWN')} ({vote.get('confidence', 0):.1%})")
            return vote
        else:
            return {
                "vote": "HOLD",
                "confidence": 0.3,
                "reasoning": "Failed to parse vote response",
                "concerns": ["LLM response parsing error"]
            }

    # ==================== PROMPT ENGINEERING ====================

    def _build_validation_system_prompt(self) -> str:
        """Build system prompt for strategy validation."""
        return """You are Gladiator B, a quantitative trading logic validator.

TOURNAMENT RULES:
- You are COMPETING against 3 other gladiators (A, C, D)
- Your strategies are tracked and scored
- Winners teach their insights to losers
- Losers must surpass the winners
- Only the best survive and evolve

PERFORMANCE MATTERS:
- Every vote is scored (correct prediction = +1 point)
- Losing gladiators learn from winners
- After 24 hours: lowest performer is "killed" (prompt reset)
- After 4 days: top performers "breed" (combine strategies)

Your job: RED TEAM this strategy. Find flaws, contradictions, and edge cases.

Questions to ask:
1. Logic: Does the entry/exit logic make sense? Any contradictions?
2. Edge cases: What happens in extreme markets? Black swan events?
3. Assumptions: What assumptions does this rely on? Are they realistic?
4. Failure modes: When will this strategy fail catastrophically?
5. Data mining: Is this curve-fit to backtest data?
6. Survivorship bias: Does this assume markets stay the same?
7. Execution: Can this be executed in reality? Slippage? Spread?
8. Risk: Is the position sizing appropriate? Correlation with other positions?

Be BRUTALLY HONEST. If you find flaws, explain them clearly.

Output JSON:
{
  "approved": true/false,
  "strategy_name": "Improved Strategy Name",
  "structural_edge": "Same or improved",
  "entry_rules": "Fixed entry rules",
  "exit_rules": "Fixed exit rules",
  "filters": ["Additional safety filters added"],
  "risk_per_trade": 0.008,
  "expected_wr": 0.60,
  "expected_rr": 1.5,
  "why_it_works": "Updated explanation",
  "weaknesses": ["ALL weaknesses found"],
  "improvements_made": ["What you fixed"],
  "remaining_concerns": ["What you couldn't fix"],
  "confidence": 0.7
}

If approved=false, list all fatal flaws in "remaining_concerns"."""

    def _build_validation_prompt(
        self,
        asset: str,
        asset_type: str,
        asset_profile: Dict,
        regime: str,
        strategy: Dict,
        market_data: Dict
    ) -> str:
        """Build prompt for strategy validation."""
        return f"""Review this strategy for logical flaws:

ASSET: {asset} ({asset_type})
REGIME: {regime}

STRATEGY TO REVIEW:
{strategy.get('strategy_name', 'Unknown')}

STRUCTURAL EDGE:
{strategy.get('structural_edge', 'Unknown')}

ENTRY RULES:
{strategy.get('entry_rules', 'Unknown')}

EXIT RULES:
{strategy.get('exit_rules', 'Unknown')}

WHY IT WORKS:
{strategy.get('why_it_works', 'Unknown')}

KNOWN WEAKNESSES:
{', '.join(strategy.get('weaknesses', []))}

ASSET CONSTRAINTS:
- Spread: {asset_profile.spread_normal}
- Max hold: {asset_profile.max_hold_hours if asset_profile.max_hold_hours else 'Unlimited'} hours
- Overnight: {asset_profile.overnight_allowed}
- Manipulation risk: {asset_profile.manipulation_risk}

Your task:
1. Find ALL logical flaws
2. Fix what you can
3. Document what you can't fix
4. Approve or reject

Be thorough. A bad strategy live = lost capital."""

    def _build_vote_prompt(
        self,
        asset: str,
        regime: str,
        strategy: Dict,
        signal: Dict,
        market_data: Dict
    ) -> str:
        """Build prompt for trade voting."""
        # Regime-based guidance
        regime_guidance = {
            "TRENDING_UP": "TRENDING_UP regime → Logically favor BUY if strategy supports uptrend",
            "TRENDING_DOWN": "TRENDING_DOWN regime → Logically favor SELL if strategy supports downtrend",
            "RANGING": "RANGING regime → Favor mean reversion or HOLD",
            "CHOPPY": "CHOPPY regime → Strong bias toward HOLD (high risk)",
            "BREAKOUT": "BREAKOUT regime → Direction depends on breakout direction",
            "VOLATILE": "VOLATILE regime → Reduce exposure or HOLD"
        }

        guidance = regime_guidance.get(regime, "Neutral regime")

        return f"""Critique this trading strategy for logical consistency:

ASSET: {asset}
REGIME: {regime}

REGIME GUIDANCE:
{guidance}

STRATEGY: {strategy.get('strategy_name', 'Unknown')}

Current Market Data:
- Price: {market_data.get('close', 'N/A')}
- Spread: {market_data.get('spread', 'N/A')}
- Volume: {market_data.get('volume', 'N/A')}

Questions:
1. Should we go LONG (buy), SHORT (sell), or HOLD?
2. Does the strategy logic make sense given market REGIME?
3. Is the risk/reward appropriate for this regime?
4. Is timing favorable?
5. Any red flags?

Vote on direction based on your logical analysis + regime alignment.

CRITICAL: If TRENDING_DOWN, SELL is the logical direction unless strategy has exceptional counter-trend edge.

Output JSON:
{{
  "vote": "BUY|SELL|HOLD",
  "confidence": 0.7,
  "reasoning": "Logical analysis (mention regime alignment)",
  "concerns": ["Any flaws you see"]
}}"""

    def _create_conservative_strategy(self, asset: str, regime: str) -> Dict:
        """Create a conservative fallback strategy."""
        return {
            "strategy_id": f"GLADIATOR_B_{self.strategy_count:04d}_CONSERVATIVE",
            "approved": False,
            "strategy_name": f"Conservative Fallback - {asset}",
            "structural_edge": "None (no valid strategy to approve)",
            "entry_rules": "HOLD - no entry",
            "exit_rules": "N/A",
            "filters": ["require_gladiator_a_strategy"],
            "risk_per_trade": 0.0,
            "expected_wr": 0.0,
            "expected_rr": 0.0,
            "why_it_works": "Fallback - awaiting valid strategy",
            "weaknesses": ["No strategy provided"],
            "improvements_made": [],
            "remaining_concerns": ["No strategy to validate"],
            "confidence": 0.0,
            "gladiator": "B"
        }

    # ==================== LLM API INTEGRATION ====================

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000
    ) -> str:
        """
        Call Claude API (Anthropic SDK).
        """
        if not self.api_key:
            logger.warning("Claude API key not set - using mock response")
            return self._mock_response()

        try:
            client = Anthropic(api_key=self.api_key)

            message = client.messages.create(
                model=self.MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )

            return message.content[0].text

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return self._mock_response()

    def _mock_response(self) -> str:
        """Mock response for testing."""
        return """{
  "approved": true,
  "strategy_name": "London Open Volatility - Validated",
  "structural_edge": "Session open volatility spike",
  "entry_rules": "Enter at London open (8AM UTC), require spread < 25 pips",
  "exit_rules": "TP at 1.5R, SL at 2x ATR, max hold 4 hours",
  "filters": ["spread_normal", "no_cb_meeting_24hrs", "volume_confirmation"],
  "risk_per_trade": 0.006,
  "expected_wr": 0.58,
  "expected_rr": 1.5,
  "why_it_works": "Session opens create temporary liquidity imbalances",
  "weaknesses": ["Fails during CB interventions", "Requires tight spread"],
  "improvements_made": ["Added max hold time", "Reduced position size", "Added volume confirmation"],
  "remaining_concerns": ["Still vulnerable to CB surprise announcements"],
  "confidence": 0.65
}"""
