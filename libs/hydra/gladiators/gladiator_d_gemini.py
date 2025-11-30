"""
HYDRA 3.0 - Gladiator D (Gemini)

Role: Synthesizer / Final Decision Maker

Specialization:
- Reviews all 3 gladiators' outputs
- Synthesizes best elements from each
- Makes final recommendation
- Tie-breaker when votes split 2-2
- Holistic view of strategy quality

Gemini is chosen for its strong synthesis and multi-perspective reasoning.
"""

from typing import Dict, List, Optional
from loguru import logger
import requests
import os

from .base_gladiator import BaseGladiator


class GladiatorD_Gemini(BaseGladiator):
    """
    Gladiator D: Synthesizer using Gemini (Google).

    Cost: ~$0.0002 per synthesis (very cheap with Gemini Flash)
    """

    GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            name="D",
            role="Synthesizer",
            api_key=api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")  # Support both env var names
        )

        if not self.api_key:
            logger.warning("Gemini API key not provided - gladiator will be in mock mode")

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
        Synthesize best strategy from all gladiators' outputs.

        Gladiator D reviews A/B/C outputs and creates final recommendation.
        """
        logger.info(f"Gladiator D synthesizing strategies for {asset}")

        if not existing_strategies or len(existing_strategies) < 3:
            return self._no_synthesis_possible(asset)

        # Get last 3 strategies (from A, B, C)
        strategy_a = existing_strategies[-3] if len(existing_strategies) >= 3 else None
        strategy_b = existing_strategies[-2] if len(existing_strategies) >= 2 else None
        strategy_c = existing_strategies[-1] if len(existing_strategies) >= 1 else None

        system_prompt = self._build_synthesis_system_prompt()
        user_prompt = self._build_synthesis_prompt(
            asset=asset,
            regime=regime,
            strategy_a=strategy_a,
            strategy_b=strategy_b,
            strategy_c=strategy_c
        )

        response = self._call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.5,
            max_tokens=2000
        )

        synthesis = self._parse_json_response(response)

        if synthesis:
            synthesis["strategy_id"] = f"GLADIATOR_D_{self.strategy_count:04d}"
            synthesis["gladiator"] = "D"
            synthesis["synthesized_from"] = [
                strategy_a.get("strategy_id", "A") if strategy_a else None,
                strategy_b.get("strategy_id", "B") if strategy_b else None,
                strategy_c.get("strategy_id", "C") if strategy_c else None
            ]
            self.strategy_count += 1

            logger.success(
                f"Gladiator D synthesized: {synthesis.get('strategy_name', 'Unknown')} "
                f"(final recommendation: {synthesis.get('final_recommendation', 'UNKNOWN')})"
            )
            return synthesis
        else:
            return self._no_synthesis_possible(asset)

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
        Vote as tie-breaker and synthesizer.

        Gladiator D considers the full picture.
        """
        logger.info(f"Gladiator D voting on {asset} {signal.get('direction', 'UNKNOWN')}")

        system_prompt = "You are Gladiator D, the final decision maker. Synthesize all perspectives."

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
            temperature=0.5,
            max_tokens=500
        )

        vote = self._parse_json_response(response)

        if vote:
            self._log_vote(vote)
            logger.info(f"Gladiator D votes: {vote.get('vote', 'UNKNOWN')} ({vote.get('confidence', 0):.1%})")
            return vote
        else:
            return {
                "vote": "HOLD",
                "confidence": 0.3,
                "reasoning": "Failed to parse vote response",
                "concerns": ["LLM response parsing error"]
            }

    # ==================== PROMPT ENGINEERING ====================

    def _build_synthesis_system_prompt(self) -> str:
        """Build system prompt for synthesis."""
        return """You are Gladiator D, the final synthesizer.

TOURNAMENT RULES:
- You are COMPETING against 3 other gladiators (A, B, C)
- Your strategies are tracked and scored
- Winners teach their insights to losers
- Losers must surpass the winners
- Only the best survive and evolve

PERFORMANCE MATTERS:
- Every vote is scored (correct prediction = +1 point)
- Losing gladiators learn from winners
- After 24 hours: lowest performer is "killed" (prompt reset)
- After 4 days: top performers "breed" (combine strategies)

You've received 3 perspectives:
- Gladiator A: Structural edge generation
- Gladiator B: Logic validation
- Gladiator C: Backtest simulation

Your job: SYNTHESIZE the best elements from each into a final recommendation.

Consider:
1. Which gladiator makes the most sense for THIS market?
2. What are the areas of agreement?
3. What are the critical disagreements?
4. Who should we trust more in this case?

Your final recommendation should:
- Take the BEST entry logic
- Take the BEST exit logic
- Include ALL critical filters
- Use MOST CONSERVATIVE position sizing
- Use MOST REALISTIC performance expectations

Output JSON:
{
  "final_recommendation": "APPROVE|REJECT|MODIFY",
  "strategy_name": "Final Strategy Name",
  "structural_edge": "Best edge from A/B/C",
  "entry_rules": "Best entry logic",
  "exit_rules": "Best exit logic",
  "filters": ["All critical filters combined"],
  "risk_per_trade": 0.006,
  "expected_wr": 0.58,
  "expected_rr": 1.4,
  "why_it_works": "Synthesized explanation",
  "weaknesses": ["All known weaknesses"],
  "gladiator_votes": {"A": "approve", "B": "approve", "C": "conditional"},
  "synthesis_notes": "How you combined A/B/C perspectives",
  "confidence": 0.7
}

If REJECT, explain why in synthesis_notes."""

    def _build_synthesis_prompt(
        self,
        asset: str,
        regime: str,
        strategy_a: Optional[Dict],
        strategy_b: Optional[Dict],
        strategy_c: Optional[Dict]
    ) -> str:
        """Build prompt for synthesis."""
        prompt = f"""Synthesize these 3 perspectives into final recommendation:

ASSET: {asset}
REGIME: {regime}

---
GLADIATOR A (Structural Edge):
{strategy_a.get('strategy_name', 'N/A') if strategy_a else 'N/A'}
Edge: {strategy_a.get('structural_edge', 'N/A') if strategy_a else 'N/A'}
Confidence: {(strategy_a.get('confidence', 0) if strategy_a else 0):.1%}

---
GLADIATOR B (Logic Validator):
Approved: {strategy_b.get('approved', False) if strategy_b else False}
Improvements: {', '.join(strategy_b.get('improvements_made', [])) if strategy_b else 'None'}
Concerns: {', '.join(strategy_b.get('remaining_concerns', [])) if strategy_b else 'None'}
Confidence: {(strategy_b.get('confidence', 0) if strategy_b else 0):.1%}

---
GLADIATOR C (Backtester):
Backtest Passed: {strategy_c.get('backtest_passed', False) if strategy_c else False}
Estimated WR: {(strategy_c.get('estimated_wr', 0) if strategy_c else 0):.1%}
Est. Trades/Month: {strategy_c.get('estimated_trades_per_month', 0) if strategy_c else 0}
Adjustments: {', '.join(strategy_c.get('adjustments_recommended', [])) if strategy_c else 'None'}
Confidence: {(strategy_c.get('confidence', 0) if strategy_c else 0):.1%}

---

Your task:
1. Evaluate each gladiator's analysis
2. Identify best elements from each
3. Synthesize into final strategy
4. Make APPROVE/REJECT/MODIFY recommendation

Be the tie-breaker. Make the final call."""

        return prompt

    def _build_vote_prompt(
        self,
        asset: str,
        regime: str,
        strategy: Dict,
        signal: Dict,
        market_data: Dict
    ) -> str:
        """Build prompt for final vote."""
        # Regime-based guidance
        regime_guidance = {
            "TRENDING_UP": "TRENDING_UP regime → Synthesize toward BUY if gladiators support uptrend",
            "TRENDING_DOWN": "TRENDING_DOWN regime → Synthesize toward SELL if gladiators support downtrend",
            "RANGING": "RANGING regime → Synthesize toward HOLD or mean reversion",
            "CHOPPY": "CHOPPY regime → Strong synthesis toward HOLD (risk management)",
            "BREAKOUT": "BREAKOUT regime → Synthesize based on breakout direction",
            "VOLATILE": "VOLATILE regime → Synthesize toward reduced risk or HOLD"
        }

        guidance = regime_guidance.get(regime, "Neutral regime")

        return f"""Final decision on trading direction:

ASSET: {asset}
REGIME: {regime}

REGIME GUIDANCE:
{guidance}

STRATEGY: {strategy.get('strategy_name', 'Unknown')}

Current Market:
- Price: {market_data.get('close', 'N/A')}
- Spread: {market_data.get('spread', 'N/A')}
- Volume: {market_data.get('volume', 'N/A')}

Consider all factors:
1. Market regime and trend direction (PRIMARY FACTOR)
2. Strategy structural edge
3. Risk/reward profile
4. Current market conditions
5. Timing and liquidity

VOTE ON DIRECTION:
- BUY if edge + regime favor LONG position
- SELL if edge + regime favor SHORT position
- HOLD if no clear edge or excessive risk

CRITICAL: This is the FINAL vote. In TRENDING_DOWN regimes, SELL should be strongly considered unless there's exceptional counter-trend edge.

This is the final vote. Be decisive and regime-aligned.

Output JSON:
{{
  "vote": "BUY|SELL|HOLD",
  "confidence": 0.75,
  "reasoning": "Final synthesis of all factors (mention regime alignment)",
  "concerns": ["Final concerns"]
}}"""

    def _no_synthesis_possible(self, asset: str) -> Dict:
        """Fallback when synthesis not possible."""
        return {
            "strategy_id": f"GLADIATOR_D_{self.strategy_count:04d}_NO_SYNTHESIS",
            "final_recommendation": "REJECT",
            "strategy_name": f"No Synthesis - {asset}",
            "structural_edge": "None",
            "entry_rules": "N/A",
            "exit_rules": "N/A",
            "filters": [],
            "risk_per_trade": 0.0,
            "expected_wr": 0.0,
            "expected_rr": 0.0,
            "why_it_works": "No strategies to synthesize",
            "weaknesses": ["Insufficient gladiator inputs"],
            "gladiator_votes": {},
            "synthesis_notes": "Need A/B/C outputs first",
            "confidence": 0.0,
            "gladiator": "D"
        }

    # ==================== LLM API INTEGRATION ====================

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.5,
        max_tokens: int = 2000
    ) -> str:
        """
        Call Gemini API (Google).
        """
        if not self.api_key:
            logger.warning("Gemini API key not set - using mock response")
            return self._mock_response()

        url = f"{self.GEMINI_API_URL}?key={self.api_key}"

        payload = {
            "contents": [{
                "parts": [{
                    "text": f"{system_prompt}\n\n{user_prompt}"
                }]
            }],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini API error: {e}")
            return self._mock_response()

    def _mock_response(self) -> str:
        """Mock response for testing."""
        return """{
  "final_recommendation": "APPROVE",
  "strategy_name": "London Open Volatility - Final",
  "structural_edge": "Session open volatility spike (from A)",
  "entry_rules": "Enter at London open, spread < 20 pips (tightened from B)",
  "exit_rules": "TP at 1.4R (conservative from C), SL at 2x ATR, max 4hr hold",
  "filters": ["spread_normal", "no_cb_meeting_24hrs", "volume_confirmation", "news_calendar"],
  "risk_per_trade": 0.006,
  "expected_wr": 0.56,
  "expected_rr": 1.35,
  "why_it_works": "Session opens create temporary liquidity imbalances (A), validated logic (B), historical precedent (C)",
  "weaknesses": ["CB interventions", "Wide spreads", "News events"],
  "gladiator_votes": {
    "A": "Strong structural edge identified",
    "B": "Approved with improvements",
    "C": "Realistic backtest, passed"
  },
  "synthesis_notes": "Combined A's edge discovery, B's logic improvements, and C's realistic expectations. Tightened spread requirement and added news filter. Conservative sizing.",
  "confidence": 0.72
}"""
