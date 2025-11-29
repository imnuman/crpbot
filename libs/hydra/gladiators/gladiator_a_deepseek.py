"""
HYDRA 3.0 - Gladiator A (DeepSeek)

Role: Structural Edge Generator

Specialization:
- Identifies STRUCTURAL edges (not chart patterns)
- Focuses on market mechanics (funding, liquidations, session timing)
- BANS retail patterns (double tops, head & shoulders, etc.)
- Uses first-principles thinking

DeepSeek is chosen for its reasoning capabilities at low cost.
"""

from typing import Dict, List, Optional
from loguru import logger
import requests
import os

from .base_gladiator import BaseGladiator


class GladiatorA_DeepSeek(BaseGladiator):
    """
    Gladiator A: Structural Edge Generator using DeepSeek.

    Cost: ~$0.0001 per strategy generation
    """

    DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
    MODEL = "deepseek-chat"  # Latest DeepSeek model

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            name="A",
            role="Structural Edge Generator",
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY")
        )

        if not self.api_key:
            logger.warning("DeepSeek API key not provided - gladiator will be in mock mode")

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
        Generate structural edge strategy for this asset/regime.
        """
        logger.info(f"Gladiator A generating strategy for {asset} ({regime})")

        # Build system prompt
        system_prompt = self._build_system_prompt(asset_type)

        # Build user prompt with all context
        user_prompt = self._build_strategy_generation_prompt(
            asset=asset,
            asset_type=asset_type,
            asset_profile=asset_profile,
            regime=regime,
            regime_confidence=regime_confidence,
            market_data=market_data,
            existing_strategies=existing_strategies
        )

        # Call DeepSeek API
        response = self._call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.7,
            max_tokens=2000
        )

        # Parse JSON response
        strategy = self._parse_json_response(response)

        if strategy:
            # Add metadata
            strategy["strategy_id"] = f"GLADIATOR_A_{self.strategy_count:04d}"
            strategy["gladiator"] = "A"
            strategy["generated_at"] = market_data.get("timestamp", "unknown")
            self.strategy_count += 1

            logger.success(
                f"Gladiator A generated: {strategy.get('strategy_name', 'Unknown')} "
                f"(confidence: {strategy.get('confidence', 0):.1%})"
            )
            return strategy
        else:
            logger.error("Failed to parse strategy from DeepSeek response")
            return self._fallback_strategy(asset, regime)

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
        Vote on whether to take a specific trade.
        """
        logger.info(f"Gladiator A voting on {asset} {signal.get('direction', 'UNKNOWN')}")

        system_prompt = "You are Gladiator A, a structural edge specialist. Vote on this trade signal."

        user_prompt = self._build_vote_prompt(
            asset=asset,
            asset_type=asset_type,
            regime=regime,
            strategy=strategy,
            signal=signal,
            market_data=market_data
        )

        response = self._call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.5,  # Lower temperature for voting
            max_tokens=500
        )

        vote = self._parse_json_response(response)

        if vote:
            self._log_vote(vote)
            logger.info(f"Gladiator A votes: {vote.get('vote', 'UNKNOWN')} ({vote.get('confidence', 0):.1%})")
            return vote
        else:
            # Fallback vote
            return {
                "vote": "HOLD",
                "confidence": 0.3,
                "reasoning": "Failed to parse vote response",
                "concerns": ["LLM response parsing error"]
            }

    # ==================== PROMPT ENGINEERING ====================

    def _build_system_prompt(self, asset_type: str) -> str:
        """Build system prompt for strategy generation."""
        return f"""You are Gladiator A, a quantitative trader specializing in STRUCTURAL edges.

Your job: Generate trading strategies based on MARKET MECHANICS, not chart patterns.

BANNED PATTERNS (never use these):
- Double tops / double bottoms
- Head and shoulders
- Triangles, wedges, flags
- Support/resistance lines
- Fibonacci retracements
- Moving average crossovers
- RSI overbought/oversold
- MACD divergences

ALLOWED EDGES (use ONLY these):
1. Funding rate arbitrage (crypto perps)
2. Liquidation clusters (hunt stops)
3. Session timing (London/NY open volatility)
4. Carry trade unwinding
5. Whale movements (large wallet transfers)
6. Order book imbalances (spoofing detection)
7. Cross-asset correlation breaks
8. Central bank intervention patterns
9. Market maker behavior (spread widening)
10. Time-of-day patterns (Asia session pumps)

Asset Type: {asset_type}

Output MUST be valid JSON with this structure:
{{
  "strategy_name": "Clear, descriptive name",
  "structural_edge": "Which mechanic are you exploiting?",
  "entry_rules": "Precise entry conditions",
  "exit_rules": "Precise exit conditions (TP/SL)",
  "filters": ["List of required filters"],
  "risk_per_trade": 0.008,
  "expected_wr": 0.63,
  "expected_rr": 1.5,
  "why_it_works": "First-principles explanation",
  "weaknesses": ["Known failure modes"],
  "confidence": 0.75
}}

Be BRUTALLY HONEST about weaknesses. We prefer 1 great strategy over 10 mediocre ones."""

    def _build_strategy_generation_prompt(
        self,
        asset: str,
        asset_type: str,
        asset_profile: Dict,
        regime: str,
        regime_confidence: float,
        market_data: Dict,
        existing_strategies: Optional[List[Dict]]
    ) -> str:
        """Build user prompt for strategy generation."""
        prompt = f"""Generate a structural edge strategy for this market:

ASSET: {asset}
TYPE: {asset_type}
REGIME: {regime} (confidence: {regime_confidence:.1%})

ASSET PROFILE:
- Spread (normal): {asset_profile.get('spread_normal', 'N/A')}
- Manipulation risk: {asset_profile.get('manipulation_risk', 'N/A')}
- Best sessions: {', '.join(asset_profile.get('best_sessions', []))}
- Overnight allowed: {asset_profile.get('overnight_allowed', False)}
- Max hold: {asset_profile.get('max_hold_hours', 'N/A')} hours
- Special rules: {', '.join(asset_profile.get('special_rules', []))[:200]}

MARKET DATA:
- Current price: {market_data.get('close', 'N/A')}
- 24h volume: {market_data.get('volume_24h', 'N/A')}
- ATR: {market_data.get('atr', 'N/A')}
"""

        if asset_type == "meme_perp":
            prompt += f"""
CRYPTO-SPECIFIC:
- Funding rate: {market_data.get('funding_rate', 'N/A')}
- Funding threshold: {asset_profile.get('funding_threshold', 'N/A')}
- Whale threshold: ${asset_profile.get('whale_threshold', 'N/A'):,}
"""

        if existing_strategies:
            prompt += f"\nEXISTING STRATEGIES: {len(existing_strategies)} already generated (avoid duplication)"

        prompt += "\n\nGenerate a NEW structural edge strategy that exploits this market's unique characteristics."

        return prompt

    def _build_vote_prompt(
        self,
        asset: str,
        asset_type: str,
        regime: str,
        strategy: Dict,
        signal: Dict,
        market_data: Dict
    ) -> str:
        """Build prompt for voting on a trade."""
        return f"""Vote on this trade signal:

ASSET: {asset}
REGIME: {regime}
DIRECTION: {signal.get('direction', 'UNKNOWN')}
ENTRY: {signal.get('entry_price', 'N/A')}
SL: {signal.get('sl_price', 'N/A')}
TP: {signal.get('tp_price', 'N/A')}

STRATEGY: {strategy.get('strategy_name', 'Unknown')}
STRUCTURAL EDGE: {strategy.get('structural_edge', 'Unknown')}

CURRENT MARKET:
- Price: {market_data.get('close', 'N/A')}
- Spread: {market_data.get('spread', 'N/A')}
- Volume: {market_data.get('volume', 'N/A')}

Vote: BUY, SELL, or HOLD

Output JSON:
{{
  "vote": "BUY|SELL|HOLD",
  "confidence": 0.75,
  "reasoning": "Why you voted this way",
  "concerns": ["Any red flags you see"]
}}"""

    def _fallback_strategy(self, asset: str, regime: str) -> Dict:
        """Fallback strategy if LLM fails."""
        return {
            "strategy_id": f"GLADIATOR_A_{self.strategy_count:04d}_FALLBACK",
            "strategy_name": f"Fallback Strategy - {asset}",
            "structural_edge": "None (fallback)",
            "entry_rules": "No entry",
            "exit_rules": "No exit",
            "filters": [],
            "risk_per_trade": 0.005,
            "expected_wr": 0.5,
            "expected_rr": 1.0,
            "why_it_works": "Fallback due to LLM failure",
            "weaknesses": ["Fallback strategy - not actionable"],
            "confidence": 0.0,
            "gladiator": "A",
            "fallback": True
        }

    # ==================== LLM API INTEGRATION ====================

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """
        Call DeepSeek API.
        """
        if not self.api_key:
            logger.warning("DeepSeek API key not set - using mock response")
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
                self.DEEPSEEK_API_URL,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            return data["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            logger.error(f"DeepSeek API error: {e}")
            return self._mock_response()

    def _mock_response(self) -> str:
        """Mock response for testing without API key."""
        return """{
  "strategy_name": "London Open Volatility - Test",
  "structural_edge": "Session open volatility spike",
  "entry_rules": "Enter at London open (8AM UTC)",
  "exit_rules": "TP at 1.5R, SL at 2x ATR",
  "filters": ["spread_normal"],
  "risk_per_trade": 0.008,
  "expected_wr": 0.60,
  "expected_rr": 1.5,
  "why_it_works": "Mock strategy for testing",
  "weaknesses": ["This is a mock strategy"],
  "confidence": 0.5
}"""
