"""
HYDRA 3.0 - Chat Interface

User interaction layer for gladiator Q&A and recommendations.

Features:
- Ask gladiators questions
- Get market recommendations
- Store user feedback for improvement
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from loguru import logger
import json
from pathlib import Path


class HydraChat:
    """
    Chat interface for user <> gladiator interaction.

    Modes:
    1. Question Mode: Ask gladiators about trades, strategies, market
    2. Recommendation Mode: Get gladiator analysis on demand
    3. Feedback Mode: Store user feedback for improvement
    """

    def __init__(self, gladiators: Dict):
        """
        Initialize chat interface.

        Args:
            gladiators: Dict of {"A": gladiator_a, "B": gladiator_b, ...}
        """
        self.engines = gladiators
        from .config import CHAT_HISTORY_FILE, USER_FEEDBACK_FILE
        self.chat_history_file = CHAT_HISTORY_FILE
        self.feedback_file = USER_FEEDBACK_FILE

        # Ensure data directory exists
        self.chat_history_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info("HydraChat initialized")

    def ask_engine(
        self,
        user_message: str,
        target: str = "all",
        context: Optional[Dict] = None
    ) -> Dict[str, str]:
        """
        Ask gladiator(s) a question.

        Args:
            user_message: User's question
            target: "A", "B", "C", "D", or "all"
            context: Optional market context (asset, regime, price, etc.)

        Returns:
            Dict of {engine_name: response}
        """
        logger.info(f"User asked: {user_message} (target: {target})")

        # Build system prompt
        system_prompt = self._build_qa_system_prompt(context)

        # Route to gladiator(s)
        responses = {}

        if target == "all":
            for name, gladiator in self.engines.items():
                try:
                    response = gladiator._call_llm(
                        system_prompt=system_prompt,
                        user_prompt=user_message,
                        temperature=0.7,
                        max_tokens=500
                    )
                    responses[name] = response
                    logger.info(f"Gladiator {name} responded: {response[:100]}...")
                except Exception as e:
                    logger.error(f"Gladiator {name} error: {e}")
                    responses[name] = f"Error: {str(e)}"
        else:
            if target not in self.engines:
                return {"error": f"Unknown gladiator: {target}"}

            try:
                gladiator = self.engines[target]
                response = gladiator._call_llm(
                    system_prompt=system_prompt,
                    user_prompt=user_message,
                    temperature=0.7,
                    max_tokens=500
                )
                responses[target] = response
                logger.info(f"Gladiator {target} responded: {response[:100]}...")
            except Exception as e:
                logger.error(f"Gladiator {target} error: {e}")
                responses[target] = f"Error: {str(e)}"

        # Save to chat history
        self._save_chat_history(
            user_message=user_message,
            gladiator_responses=responses,
            context=context
        )

        return responses

    def get_recommendation(
        self,
        asset: str,
        market_data: Dict,
        regime: str,
        ask_all: bool = True
    ) -> Dict[str, Any]:
        """
        Get gladiator recommendations for an asset.

        This is a structured recommendation request (not free-form chat).

        Args:
            asset: Asset symbol (e.g., "BTC-USD")
            market_data: Current market data (price, volume, ATR, etc.)
            regime: Current market regime
            ask_all: If True, ask all gladiators; if False, only D (synthesizer)

        Returns:
            {
                "asset": "BTC-USD",
                "regime": "TRENDING_DOWN",
                "timestamp": "2025-11-30...",
                "recommendations": {
                    "A": {vote: "SELL", confidence: 0.8, reasoning: "..."},
                    "B": {...},
                    ...
                },
                "consensus": "SELL",
                "user_feedback": null  # To be filled later
            }
        """
        logger.info(f"Getting recommendations for {asset} ({regime})")

        # Build recommendation prompt
        prompt = self._build_recommendation_prompt(asset, market_data, regime)

        recommendations = {}

        if ask_all:
            # Ask all 4 gladiators
            for name, gladiator in self.engines.items():
                try:
                    response = gladiator._call_llm(
                        system_prompt=self._build_recommendation_system_prompt(name),
                        user_prompt=prompt,
                        temperature=0.5,
                        max_tokens=400
                    )

                    # Try to parse JSON response
                    try:
                        parsed = json.loads(response)
                        recommendations[name] = parsed
                    except json.JSONDecodeError:
                        recommendations[name] = {
                            "vote": "HOLD",
                            "confidence": 0.5,
                            "reasoning": response,
                            "parse_error": True
                        }
                except Exception as e:
                    logger.error(f"Gladiator {name} recommendation error: {e}")
                    recommendations[name] = {
                        "vote": "HOLD",
                        "confidence": 0.0,
                        "reasoning": f"Error: {str(e)}",
                        "error": True
                    }
        else:
            # Only ask Gladiator D (synthesizer)
            try:
                gladiator_d = self.engines["D"]
                response = gladiator_d._call_llm(
                    system_prompt=self._build_recommendation_system_prompt("D"),
                    user_prompt=prompt,
                    temperature=0.5,
                    max_tokens=400
                )

                try:
                    parsed = json.loads(response)
                    recommendations["D"] = parsed
                except json.JSONDecodeError:
                    recommendations["D"] = {
                        "vote": "HOLD",
                        "confidence": 0.5,
                        "reasoning": response,
                        "parse_error": True
                    }
            except Exception as e:
                logger.error(f"Gladiator D recommendation error: {e}")
                recommendations["D"] = {
                    "vote": "HOLD",
                    "confidence": 0.0,
                    "reasoning": f"Error: {str(e)}",
                    "error": True
                }

        # Calculate consensus
        votes = [r.get("vote", "HOLD") for r in recommendations.values()]
        buy_count = votes.count("BUY")
        sell_count = votes.count("SELL")

        if buy_count > sell_count:
            consensus = "BUY"
        elif sell_count > buy_count:
            consensus = "SELL"
        else:
            consensus = "HOLD"

        result = {
            "asset": asset,
            "regime": regime,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "market_data": market_data,
            "recommendations": recommendations,
            "consensus": consensus,
            "user_feedback": None
        }

        # Save recommendation
        self._save_recommendation(result)

        return result

    def save_feedback(
        self,
        recommendation_id: str,
        feedback: Dict[str, Any]
    ):
        """
        Save user feedback on a recommendation.

        Args:
            recommendation_id: Timestamp or ID of recommendation
            feedback: {
                "helpful": true/false,
                "accurate": true/false,
                "user_vote": "BUY|SELL|HOLD",
                "notes": "User comments...",
                "outcome": "win|loss|neutral"  # After trade closes
            }
        """
        logger.info(f"Saving feedback for recommendation {recommendation_id}")

        feedback_entry = {
            "recommendation_id": recommendation_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **feedback
        }

        try:
            with open(self.feedback_file, 'a') as f:
                f.write(json.dumps(feedback_entry) + '\n')
            logger.success(f"Feedback saved: {feedback_entry}")
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")

    # ==================== PRIVATE METHODS ====================

    def _build_qa_system_prompt(self, context: Optional[Dict]) -> str:
        """Build system prompt for Q&A mode."""
        prompt = """You are a HYDRA gladiator - a specialist AI trader.

Answer the user's question clearly and concisely based on your specialty:
- Gladiator A: Structural edges (funding, liquidations, timing)
- Gladiator B: Logic validation (flaws, edge cases, risks)
- Gladiator C: Historical patterns (backtesting, precedents)
- Gladiator D: Synthesis (combining perspectives, final decisions)

Keep responses under 3-4 sentences unless more detail is needed.
Be honest about limitations and uncertainties."""

        if context:
            prompt += f"\n\nCurrent market context:"
            if "asset" in context:
                prompt += f"\n- Asset: {context['asset']}"
            if "regime" in context:
                prompt += f"\n- Regime: {context['regime']}"
            if "price" in context:
                prompt += f"\n- Price: {context['price']}"

        return prompt

    def _build_recommendation_system_prompt(self, engine_name: str) -> str:
        """Build system prompt for recommendation mode."""
        return f"""You are Gladiator {engine_name}, providing a trading recommendation.

Analyze the market data and provide your vote.

Output JSON:
{{
  "vote": "BUY|SELL|HOLD",
  "confidence": 0.75,
  "reasoning": "Brief explanation (2-3 sentences)",
  "concerns": ["Any risks or caveats"]
}}"""

    def _build_recommendation_prompt(
        self,
        asset: str,
        market_data: Dict,
        regime: str
    ) -> str:
        """Build user prompt for recommendation request."""
        return f"""Provide your trading recommendation for this market:

ASSET: {asset}
REGIME: {regime}

MARKET DATA:
- Price: {market_data.get('close', 'N/A')}
- Volume: {market_data.get('volume', 'N/A')}
- ATR: {market_data.get('atr', 'N/A')}
- Spread: {market_data.get('spread', 'N/A')}

Should we BUY, SELL, or HOLD? Explain your reasoning."""

    def _save_chat_history(
        self,
        user_message: str,
        gladiator_responses: Dict[str, str],
        context: Optional[Dict]
    ):
        """Save chat interaction to history file."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_message": user_message,
            "gladiator_responses": gladiator_responses,
            "context": context
        }

        try:
            with open(self.chat_history_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to save chat history: {e}")

    def _save_recommendation(self, recommendation: Dict):
        """Save recommendation to file for feedback tracking."""
        from .config import RECOMMENDATIONS_FILE
        rec_file = RECOMMENDATIONS_FILE

        try:
            with open(rec_file, 'a') as f:
                f.write(json.dumps(recommendation) + '\n')
        except Exception as e:
            logger.error(f"Failed to save recommendation: {e}")
