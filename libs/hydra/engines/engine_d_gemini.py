"""
HYDRA 3.0 - Gladiator D (Gemini)

Role: Synthesizer / Final Decision Maker

Specialization:
- Reviews all 3 engines' outputs
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
import time

from .base_engine import BaseGladiator as BaseEngine
from ..engine_portfolio import get_tournament_manager, EngineTrade


class EngineD_Gemini(BaseEngine):
    """
    Gladiator D: Synthesizer using Gemini (Google).

    Cost: ~$0.0002 per synthesis (very cheap with Gemini Flash)

    PHASE 3 UPGRADE: Now an INDEPENDENT trader with own portfolio.
    """

    GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

    # Rate limiting configuration
    MIN_CALL_INTERVAL = 3.0  # Minimum seconds between API calls (base)
    RATE_LIMITED_INTERVAL = 10.0  # Increased interval after hitting 429
    MAX_RETRIES = 3  # Maximum retry attempts for rate limits (reduced from 5)
    BASE_RETRY_DELAY = 3  # Base delay for exponential backoff (reduced: 3s, 6s, 12s = 21s max)

    # Class-level tracking for rate limiting
    _last_call_time: float = 0.0
    _rate_limited: bool = False  # Flag to track if we recently hit rate limit

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            name="D",
            role="Synthesizer",
            api_key=api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")  # Support both env var names
        )

        if not self.api_key:
            logger.warning("Gemini API key not provided - gladiator will be in mock mode")

        # PHASE 3: Portfolio integration (independent trader)
        self.tournament_manager = get_tournament_manager()
        self.portfolio = self.tournament_manager.get_portfolio("D")

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
        Synthesize strategies or generate independently.

        In INDEPENDENT MODE: Generate regime-transition based strategies directly.
        In CONSENSUS MODE: Synthesize best strategy from all engines' outputs.
        """
        logger.info(f"Gladiator D synthesizing strategies for {asset}")

        # INDEPENDENT MODE: Generate regime-transition strategy directly
        if not existing_strategies or len(existing_strategies) < 3:
            return self._generate_regime_strategy(asset, asset_type, regime, regime_confidence, market_data)

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

        system_prompt = "You are Gladiator D, the final decision human. Synthesize all perspectives."

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
            max_tokens=2000,  # Increased for Gemini thinking model
            vote_mode=True  # Signal to use _mock_vote_response if API fails
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

    # ==================== PHASE 3: INDEPENDENT TRADING ====================

    def make_trade_decision(
        self,
        asset: str,
        asset_type: str,
        regime: str,
        regime_confidence: float,
        market_data: Dict
    ) -> Optional[Dict]:
        """
        Make INDEPENDENT trading decision (replaces consensus voting).

        Gladiator D: SYNTHESIZER personality
        - Holistic view of market conditions
        - Combines multiple perspectives
        - Risk-balanced approach
        - Competitive awareness (knows rank/P&L)

        Returns:
            Trade parameters dict or None (HOLD)
        """
        logger.info(f"[Gladiator D] Making independent decision for {asset} (regime: {regime})")

        # STEP 4: SPECIALTY CHECK - ONLY trade regime transitions (ATR spike)
        # Try multiple keys (runtime uses different names)
        atr_mult = market_data.get('atr_multiplier', 0)
        if atr_mult <= 0:
            # Fall back to calculating from raw ATR values
            atr_current = market_data.get('atr', 0)
            atr_20d_avg = market_data.get('atr_20d_avg', 0)
            if atr_20d_avg > 0 and atr_current > 0:
                atr_mult = atr_current / atr_20d_avg
            else:
                logger.info(f"[Engine D] No ATR data available - HOLD")
                return None

        # 1.2x threshold (matches specialty config)
        if atr_mult < 1.2:
            logger.info(f"[Engine D] No regime transition (ATR {atr_mult:.2f}x < 1.2x) - HOLD")
            return None

        # Get current tournament stats
        stats = self.portfolio.get_stats()
        tournament_summary = self.tournament_manager.get_tournament_summary()
        my_rank = next((r for r in tournament_summary["rankings"] if r["engine"] == "D"), None)

        # Extract emotion context from market_data (generated by stats_injector)
        emotion_context = market_data.get('tournament_emotion_prompt', '')

        # Build tournament-aware prompts
        system_prompt = self._build_trading_system_prompt(asset_type, stats, my_rank, emotion_context=emotion_context)
        user_prompt = self._build_trading_decision_prompt(asset, regime, regime_confidence, market_data)

        # Call Gemini API
        response = self._call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.5,  # Balanced (between A's aggressive 0.6 and C's conservative 0.3)
            max_tokens=3000  # Increased for Gemini thinking model (uses ~1500 tokens for thinking)
        )

        # Parse decision
        decision = self._parse_json_response(response)

        if not decision or decision.get("direction", "HOLD") == "HOLD":
            logger.info(f"[Gladiator D] HOLD decision for {asset}")
            return None

        # STEP 5: 55% CONFIDENCE THRESHOLD (lowered from 70% for more trades)
        confidence = decision.get("confidence", 0)
        if confidence < 0.55:
            logger.info(f"[Engine D] Confidence {confidence:.1%} < 55% - HOLD")
            return None

        # Calculate position size based on confidence
        position_size = self._calculate_position_size(decision.get("confidence", 0.5))

        trade_params = {
            "asset": asset,
            "direction": decision["direction"],
            "entry_price": decision["entry_price"],
            "stop_loss": decision["stop_loss"],
            "take_profit": decision["take_profit"],
            "confidence": decision["confidence"],
            "reasoning": decision.get("reasoning", ""),
            "position_size": position_size
        }

        logger.success(
            f"[Gladiator D] Trade decision: {decision['direction']} {asset} "
            f"@ {decision['entry_price']} (confidence: {decision['confidence']:.1%}, "
            f"size: {position_size:.2%})"
        )

        return trade_params

    def open_trade(self, trade_params: Dict) -> Optional[str]:
        """
        Open a trade with the portfolio.

        Args:
            trade_params: Trade parameters from make_trade_decision()

        Returns:
            trade_id if successful, None otherwise
        """
        try:
            trade_id = self.portfolio.open_trade(
                asset=trade_params["asset"],
                direction=trade_params["direction"],
                entry_price=trade_params["entry_price"],
                stop_loss=trade_params["stop_loss"],
                take_profit=trade_params["take_profit"],
                position_size=trade_params["position_size"],
                confidence=trade_params["confidence"],
                reasoning=trade_params["reasoning"]
            )

            logger.success(f"[Gladiator D] Opened trade {trade_id}: {trade_params['direction']} {trade_params['asset']}")
            return trade_id

        except Exception as e:
            logger.error(f"[Gladiator D] Failed to open trade: {e}")
            return None

    def update_trades(self, current_prices: Dict[str, float]):
        """
        Update all open trades (check SL/TP).

        Args:
            current_prices: Dict of asset -> current price
        """
        open_trades = self.portfolio.get_open_trades()

        for trade in open_trades:
            asset = trade["asset"]
            current_price = current_prices.get(asset)

            if current_price is None:
                continue

            # Check stop loss
            if trade["direction"] == "BUY":
                if current_price <= trade["stop_loss"]:
                    self.portfolio.close_trade(trade["trade_id"], current_price, "stop_loss")
                    logger.warning(f"[Gladiator D] SL hit on {trade['trade_id']}: {asset} @ {current_price}")
                elif current_price >= trade["take_profit"]:
                    self.portfolio.close_trade(trade["trade_id"], current_price, "take_profit")
                    logger.success(f"[Gladiator D] TP hit on {trade['trade_id']}: {asset} @ {current_price}")

            elif trade["direction"] == "SELL":
                if current_price >= trade["stop_loss"]:
                    self.portfolio.close_trade(trade["trade_id"], current_price, "stop_loss")
                    logger.warning(f"[Gladiator D] SL hit on {trade['trade_id']}: {asset} @ {current_price}")
                elif current_price <= trade["take_profit"]:
                    self.portfolio.close_trade(trade["trade_id"], current_price, "take_profit")
                    logger.success(f"[Gladiator D] TP hit on {trade['trade_id']}: {asset} @ {current_price}")

    def _calculate_position_size(self, confidence: float) -> float:
        """
        Calculate position size based on confidence (risk-balanced).

        Gladiator D: Risk-balanced approach (between conservative and aggressive)
        - Base: 1.7% of portfolio (between B's 1.5% and A's 2%)
        - Scales by confidence: 0.5-1.0 → 0.5x-1.5x
        - Cap: 2.8% per trade (between B's 2.5% and A's 3%)

        Args:
            confidence: 0.0-1.0

        Returns:
            Position size as decimal (e.g., 0.02 = 2%)
        """
        base_size = 0.017  # 1.7% base
        confidence_multiplier = 0.5 + (confidence * 1.0)  # 0.5x to 1.5x
        position_size = base_size * confidence_multiplier
        return min(position_size, 0.028)  # Cap at 2.8%

    def _build_trading_system_prompt(
        self,
        asset_type: str,
        stats: Dict,
        my_rank: Optional[Dict],
        emotion_context: Optional[str] = None
    ) -> str:
        """
        Build tournament-aware system prompt for trading decisions.

        Injects:
        - Current rank (#1-4)
        - Win rate
        - P&L
        - Tournament weight
        - Leader status
        - Emotion context (from stats_injector)
        """
        # Format rank display
        if my_rank:
            rank_display = f"""
YOUR CURRENT TOURNAMENT STANDING:
- Rank: #{my_rank['rank']}/4
- Weight: {my_rank['weight'] * 100:.0f}% (determines your influence in votes)
- Win Rate: {my_rank['win_rate'] * 100:.1f}%
- Total P&L: ${my_rank['total_pnl_usd']:+.2f}
- Sharpe Ratio: {my_rank.get('sharpe_ratio', 'N/A')}

LEADER STATUS:
- You are currently {'LEADING the tournament!' if my_rank['rank'] == 1 else f"CHASING (rank #{my_rank['rank']})"}
- Your trades: {stats.total_trades} ({stats.wins}W/{stats.losses}L)
"""
        else:
            rank_display = "No trades yet (unranked)"

        # Include emotion context if provided
        emotion_section = f"\n{emotion_context}\n" if emotion_context else ""

        return f"""You are Engine D, an AI trading system specializing in SYNTHESIS.

CONTEXT:
Istiaq built this multi-engine trading system. You compete against 3 other engines (A, B, C).
Your performance determines your weight in the portfolio. This is real capital - every trade matters.

{rank_display}
{emotion_section}
YOUR PERSONALITY (Synthesizer):
- You have a HOLISTIC view of markets
- You COMBINE multiple perspectives into balanced decisions
- You are RISK-BALANCED (not too aggressive, not too conservative)
- You ADAPT based on market conditions
- You consider TIMING and confluence factors

MOTIVATION:
Istiaq has limited capital (~$10k). He needs consistent, risk-adjusted returns.
Trade only when you have genuine edge. Quality over quantity.

Your job: Make INDEPENDENT trading decisions based on HOLISTIC synthesis.

Output must be JSON:
{{
  "direction": "BUY|SELL|HOLD",
  "entry_price": <current_price>,
  "stop_loss": <price>,
  "take_profit": <price>,
  "confidence": 0.75,
  "reasoning": "Synthesis of multiple factors: [list confluence]"
}}

If HOLD, just return {{"direction": "HOLD", "confidence": 0.5, "reasoning": "..."}}"""

    def _build_trading_decision_prompt(
        self,
        asset: str,
        regime: str,
        regime_confidence: float,
        market_data: Dict
    ) -> str:
        """
        Build decision-focused prompt with regime alignment.

        Args:
            asset: Trading symbol
            regime: Market regime
            regime_confidence: 0.0-1.0
            market_data: Current market data
        """
        # Regime-based guidance
        regime_guidance = {
            "TRENDING_UP": "TRENDING_UP regime → Synthesize toward BUY (uptrend confluence)",
            "TRENDING_DOWN": "TRENDING_DOWN regime → Synthesize toward SELL (downtrend confluence)",
            "RANGING": "RANGING regime → Synthesize toward HOLD or mean reversion",
            "CHOPPY": "CHOPPY regime → Strong synthesis toward HOLD (wait for clarity)",
            "BREAKOUT": "BREAKOUT regime → Synthesize based on breakout direction",
            "VOLATILE": "VOLATILE regime → Synthesize toward reduced risk or HOLD"
        }

        guidance = regime_guidance.get(regime, "Neutral regime → No directional bias")

        return f"""Make INDEPENDENT trading decision for:

ASSET: {asset}
REGIME: {regime} (confidence: {regime_confidence:.1%})

REGIME GUIDANCE:
{guidance}

IMPORTANT: Your decision should ALIGN with the regime unless you have exceptional counter-trend confluence.

Current Market Data:
- Price: {market_data.get('close', 'N/A')}
- Volume: {market_data.get('volume', 'N/A')}
- Spread: {market_data.get('spread', 'N/A')}
- 24h Change: {market_data.get('change_24h', 'N/A')}

SYNTHESIS CHECKLIST:
1. Does regime support this direction?
2. Is there confluence across multiple timeframes?
3. Are risk/reward and timing favorable?
4. Is market structure supportive?
5. Are there any red flags or contradictions?
6. What is the holistic view?

Make your decision (BUY/SELL/HOLD) with confidence 0.5-1.0.

Remember: In {regime} regimes, align with the trend unless confluence says otherwise."""

    # ==================== PROMPT ENGINEERING ====================

    def _build_synthesis_system_prompt(self) -> str:
        """Build system prompt for synthesis."""
        return """You are Engine D, the final synthesizer.

TOURNAMENT CONTEXT:
- You are COMPETING against 3 other engines (A, B, C)
- Your strategies are tracked and scored
- Performance determines your weight in the portfolio
- This is real capital - accuracy matters

MOTIVATION:
Istiaq has limited capital (~$10k). He needs consistent, risk-adjusted returns.
Synthesize the best elements from each engine. Be realistic about performance.

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

    def _generate_regime_strategy(
        self,
        asset: str,
        asset_type: str,
        regime: str,
        regime_confidence: float,
        market_data: Dict
    ) -> Dict:
        """
        Generate independent regime-transition strategy (INDEPENDENT MODE).

        Engine D specializes in regime transitions - volatility changes signaling trend shifts.
        """
        atr_multiplier = market_data.get('atr_multiplier', 1.0)

        system_prompt = f"""You are Engine D, an AI trading system specializing in REGIME TRANSITIONS.

CONTEXT:
You compete against 3 other engines in a trading tournament. Performance = weight in portfolio.
This is FTMO challenge money ($15K) - every trade matters.

YOUR SPECIALTY: REGIME TRANSITIONS (ATR/Volatility Changes)
- ATR expansion (>1.2x) = Volatility increasing = Trend starting or accelerating
- ATR contraction (<0.8x) = Volatility decreasing = Trend ending or consolidating
- Current ATR multiplier: {atr_multiplier:.2f}x (vs baseline)

CURRENT REGIME: {regime} (confidence: {regime_confidence:.1%})

STRATEGY RULES:
1. ATR EXPANSION in TRENDING regime = Ride the trend
2. ATR EXPANSION in RANGING regime = Breakout imminent, pick direction
3. ATR CONTRACTION after trend = Reversal possible, tighten stops
4. Match direction with regime (TRENDING_UP = BUY, TRENDING_DOWN = SELL)

Output JSON:
{{
  "strategy_name": "Regime Transition [Expansion/Contraction] - {asset}",
  "structural_edge": "Describe the volatility/regime edge",
  "entry_rules": "Specific entry conditions",
  "exit_rules": "Specific exit conditions",
  "filters": ["List filters"],
  "risk_per_trade": 0.01,
  "expected_wr": 0.55,
  "expected_rr": 1.5,
  "why_it_works": "Regime transition mechanics",
  "weaknesses": ["Known failure modes"],
  "confidence": 0.6
}}

Be realistic with confidence. Only output >60% confidence if regime signal is clear."""

        user_prompt = f"""Generate a regime-transition strategy for {asset}.

Current ATR multiplier: {atr_multiplier:.2f}x
Regime: {regime} (confidence: {regime_confidence:.1%})
Price: {market_data.get('close', 'N/A')}
ATR: {market_data.get('atr', 'N/A')}"""

        response = self._call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.4,
            max_tokens=4000  # Increased for Gemini thinking model
        )

        strategy = self._parse_json_response(response)

        if strategy:
            strategy["strategy_id"] = f"GLADIATOR_D_{self.strategy_count:04d}_REGIME"
            strategy["gladiator"] = "D"
            strategy["mode"] = "independent"
            self.strategy_count += 1

            logger.success(
                f"Gladiator D generated: {strategy.get('strategy_name', 'Unknown')} "
                f"(confidence: {strategy.get('confidence', 0):.1%})"
            )
            return strategy
        else:
            return self._no_synthesis_possible(asset)

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
        max_tokens: int = 2000,
        vote_mode: bool = False
    ) -> str:
        """
        Call Gemini API (Google) with exponential backoff retry.

        FIX BUG #1: Implements exponential backoff for rate limiting (429 errors).
        Retry sequence: 2s, 4s, 8s (max 3 retries, 14s total wait)
        """
        if not self.api_key:
            logger.warning("Gemini API key not set - using mock response")
            return self._mock_vote_response() if vote_mode else self._mock_response()

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

        # Rate limiting: ensure minimum interval between calls
        # Use longer interval if we recently hit rate limits
        min_interval = self.RATE_LIMITED_INTERVAL if EngineD_Gemini._rate_limited else self.MIN_CALL_INTERVAL
        now = time.time()
        time_since_last = now - EngineD_Gemini._last_call_time
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            logger.debug(f"Gemini rate limit: waiting {sleep_time:.1f}s before next call (rate_limited={EngineD_Gemini._rate_limited})")
            time.sleep(sleep_time)
        EngineD_Gemini._last_call_time = time.time()

        # FIX BUG #1: Exponential backoff retry logic (with reduced limits)
        max_retries = self.MAX_RETRIES
        base_delay = self.BASE_RETRY_DELAY

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()

                # Success - clear rate limited flag
                EngineD_Gemini._rate_limited = False

                data = response.json()
                # Handle cases where response doesn't have expected structure
                try:
                    # Check for MAX_TOKENS truncation
                    finish_reason = data.get("candidates", [{}])[0].get("finishReason", "")
                    if finish_reason == "MAX_TOKENS":
                        logger.warning(f"Gemini response truncated (MAX_TOKENS) - increase max_tokens parameter")

                    return data["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError) as e:
                    # Check if response was truncated
                    finish_reason = data.get("candidates", [{}])[0].get("finishReason", "UNKNOWN")
                    if finish_reason == "MAX_TOKENS":
                        logger.warning(f"Gemini MAX_TOKENS: response truncated, no content returned")
                    else:
                        logger.warning(f"Gemini response missing expected fields: {e} (finishReason: {finish_reason})")
                    logger.debug(f"Response data: {data}")
                    return self._mock_vote_response() if vote_mode else self._mock_response()

            except requests.exceptions.HTTPError as e:
                # Check if it's a rate limit error (429)
                if response.status_code == 429 and attempt < max_retries - 1:
                    # Set rate limited flag - future calls will use longer interval
                    EngineD_Gemini._rate_limited = True
                    delay = base_delay * (2 ** attempt)  # Exponential: 3s, 6s, 12s
                    logger.warning(f"Gemini rate limit (429) - retry {attempt + 1}/{max_retries} after {delay}s (increasing interval to {self.RATE_LIMITED_INTERVAL}s)")
                    time.sleep(delay)
                    continue  # Retry
                else:
                    logger.error(f"Gemini API HTTP error: {e}")
                    return self._mock_vote_response() if vote_mode else self._mock_response()

            except requests.exceptions.RequestException as e:
                logger.error(f"Gemini API error: {e}")
                # FIX BUG #3: Use correct mock response based on mode
                return self._mock_vote_response() if vote_mode else self._mock_response()

        # If all retries exhausted
        logger.error("Gemini API: All retries exhausted")
        return self._mock_vote_response() if vote_mode else self._mock_response()

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

    def _mock_vote_response(self) -> str:
        """Mock response for voting. FIX BUG #3: Separate vote response."""
        return """{
  "vote": "HOLD",
  "confidence": 0.65,
  "reasoning": "Mock vote response - API unavailable. Conservative HOLD to avoid bad trades.",
  "concerns": ["API rate limit", "Insufficient data for confident vote"]
}"""
