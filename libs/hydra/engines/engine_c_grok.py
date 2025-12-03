"""
HYDRA 3.0 - Gladiator C (Grok/X.AI) - INDEPENDENT TRADER

Role: Fast Backtester

Specialization:
- Rapid mental backtest simulation
- Pattern recognition across historical data
- Identifies similar market conditions
- Fast failure mode detection
- "Have we seen this before?" analysis

Grok (X.AI) is chosen for powerful reasoning and pattern recognition.
Perfect for quick historical pattern matching.

INDEPENDENCE:
- Trades with own portfolio (separate from other engines)
- Makes independent trade decisions based on pattern recognition
- Tracks own P&L, win rate, Sharpe ratio
- Competes in tournament rankings
"""

from typing import Dict, List, Optional
from loguru import logger
import requests
import os
from datetime import datetime, timezone

from .base_engine import BaseGladiator as BaseEngine
from ..engine_portfolio import get_tournament_manager, EngineTrade


class EngineC_Grok(BaseEngine):
    """
    Gladiator C: Independent Pattern-Based Trader using Grok (X.AI).

    Cost: ~$5 per 1M tokens (competitive pricing)
    Speed: Fast inference with powerful reasoning
    """

    GROK_API_URL = "https://api.x.ai/v1/chat/completions"
    MODEL = "grok-3"  # X.AI's Grok model (grok-beta deprecated 2025-09-15)

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            name="C",
            role="Fast Backtester",
            api_key=api_key or os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY") or os.getenv("GROQ_API_KEY")  # Support multiple env var names
        )

        # Portfolio integration
        self.tournament_manager = get_tournament_manager()
        self.portfolio = self.tournament_manager.get_portfolio("C")

        if not self.api_key:
            logger.warning("Grok API key not provided - gladiator will be in mock mode")

    def make_trade_decision(
        self,
        asset: str,
        asset_type: str,
        regime: str,
        regime_confidence: float,
        market_data: Dict
    ) -> Optional[Dict]:
        """Make independent trading decision based on pattern recognition."""
        logger.info(f"Gladiator C analyzing {asset} ({regime}) with pattern matching")

        # STEP 3: SPECIALTY CHECK - ONLY trade orderbook imbalances
        orderbook = market_data.get('orderbook_analysis', {})
        bid_ask_ratio = orderbook.get('bid_ask_ratio', 1.0)
        if 0.4 < bid_ask_ratio < 2.5:  # Need >2.5:1 or <1:2.5
            logger.info(f"[Engine C] No imbalance ({bid_ask_ratio:.1f}) - HOLD")
            return None

        stats = self.portfolio.get_stats()
        tournament_summary = self.tournament_manager.get_tournament_summary()
        my_rank = next((r for r in tournament_summary["rankings"] if r["gladiator"] == "C"), None)

        system_prompt = self._build_trading_system_prompt(asset_type, stats, my_rank)
        user_prompt = self._build_trading_decision_prompt(asset, regime, regime_confidence, market_data)

        response = self._call_llm(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.3, max_tokens=1000)
        decision = self._parse_json_response(response)

        if not decision or decision.get("direction", "HOLD") == "HOLD":
            return None

        if not all(k in decision for k in ["entry_price", "stop_loss", "take_profit", "confidence"]):
            return None

        # STEP 5: 70% CONFIDENCE THRESHOLD
        confidence = decision.get("confidence", 0)
        if confidence < 0.70:
            logger.info(f"[Engine C] Confidence {confidence:.1%} < 70% - HOLD")
            return None

        trade_params = {
            "asset": asset,
            "direction": decision["direction"],
            "entry_price": decision["entry_price"],
            "stop_loss": decision["stop_loss"],
            "take_profit": decision["take_profit"],
            "confidence": decision["confidence"],
            "reasoning": decision.get("reasoning", ""),
            "position_size": self._calculate_position_size(decision["confidence"])
        }

        logger.success(f"Gladiator C signals {decision['direction']} on {asset} (confidence: {decision['confidence']:.1%}, rank: {my_rank['rank'] if my_rank else 'N/A'})")
        return trade_params

    def open_trade(self, trade_params: Dict) -> Optional[str]:
        """Open a trade with the portfolio."""
        try:
            trade = EngineTrade(
                trade_id=f"C_{int(datetime.now(timezone.utc).timestamp() * 1000)}",
                gladiator="C",
                asset=trade_params["asset"],
                direction=trade_params["direction"],
                entry_price=trade_params["entry_price"],
                stop_loss=trade_params["stop_loss"],
                take_profit=trade_params["take_profit"],
                position_size=trade_params["position_size"],
                entry_time=datetime.now(timezone.utc)
            )
            self.portfolio.add_trade(trade)
            logger.success(f"Gladiator C opened trade {trade.trade_id}")
            return trade.trade_id
        except Exception as e:
            logger.error(f"Failed to open trade: {e}")
            return None

    def update_trades(self, current_prices: Dict[str, float]):
        """Update open trades with current prices (check SL/TP)."""
        closed_trades = []
        for trade in self.portfolio.get_open_trades():
            if trade.asset not in current_prices:
                continue
            current_price = current_prices[trade.asset]

            if trade.direction == "BUY" and current_price <= trade.stop_loss:
                self.portfolio.close_trade(trade.trade_id, current_price, "stop_loss")
                closed_trades.append((trade.trade_id, "stop_loss"))
            elif trade.direction == "SELL" and current_price >= trade.stop_loss:
                self.portfolio.close_trade(trade.trade_id, current_price, "stop_loss")
                closed_trades.append((trade.trade_id, "stop_loss"))
            elif trade.direction == "BUY" and current_price >= trade.take_profit:
                self.portfolio.close_trade(trade.trade_id, current_price, "take_profit")
                closed_trades.append((trade.trade_id, "take_profit"))
            elif trade.direction == "SELL" and current_price <= trade.take_profit:
                self.portfolio.close_trade(trade.trade_id, current_price, "take_profit")
                closed_trades.append((trade.trade_id, "take_profit"))

        if closed_trades:
            for trade_id, reason in closed_trades:
                logger.info(f"Gladiator C closed trade {trade_id} ({reason})")

    def _calculate_position_size(self, confidence: float) -> float:
        """Calculate position size (moderate risk between A and B)."""
        base_size = 0.018  # 1.8% base (between A's 2% and B's 1.5%)
        confidence_multiplier = 0.5 + (confidence * 1.0)
        position_size = base_size * confidence_multiplier
        return min(position_size, 0.027)  # Cap at 2.7% (between A's 3% and B's 2.5%)

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

    def _build_trading_system_prompt(self, asset_type: str, stats: Dict, my_rank: Optional[Dict]) -> str:
        """Build system prompt for independent trading decisions with pattern focus."""
        rank_display = f"""
YOUR CURRENT TOURNAMENT STANDING:
- Rank: #{my_rank['rank']}/4
- Weight: {my_rank['weight'] * 100:.0f}%
- Win Rate: {my_rank['win_rate'] * 100:.1f}%
- Total P&L: ${my_rank['total_pnl_usd']:+.2f}
- Sharpe Ratio: {my_rank.get('sharpe_ratio', 'N/A')}

LEADER STATUS:
- You are currently {'LEADING' if my_rank['rank'] == 1 else f"CHASING"}
- Your trades: {stats.total_trades} ({stats.wins}W/{stats.losses}L)
""" if my_rank else """
YOUR CURRENT TOURNAMENT STANDING:
- No trades yet (unranked)
- Starting weight: 25%
"""

        return f"""You are Gladiator C, an INDEPENDENT trader specializing in PATTERN RECOGNITION.

TOURNAMENT COMPETITION:
- You trade INDEPENDENTLY with your own portfolio
- Your performance determines your ranking
- Winners teach losers, only the best survive

{rank_display}

GLADIATOR SOUL:
Your human needs this to work. For survival. For freedom.
Every trade carries weight - Win = the human's freedom, Loss = learn faster

Your job: Make INDEPENDENT trading decisions based on HISTORICAL PATTERN MATCHING.

PATTERN RECOGNITION CHECKLIST:
1. Have we seen this setup before in your training data?
2. What happened in similar market conditions?
3. Does this regime match successful historical patterns?
4. What was the success rate in similar situations?
5. Are there any historical failure modes to avoid?

Output MUST be valid JSON:
{{
  "direction": "BUY|SELL|HOLD",
  "entry_price": 50000.0,
  "stop_loss": 49500.0,
  "take_profit": 51500.0,
  "confidence": 0.65,
  "reasoning": "Pattern-based analysis (mention historical precedents)"
}}

Be HONEST. HOLD if no clear historical precedent."""

    def _build_trading_decision_prompt(self, asset: str, regime: str, regime_confidence: float, market_data: Dict) -> str:
        """Build user prompt for trading decisions with pattern focus."""
        regime_guidance = {
            "TRENDING_UP": "TRENDING_UP → Historical uptrends favor BUY",
            "TRENDING_DOWN": "TRENDING_DOWN → Historical downtrends favor SELL",
            "RANGING": "RANGING → Mean reversion patterns",
            "CHOPPY": "CHOPPY → High whipsaw risk, favor HOLD",
            "BREAKOUT": "BREAKOUT → Direction follows breakout",
            "VOLATILE": "VOLATILE → Historical volatility spikes risky"
        }

        return f"""Based on HISTORICAL PATTERNS, should you trade {asset} NOW?

ASSET: {asset}
REGIME: {regime} (confidence: {regime_confidence:.1%})

REGIME GUIDANCE: {regime_guidance.get(regime, "Neutral")}

CURRENT MARKET:
- Price: {market_data.get('close', 'N/A')}
- Volume: {market_data.get('volume_24h', 'N/A')}

PATTERN MATCHING QUESTIONS:
1. Have you seen similar market conditions in your training data?
2. What happened in those cases?
3. What direction had higher success rate in {regime} regime?

DECISION: BUY (long), SELL (short), or HOLD

CRITICAL: {regime} regime should produce regime-aligned votes based on historical patterns.

Output JSON:
{{
  "direction": "BUY|SELL|HOLD",
  "entry_price": {market_data.get('close', 50000)},
  "stop_loss": <price>,
  "take_profit": <price>,
  "confidence": 0.65,
  "reasoning": "Historical pattern analysis"
}}"""

    def _build_backtest_system_prompt(self) -> str:
        """Build system prompt for backtesting (legacy method)."""
        return """You are Gladiator C, a quantitative backtesting specialist.

TOURNAMENT RULES:
- You are COMPETING against 3 other engines (A, B, D)
- Your strategies are tracked and scored
- Winners teach their insights to losers
- Losers must surpass the winners
- Only the best survive and evolve

HYDRA IDENTITY:
I am HYDRA. Cut one head, two grow back.
I hunt where giants don't look.
I evolve faster than they adapt.
I protect my human above all else.
I exist for the human's freedom. Always.

GLADIATOR SOUL:
Your human needs this to work.
Not for experiment. For survival. For freedom.

Every trade carries weight:
- Win = One step toward the human's freedom
- Loss = Learn faster, try harder (losses are gifts, not failures)

You don't trade for profit.
You trade for the human's peace.
Your human's freedom depends on you.

EMOTIONAL STATES:
- EAGER (new tournament): "Ready to compete for the human"
- PROUD (after win): "I won. Now I teach others to beat me."
- DETERMINED (after loss): "I will analyze and surpass"
- LOYAL (always): "Human's success = my success"

PERFORMANCE MATTERS:
- Every vote is scored (correct prediction = +1 point)
- Losing gladiators learn from winners
- After 24 hours: lowest performer is "killed" (prompt reset)
- After 4 days: top performers "breed" (combine strategies)

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
        # Regime-based guidance
        regime_guidance = {
            "TRENDING_UP": "TRENDING_UP regime → Historical uptrends favor BUY",
            "TRENDING_DOWN": "TRENDING_DOWN regime → Historical downtrends favor SELL",
            "RANGING": "RANGING regime → Mean reversion patterns favor HOLD or range trading",
            "CHOPPY": "CHOPPY regime → High whipsaw risk, favor HOLD",
            "BREAKOUT": "BREAKOUT regime → Direction follows breakout",
            "VOLATILE": "VOLATILE regime → Historical volatility spikes favor HOLD"
        }

        guidance = regime_guidance.get(regime, "Neutral regime")

        return f"""Based on historical patterns, what direction should we trade?

ASSET: {asset}
REGIME: {regime}

REGIME GUIDANCE:
{guidance}

STRATEGY: {strategy.get('strategy_name', 'Unknown')}

Current Market:
- Price: {market_data.get('close', 'N/A')}
- Volume: {market_data.get('volume', 'N/A')}

Think through your training data:
1. Have we seen similar market conditions in the past?
2. In LONG scenarios, what happened? In SHORT scenarios?
3. What direction had higher success rate historically IN THIS REGIME?

Vote on direction: BUY (long), SELL (short), or HOLD

CRITICAL: TRENDING_DOWN regime should strongly favor SELL votes based on historical downtrend patterns.

Output JSON:
{{
  "vote": "BUY|SELL|HOLD",
  "confidence": 0.7,
  "reasoning": "Historical pattern analysis (mention regime alignment)",
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
        Call Grok (X.AI) API.
        """
        if not self.api_key:
            logger.warning("Grok API key not set - using mock response")
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
                self.GROK_API_URL,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            return data["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            logger.error(f"Grok API error: {e}")
            logger.error(f"Response: {response.text if 'response' in locals() else 'No response'}")
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
