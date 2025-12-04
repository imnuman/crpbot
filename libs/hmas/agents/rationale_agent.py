"""
Rationale Agent (Claude) - Layer 4
Explanation & Memory

Responsibilities:
1. Generate human-readable trade rationale
2. Explain the statistical basis for 80%+ WR
3. Store trade outcome for learning
4. Provide accountability & transparency
"""
from typing import Dict, Any
from libs.hmas.agents.base_agent import BaseAgent
from libs.hmas.clients.claude_client import ClaudeClient


class RationaleAgent(BaseAgent):
    """
    Rationale Agent - Explanation & Memory (Layer 4)

    Uses Claude Haiku for fast, articulate trade explanations.
    """

    def __init__(self, api_key: str):
        super().__init__(name="Rationale Agent (Claude)", api_key=api_key)
        self.client = ClaudeClient(api_key=api_key)

    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trade rationale

        Args:
            data: All trade information
                {
                    'symbol': 'GBPUSD',
                    'alpha_hypothesis': {...},
                    'execution_audit': {...},
                    'historical_performance': {...}
                }

        Returns:
            Markdown-formatted rationale
        """
        prompt = self._build_rationale_prompt(data)

        system_prompt = """You are a trading psychology expert and performance analyst.

Your role:
1. Explain WHY this trade setup has 80%+ historical win rate
2. Provide clear, confidence-building rationale
3. Detail risk management (FTMO compliance)
4. Calculate expected value (EV)

Output format: Clear markdown with these sections:
- Setup Description
- Statistical Edge
- Risk Management
- FTMO Compliance
- Conclusion

Be concise but thorough. Focus on FACTS and STATISTICS, not hype."""

        try:
            response = await self.client.analyze(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.7,  # Higher for natural language
                max_tokens=1000
            )

            return {
                'rationale': response,
                'agent': 'Rationale Agent (Claude)'
            }

        except Exception as e:
            return {
                'rationale': f"Error generating rationale: {str(e)}",
                'error': str(e)
            }

    def _build_rationale_prompt(self, data: Dict[str, Any]) -> str:
        """Build prompt for rationale generation"""

        symbol = data.get('symbol', 'UNKNOWN')
        hypothesis = data.get('alpha_hypothesis', {})
        audit = data.get('execution_audit', {})
        historical = data.get('historical_performance', {})

        action = hypothesis.get('action', 'HOLD')
        entry = hypothesis.get('entry', 0)
        sl = hypothesis.get('sl', 0)
        tp = hypothesis.get('tp', 0)
        confidence = hypothesis.get('confidence', 0)
        setup_type = hypothesis.get('setup_type', 'unknown')

        # Calculate R:R ratio
        risk_pips = abs(entry - sl) * 10000
        reward_pips = abs(entry - tp) * 10000
        rr_ratio = reward_pips / risk_pips if risk_pips > 0 else 0

        prompt = f"""# TRADE RATIONALE REQUEST

## Trade Details
- Symbol: {symbol}
- Action: {action}
- Entry: {entry:.5f}
- Stop Loss: {sl:.5f} ({risk_pips:.1f} pips)
- Take Profit: {tp:.5f} ({reward_pips:.1f} pips)
- R:R Ratio: {rr_ratio:.2f}:1
- Confidence: {confidence:.0%}
- Setup Type: {setup_type}

## Strategy Context
- Mean Reversion at Bollinger Band Extremes
- 200-MA Trend Filter
- RSI Confirmation
- Historical Win Rate Target: 80%+

## Execution Audit
- Cost Check: {audit.get('cost_check', {}).get('status', 'N/A')}
- Cost/TP Ratio: {audit.get('cost_check', {}).get('cost_to_tp_ratio', 0):.1%}
- ALM Active: {audit.get('alm_setup', {}).get('active', False)}

## Historical Performance (If Available)
{historical.get('summary', 'No historical data available')}

## Your Task
Write a clear, compelling trade rationale that explains:

1. **Setup Description**: What mean reversion pattern occurred?
2. **Statistical Edge**: Why does this setup have 80%+ win rate?
3. **Risk Management**: How is risk controlled (1.0% account risk)?
4. **FTMO Compliance**: Verify within daily 4.5% and max 9% limits
5. **Expected Value**: Calculate EV = (WR × R:R) - (1-WR × 1)
6. **Conclusion**: HIGH PROBABILITY or AVOID

Format your response in clear markdown sections."""

        return prompt
