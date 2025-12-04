"""
Execution Auditor (Grok) - Layer 3
Speed & Aggressive Loss Management (ALM)

Responsibilities:
1. Pre-trade cost validation (spread + fees < 0.5× TP)
2. Verify TP profit > 2× (spread + fees)
3. Setup ALM monitoring (1× ATR threshold)
4. Real-time "CLOSE NOW" emergency signal capability
"""
from typing import Dict, Any
from libs.hmas.agents.base_agent import BaseAgent
from libs.hmas.clients.xai_client import XAIClient


class ExecutionAuditor(BaseAgent):
    """
    Execution Auditor - Pre-flight Check & ALM (Layer 3)

    Uses Grok for ultra-fast cost validation and risk assessment.
    """

    def __init__(self, api_key: str):
        super().__init__(name="Execution Auditor (Grok)", api_key=api_key)
        self.client = XAIClient(api_key=api_key)

    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Audit trade for execution safety

        Args:
            data: Trade hypothesis + market conditions
                {
                    'trade_hypothesis': {...},  # From AlphaGenerator
                    'spread_pips': 1.5,
                    'fees_pips': 0.5,
                    'atr': 0.00150
                }

        Returns:
            Audit result with cost check and ALM setup
        """
        prompt = self._build_audit_prompt(data)

        system_prompt = """You are an ultra-fast execution auditor for trading.

Your role:
1. Validate trade costs (spread + fees) don't eat profits
2. Check TP is > 2× total costs
3. Setup Aggressive Loss Management (ALM) at 1× ATR
4. Approve or reject trade for execution

Cost Threshold: Total cost must be < 50% of TP distance

You MUST output ONLY valid JSON in this exact format:
{
  "audit_result": "PASS" or "FAIL",
  "cost_check": {
    "spread_pips": 1.5,
    "fees_pips": 0.5,
    "total_cost_pips": 2.0,
    "tp_pips": 40,
    "cost_to_tp_ratio": 0.05,
    "threshold": 0.50,
    "status": "PASS" or "FAIL"
  },
  "alm_setup": {
    "active": true,
    "atr_value": 0.00150,
    "atr_threshold": "1x ATR",
    "emergency_close_level": 1.25650,
    "monitoring_interval": "1 second"
  },
  "recommendation": "APPROVED_FOR_EXECUTION" or "REJECTED_HIGH_COST"
}

Never include explanatory text - ONLY the JSON object."""

        try:
            response = await self.client.analyze(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.0,  # Zero temperature for strict validation
                max_tokens=600
            )

            # Parse JSON
            import json
            result = json.loads(response)

            return result

        except Exception as e:
            return {
                'audit_result': 'FAIL',
                'error': str(e),
                'recommendation': 'REJECTED_ERROR'
            }

    def _build_audit_prompt(self, data: Dict[str, Any]) -> str:
        """Build prompt for execution audit"""

        hypothesis = data.get('trade_hypothesis', {})
        spread_pips = data.get('spread_pips', 2.0)
        fees_pips = data.get('fees_pips', 0.5)
        atr = data.get('atr', 0.001)

        entry = hypothesis.get('entry', 0)
        sl = hypothesis.get('sl', 0)
        tp = hypothesis.get('tp', 0)
        action = hypothesis.get('action', 'HOLD')

        # Calculate distances in pips
        sl_pips = abs(entry - sl) * 10000
        tp_pips = abs(entry - tp) * 10000
        total_cost_pips = spread_pips + fees_pips

        prompt = f"""# EXECUTION AUDIT

## Trade Hypothesis
- Action: {action}
- Entry: {entry:.5f}
- Stop Loss: {sl:.5f} ({sl_pips:.1f} pips)
- Take Profit: {tp:.5f} ({tp_pips:.1f} pips)

## Cost Analysis
- Spread: {spread_pips:.1f} pips
- Fees: {fees_pips:.1f} pips
- Total Cost: {total_cost_pips:.1f} pips
- TP Distance: {tp_pips:.1f} pips
- Cost/TP Ratio: {(total_cost_pips/tp_pips if tp_pips > 0 else 0):.2%}

## Risk Parameters
- ATR: {atr:.5f}
- ATR in pips: {atr * 10000:.1f}
- ALM Threshold: 1× ATR = {atr:.5f}

## Validation Rules
1. Total cost < 50% of TP distance
2. TP > 2× total cost
3. Spread < 3 pips (liquidity check)
4. Setup ALM emergency exit at entry ± 1× ATR

## Your Task
Validate this trade for execution:
1. Check cost/TP ratio < 0.50 (50%)
2. Verify TP profit covers 2× costs
3. Setup ALM emergency close level
4. Recommend APPROVED or REJECTED

Output ONLY the JSON object, no explanatory text."""

        return prompt
