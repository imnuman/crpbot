"""
Mother AI (Gemini) - Layer 1
Orchestration & Risk Governance

Responsibilities:
1. Orchestrate all 3 specialist agents
2. Calculate lot size (1.0% risk per trade)
3. Perform final FTMO compliance check
4. Synthesize final signal from all inputs
5. Make final GO/NO-GO decision
"""
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from libs.hmas.agents.base_agent import BaseAgent
from libs.hmas.clients.gemini_client import GeminiClient


class MotherAI(BaseAgent):
    """
    Mother AI - Orchestration & Risk Governance (Layer 1)

    Uses Google Gemini 2.0 Flash for fast, cost-effective orchestration.
    """

    def __init__(self, api_key: str):
        super().__init__(name="Mother AI (Gemini)", api_key=api_key)
        self.client = GeminiClient(api_key=api_key)

    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrate trade signal generation

        Args:
            data: Contains all agent inputs and market data
                {
                    'symbol': 'GBPUSD',
                    'alpha_hypothesis': {...},  # From DeepSeek
                    'execution_audit': {...},    # From Grok
                    'rationale': str,            # From Claude
                    'account_balance': float,
                    'current_price': float
                }

        Returns:
            Final trade signal or rejection
        """
        # Extract inputs
        symbol = data['symbol']
        alpha_hypothesis = data.get('alpha_hypothesis', {})
        execution_audit = data.get('execution_audit', {})
        rationale = data.get('rationale', '')
        account_balance = data.get('account_balance', 10000)

        # Build orchestration prompt
        prompt = self._build_orchestration_prompt(
            symbol=symbol,
            alpha_hypothesis=alpha_hypothesis,
            execution_audit=execution_audit,
            rationale=rationale,
            account_balance=account_balance
        )

        system_instruction = """You are Mother AI, the final decision-maker in a hierarchical trading system.

Your role:
1. Review inputs from 3 specialist agents (Alpha Generator, Execution Auditor, Rationale Agent)
2. Calculate exact lot size for 1.0% account risk
3. Verify FTMO compliance (4.5% daily limit, 9% max limit)
4. Make final GO/NO-GO decision

You MUST output ONLY valid JSON in this exact format:
{
  "decision": "APPROVED" or "REJECTED",
  "action": "BUY_STOP" or "SELL_STOP" or "HOLD",
  "entry": 1.25500,
  "stop_loss": 1.25650,
  "take_profit": 1.25100,
  "lot_size": 0.67,
  "risk_percent": 1.0,
  "reward_risk_ratio": 2.67,
  "ftmo_compliant": true,
  "rejection_reason": "string or null",
  "confidence": 0.85
}

Never include explanatory text - ONLY the JSON object."""

        try:
            # Call Gemini API
            response = await self.client.chat(
                messages=[{"role": "user", "content": prompt}],
                system_instruction=system_instruction,
                temperature=0.3,  # Low temperature for consistent decisions
                max_tokens=1000
            )

            # Parse JSON response
            import json
            result = json.loads(response)

            # Add metadata
            result['timestamp'] = datetime.now(timezone.utc).isoformat()
            result['agent'] = 'Mother AI (Gemini)'
            result['symbol'] = symbol

            return result

        except Exception as e:
            # Return error result
            return {
                'decision': 'REJECTED',
                'action': 'HOLD',
                'rejection_reason': f'Mother AI error: {str(e)}',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

    def _build_orchestration_prompt(
        self,
        symbol: str,
        alpha_hypothesis: Dict[str, Any],
        execution_audit: Dict[str, Any],
        rationale: str,
        account_balance: float
    ) -> str:
        """Build prompt for Mother AI orchestration"""

        prompt = f"""# TRADE SIGNAL ORCHESTRATION

## Symbol
{symbol}

## Agent Inputs

### 1. Alpha Generator (DeepSeek)
Trade Hypothesis:
- Action: {alpha_hypothesis.get('action', 'N/A')}
- Entry: {alpha_hypothesis.get('entry', 'N/A')}
- Stop Loss: {alpha_hypothesis.get('sl', 'N/A')}
- Take Profit: {alpha_hypothesis.get('tp', 'N/A')}
- Confidence: {alpha_hypothesis.get('confidence', 'N/A')}
- Setup Type: {alpha_hypothesis.get('setup_type', 'N/A')}
- Evidence: {alpha_hypothesis.get('evidence', 'N/A')}

### 2. Execution Auditor (Grok)
Pre-flight Check:
- Audit Result: {execution_audit.get('audit_result', 'N/A')}
- Cost Check: {execution_audit.get('cost_check', {}).get('status', 'N/A')}
- Spread: {execution_audit.get('cost_check', {}).get('spread_pips', 'N/A')} pips
- Cost/TP Ratio: {execution_audit.get('cost_check', {}).get('cost_to_tp_ratio', 'N/A')}
- ALM Active: {execution_audit.get('alm_setup', {}).get('active', 'N/A')}
- Recommendation: {execution_audit.get('recommendation', 'N/A')}

### 3. Rationale Agent (Claude)
{rationale}

## Risk Parameters
- Account Balance: ${account_balance:,.2f}
- Target Risk: 1.0% per trade
- FTMO Daily Loss Limit: 4.5%
- FTMO Max Loss Limit: 9.0%

## Your Task
1. Review all 3 agent inputs
2. Calculate exact lot size for 1.0% risk:
   - Risk Amount = ${account_balance:,.2f} × 0.01 = ${account_balance * 0.01:,.2f}
   - Stop Loss pips = (entry - SL) in pips
   - Lot Size = Risk Amount / (SL pips × pip value)
3. Verify FTMO compliance
4. Make final decision: APPROVED or REJECTED

Output ONLY the JSON object, no explanatory text."""

        return prompt

    async def calculate_lot_size(
        self,
        account_balance: float,
        risk_percent: float,
        entry_price: float,
        stop_loss: float,
        pip_value: float = 10.0  # Standard for 1 lot forex
    ) -> float:
        """
        Calculate lot size for exact risk percentage

        Args:
            account_balance: Account size in USD
            risk_percent: Risk as decimal (0.01 = 1.0%)
            entry_price: Entry price
            stop_loss: Stop loss price
            pip_value: Value per pip for 1 standard lot (default $10)

        Returns:
            Lot size (e.g., 0.67 lots)
        """
        # Risk amount in USD
        risk_amount = account_balance * risk_percent

        # Stop loss in pips (assuming 4 decimal places for forex)
        sl_pips = abs(entry_price - stop_loss) * 10000

        # Lot size calculation
        # Risk Amount = SL pips × pip value × lot size
        # Therefore: lot size = Risk Amount / (SL pips × pip value)
        lot_size = risk_amount / (sl_pips * pip_value)

        # Round to 2 decimals
        return round(lot_size, 2)

    def __repr__(self) -> str:
        return f"MotherAI(client={self.client})"
