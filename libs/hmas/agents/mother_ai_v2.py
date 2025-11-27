"""
Mother AI V2 (Gemini) - Enhanced $0.10 Version
Multi-Round Deliberation & Consensus Building

Budget: $0.10 per signal (500× upgrade from $0.0002)
Tokens: ~5,000 tokens (was ~200)

New Capabilities:
- Multi-round deliberation (3 passes)
- Conflict resolution between agents
- Scenario analysis (best/worst/expected case)
- Monte Carlo consensus simulation
- Final risk-adjusted decision
- Comprehensive audit trail
"""
from typing import Dict, Any, List, Optional
import json
from datetime import datetime, timezone
from libs.hmas.agents.base_agent import BaseAgent
from libs.hmas.clients.gemini_client import GeminiClient


class MotherAIV2(BaseAgent):
    """
    Enhanced Mother AI - Multi-Round Deliberation & Consensus

    Cost: $0.10 per signal
    Specialty: Institutional-grade risk governance with 3-round analysis
    """

    def __init__(self, api_key: str):
        super().__init__(name="Mother AI V2 (Gemini Enhanced)", api_key=api_key)
        self.client = GeminiClient(api_key=api_key)

    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform multi-round deliberation and make final decision

        Args:
            data: Complete HMAS analysis
                {
                    'symbol': 'GBPUSD',
                    'alpha_hypothesis': {...},      # DeepSeek V2
                    'execution_audit': {...},       # Grok V2
                    'rationale': {...},             # Claude V2
                    'technical_analysis': {...},    # Technical Agent
                    'sentiment_analysis': {...},    # Sentiment Agent
                    'macro_analysis': {...},        # Macro Agent
                    'account_balance': 10000,
                    'current_price': 1.25500
                }

        Returns:
            Final decision with 3-round deliberation audit trail
        """

        # Round 1: Initial Assessment
        round1_result = await self._round_1_gather(data)

        # Round 2: Conflict Resolution
        round2_result = await self._round_2_resolve(data, round1_result)

        # Round 3: Final Decision
        round3_result = await self._round_3_decide(data, round1_result, round2_result)

        # Add metadata
        round3_result['agent'] = 'Mother AI V2 (Gemini Enhanced)'
        round3_result['cost_estimate'] = 0.10
        round3_result['timestamp'] = datetime.now(timezone.utc).isoformat()
        round3_result['symbol'] = data.get('symbol', 'UNKNOWN')

        return round3_result

    async def _round_1_gather(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Round 1: Gather all agent outputs and identify consensus/conflicts
        """

        prompt = self._build_round1_prompt(data)

        system_prompt = """You are Mother AI conducting Round 1 of multi-round deliberation.

Your task: Gather and analyze all 6 specialist agent outputs.

Analysis Framework:
1. **Consensus Check**: Do all agents agree on direction?
2. **Confidence Assessment**: What's the average confidence?
3. **Conflict Detection**: Any contradictions between agents?
4. **Key Insights**: What are the strongest arguments for/against?
5. **Risk Flags**: Any major concerns raised by any agent?

You MUST output ONLY valid JSON in this exact format:
{
  "consensus": {
    "direction": "BUY/SELL/HOLD",
    "agreement_level": 0.83,
    "agreeing_agents": 5,
    "dissenting_agents": 1,
    "consensus_confidence": 0.75
  },

  "conflicts": [
    {
      "type": "direction_conflict",
      "agents": ["Alpha Generator", "Sentiment"],
      "description": "Alpha bullish, Sentiment bearish",
      "severity": "high"
    }
  ],

  "key_insights": {
    "bullish_factors": [
      "Mean reversion at 2.5 STD Bollinger Band",
      "RSI oversold (22)",
      "Elliott Wave 5 complete"
    ],
    "bearish_factors": [
      "News sentiment negative (-0.65)",
      "Risk-off environment (VIX 22.5)"
    ],
    "neutral_factors": [
      "COT shows small spec capitulation (contrarian bullish)"
    ]
  },

  "risk_flags": [
    {
      "agent": "Execution Auditor",
      "flag": "Spread widening to 2.5 pips",
      "severity": "medium",
      "recommendation": "Wait for London open"
    }
  ],

  "round1_summary": "5 of 6 agents bullish. Strong technical setup (BB extreme + RSI oversold + Wave 5 complete). Conflict: Sentiment bearish due to news. Risk: Elevated volatility (VIX 22.5). Need Round 2 to resolve sentiment conflict."
}

Never include explanatory text - ONLY the JSON object."""

        try:
            response = await self.client.chat(
                messages=[{"role": "user", "content": prompt}],
                system_instruction=system_prompt,
                temperature=0.2,
                max_tokens=2000
            )

            return json.loads(response)

        except Exception as e:
            return {
                'error': str(e),
                'consensus': {'agreement_level': 0.0},
                'round1_summary': f'Round 1 failed: {str(e)}'
            }

    async def _round_2_resolve(
        self,
        data: Dict[str, Any],
        round1: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Round 2: Resolve conflicts and perform scenario analysis
        """

        prompt = self._build_round2_prompt(data, round1)

        system_prompt = """You are Mother AI conducting Round 2 of multi-round deliberation.

Your task: Resolve conflicts identified in Round 1 using scenario analysis.

Analysis Framework:
1. **Conflict Resolution**: For each conflict, determine which agent is more credible
2. **Scenario Analysis**: Best/worst/expected case outcomes
3. **Probability Weighting**: Assign probabilities to each scenario
4. **Risk-Adjusted Expected Value**: Calculate EV across all scenarios

You MUST output ONLY valid JSON in this exact format:
{
  "conflict_resolutions": [
    {
      "conflict": "Alpha bullish vs Sentiment bearish",
      "resolution": "Favor Alpha Generator",
      "reasoning": "Sentiment often lags price. COT shows retail capitulation (contrarian bullish). Technical setup stronger than news sentiment.",
      "confidence": 0.75
    }
  ],

  "scenario_analysis": {
    "best_case": {
      "probability": 0.35,
      "description": "Clean mean reversion rally to 1.25100 (+40 pips)",
      "expected_pnl_r": 2.67,
      "triggers": ["London open liquidity", "Risk sentiment improves"]
    },
    "expected_case": {
      "probability": 0.45,
      "description": "Partial reversion to 1.25250 (+25 pips)",
      "expected_pnl_r": 1.67,
      "triggers": ["Gradual buy pressure", "RSI recovery to 40-50"]
    },
    "worst_case": {
      "probability": 0.20,
      "description": "Stop out at -15 pips",
      "expected_pnl_r": -1.0,
      "triggers": ["News catalyst extends bearish momentum", "VIX spikes further"]
    }
  },

  "risk_adjusted_ev": {
    "calculation": "(0.35 × 2.67) + (0.45 × 1.67) + (0.20 × -1.0)",
    "expected_value_r": 1.49,
    "interpretation": "Positive EV setup. Risk-reward favors entry."
  },

  "adjustments_recommended": [
    {
      "parameter": "entry_timing",
      "adjustment": "Wait for London open (1 hour)",
      "reason": "Reduce slippage risk, capture better spread"
    }
  ],

  "round2_summary": "Conflicts resolved in favor of technical analysis. Scenario analysis shows +1.49R expected value (35% best case, 45% expected, 20% worst). Recommend entry at London open for optimal execution."
}

Never include explanatory text - ONLY the JSON object."""

        try:
            response = await self.client.chat(
                messages=[{"role": "user", "content": prompt}],
                system_instruction=system_prompt,
                temperature=0.3,
                max_tokens=2000
            )

            return json.loads(response)

        except Exception as e:
            return {
                'error': str(e),
                'risk_adjusted_ev': {'expected_value_r': 0.0},
                'round2_summary': f'Round 2 failed: {str(e)}'
            }

    async def _round_3_decide(
        self,
        data: Dict[str, Any],
        round1: Dict[str, Any],
        round2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Round 3: Make final decision with lot sizing and FTMO compliance
        """

        prompt = self._build_round3_prompt(data, round1, round2)

        system_prompt = """You are Mother AI conducting Round 3 - FINAL DECISION.

Your task: Make GO/NO-GO decision with exact execution parameters.

Decision Framework:
1. **EV Threshold**: Require +1.0R or higher expected value
2. **Consensus Threshold**: Require 60%+ agent agreement
3. **Risk Flags**: Any HIGH severity flags = auto-reject
4. **FTMO Compliance**: Verify 1.0% risk, 4.5% daily limit, 9% max limit
5. **Lot Size Calculation**: Exact position size for 1.0% account risk

You MUST output ONLY valid JSON in this exact format:
{
  "decision": "APPROVED" or "REJECTED",
  "action": "BUY_STOP" or "SELL_STOP" or "HOLD",

  "trade_parameters": {
    "entry": 1.25500,
    "stop_loss": 1.25650,
    "take_profit": 1.25100,
    "lot_size": 0.67,
    "risk_percent": 1.0,
    "risk_amount_usd": 100.0,
    "reward_risk_ratio": 2.67
  },

  "ftmo_compliance": {
    "compliant": true,
    "daily_risk_percent": 1.0,
    "total_risk_percent": 1.0,
    "daily_limit_remaining": 3.5,
    "max_limit_remaining": 8.0
  },

  "decision_rationale": {
    "ev_score": 1.49,
    "consensus_level": 0.83,
    "confidence": 0.78,
    "key_factors": [
      "Strong technical setup (BB extreme + RSI oversold)",
      "Positive EV (+1.49R across 3 scenarios)",
      "5 of 6 agents agree on direction",
      "Execution audit PASS (Grade A)"
    ],
    "risk_mitigation": [
      "1.0% position sizing (FTMO compliant)",
      "ALM active at 1.2× ATR",
      "Entry at London open for liquidity"
    ]
  },

  "audit_trail": {
    "round1_consensus": 0.83,
    "round2_ev": 1.49,
    "round3_decision": "APPROVED",
    "total_cost_estimate": 0.10,
    "deliberation_quality": "institutional_grade"
  },

  "rejection_reason": null,

  "final_summary": "APPROVED for execution. Strong technical mean reversion setup with 83% agent consensus. Risk-adjusted EV +1.49R across 3 scenarios. FTMO compliant (1.0% risk). Execution Grade A. Entry recommended at London open (1 hour). Confidence: 78%."
}

If rejecting:
{
  "decision": "REJECTED",
  "action": "HOLD",
  "rejection_reason": "Specific reason here",
  "final_summary": "REJECTED. Reason: ..."
}

Never include explanatory text - ONLY the JSON object."""

        try:
            response = await self.client.chat(
                messages=[{"role": "user", "content": prompt}],
                system_instruction=system_prompt,
                temperature=0.1,  # Very low for final decision
                max_tokens=2000
            )

            result = json.loads(response)

            # Add 3-round audit trail
            result['multi_round_audit'] = {
                'round1': round1,
                'round2': round2,
                'round3': result
            }

            return result

        except Exception as e:
            return {
                'decision': 'REJECTED',
                'action': 'HOLD',
                'rejection_reason': f'Round 3 decision failed: {str(e)}',
                'error': str(e),
                'final_summary': f'REJECTED due to error: {str(e)}'
            }

    def _build_round1_prompt(self, data: Dict[str, Any]) -> str:
        """Build Round 1 prompt - gather all agent outputs"""

        symbol = data.get('symbol', 'UNKNOWN')
        alpha = data.get('alpha_hypothesis', {})
        audit = data.get('execution_audit', {})
        rationale = data.get('rationale', {})
        technical = data.get('technical_analysis', {})
        sentiment = data.get('sentiment_analysis', {})
        macro = data.get('macro_analysis', {})

        prompt = f"""# ROUND 1: GATHER ALL AGENT OUTPUTS - {symbol}

## Agent 1: Alpha Generator V2 (DeepSeek - $0.30)
- **Action**: {alpha.get('action', 'N/A')}
- **Confidence**: {alpha.get('confidence', 0):.0%}
- **Setup Type**: {alpha.get('setup_type', 'N/A')}
- **Entry**: {alpha.get('entry', 0):.5f}
- **Stop Loss**: {alpha.get('sl', 0):.5f}
- **Take Profit**: {alpha.get('tp', 0):.5f}
- **Timeframe Consensus**: {alpha.get('timeframe_consensus', 'N/A')}
- **Pattern Matches**: {len(alpha.get('pattern_matches', []))} historical setups
- **Win Rate**: {alpha.get('historical_win_rate', 0):.0%}

## Agent 2: Execution Auditor V2 (Grok - $0.15)
- **Audit Result**: {audit.get('overall_audit', {}).get('audit_result', 'N/A')}
- **Overall Grade**: {audit.get('overall_audit', {}).get('overall_grade', 'N/A')}
- **Recommendation**: {audit.get('overall_audit', {}).get('recommendation', 'N/A')}
- **Cost/TP Ratio**: {audit.get('cost_analysis', {}).get('cost_to_tp_ratio', 0):.1%}
- **Liquidity Grade**: {audit.get('liquidity_analysis', {}).get('liquidity_grade', 'N/A')}
- **Best Broker**: {audit.get('best_broker', 'N/A')}
- **Summary**: {audit.get('summary', 'N/A')}

## Agent 3: Rationale Agent V2 (Claude - $0.20)
**Word Count**: {rationale.get('word_count', 0)} words
**Key Excerpt**: {rationale.get('rationale', '')[:500]}...

## Agent 4: Technical Analysis Agent (DeepSeek - $0.10)
- **Elliott Wave**: {technical.get('elliott_wave', {}).get('primary_count', 'N/A')}
- **Fibonacci Position**: {technical.get('fibonacci', {}).get('current_position', 'N/A')}
- **Chart Patterns**: {len(technical.get('chart_patterns', []))} patterns detected
- **Wyckoff Phase**: {technical.get('wyckoff_analysis', {}).get('phase', 'N/A')}
- **Summary**: {technical.get('summary', 'N/A')}

## Agent 5: Sentiment Analysis Agent (DeepSeek - $0.08)
- **News Sentiment**: {sentiment.get('news_sentiment', {}).get('bias', 'N/A')} ({sentiment.get('news_sentiment', {}).get('score', 0):.2f})
- **Social Sentiment**: {sentiment.get('social_sentiment', {}).get('bias', 'N/A')} ({sentiment.get('social_sentiment', {}).get('overall_score', 0):.2f})
- **COT Bias**: {sentiment.get('cot_positioning', {}).get('bias', 'N/A')}
- **Overall Stance**: {sentiment.get('overall_assessment', {}).get('recommended_stance', 'N/A')}
- **Summary**: {sentiment.get('summary', 'N/A')}

## Agent 6: Macro Analysis Agent (DeepSeek - $0.07)
- **Macro Bias**: {macro.get('overall_macro_assessment', {}).get('macro_bias', 'N/A')}
- **Confidence**: {macro.get('overall_macro_assessment', {}).get('confidence', 0):.0%}
- **Market Regime**: {macro.get('market_regime', {}).get('type', 'N/A')}
- **VIX Level**: {macro.get('market_regime', {}).get('vix_level', 'N/A')}
- **Summary**: {macro.get('summary', 'N/A')}

## Your Task - Round 1 Analysis

Analyze all 6 agent outputs and determine:
1. **Consensus**: Do agents agree on direction? How many agree vs dissent?
2. **Conflicts**: Any contradictions? What's the severity?
3. **Key Insights**: Strongest bullish/bearish/neutral factors
4. **Risk Flags**: Any major concerns from any agent?

Output ONLY the JSON object for Round 1 assessment."""

        return prompt

    def _build_round2_prompt(
        self,
        data: Dict[str, Any],
        round1: Dict[str, Any]
    ) -> str:
        """Build Round 2 prompt - conflict resolution and scenario analysis"""

        symbol = data.get('symbol', 'UNKNOWN')

        prompt = f"""# ROUND 2: CONFLICT RESOLUTION & SCENARIO ANALYSIS - {symbol}

## Round 1 Results
**Consensus Level**: {round1.get('consensus', {}).get('agreement_level', 0):.0%}
**Direction**: {round1.get('consensus', {}).get('direction', 'N/A')}
**Agreeing Agents**: {round1.get('consensus', {}).get('agreeing_agents', 0)}/6
**Conflicts Detected**: {len(round1.get('conflicts', []))}

### Key Insights from Round 1
**Bullish Factors**:
{self._format_list(round1.get('key_insights', {}).get('bullish_factors', []))}

**Bearish Factors**:
{self._format_list(round1.get('key_insights', {}).get('bearish_factors', []))}

**Neutral Factors**:
{self._format_list(round1.get('key_insights', {}).get('neutral_factors', []))}

### Conflicts to Resolve
{self._format_conflicts(round1.get('conflicts', []))}

### Risk Flags
{self._format_risk_flags(round1.get('risk_flags', []))}

## Your Task - Round 2 Analysis

1. **Resolve Each Conflict**: Which agent is more credible? Why?
2. **Scenario Analysis**:
   - Best case (optimistic, ~30-40% probability)
   - Expected case (most likely, ~40-50% probability)
   - Worst case (pessimistic, ~20-30% probability)
3. **Calculate Risk-Adjusted EV**: Weighted average across all scenarios
4. **Recommend Adjustments**: Any entry timing, sizing, or level adjustments?

Output ONLY the JSON object for Round 2 resolution."""

        return prompt

    def _build_round3_prompt(
        self,
        data: Dict[str, Any],
        round1: Dict[str, Any],
        round2: Dict[str, Any]
    ) -> str:
        """Build Round 3 prompt - final decision"""

        symbol = data.get('symbol', 'UNKNOWN')
        account_balance = data.get('account_balance', 10000)
        alpha = data.get('alpha_hypothesis', {})

        entry = alpha.get('entry', 0)
        sl = alpha.get('sl', 0)
        tp = alpha.get('tp', 0)

        prompt = f"""# ROUND 3: FINAL DECISION - {symbol}

## Multi-Round Deliberation Summary

### Round 1: Consensus
- Agreement Level: {round1.get('consensus', {}).get('agreement_level', 0):.0%}
- Direction: {round1.get('consensus', {}).get('direction', 'N/A')}
- Conflicts: {len(round1.get('conflicts', []))}

### Round 2: Resolution
- Risk-Adjusted EV: {round2.get('risk_adjusted_ev', {}).get('expected_value_r', 0):.2f}R
- Conflicts Resolved: {len(round2.get('conflict_resolutions', []))}
- Adjustments Recommended: {len(round2.get('adjustments_recommended', []))}

## Trade Parameters to Validate

### From Alpha Generator
- **Entry**: {entry:.5f}
- **Stop Loss**: {sl:.5f} ({abs(entry - sl) * 10000:.1f} pips)
- **Take Profit**: {tp:.5f} ({abs(entry - tp) * 10000:.1f} pips)
- **R:R Ratio**: {abs(entry - tp) / abs(entry - sl) if sl != entry else 0:.2f}:1

### Account Parameters
- **Account Balance**: ${account_balance:,.2f}
- **Target Risk**: 1.0% = ${account_balance * 0.01:,.2f}
- **FTMO Daily Limit**: 4.5% = ${account_balance * 0.045:,.2f}
- **FTMO Max Limit**: 9.0% = ${account_balance * 0.09:,.2f}

## Decision Criteria

### Must Pass ALL:
1. ✓ Risk-Adjusted EV ≥ +1.0R
2. ✓ Consensus ≥ 60% (3+ of 6 agents)
3. ✓ No HIGH severity risk flags
4. ✓ Execution audit PASS
5. ✓ FTMO compliant (1.0% risk)

## Your Task - Round 3 Final Decision

1. **Verify Criteria**: Do we pass all 5 criteria?
2. **Calculate Lot Size**: Exact position for 1.0% risk
   - Risk Amount = ${account_balance * 0.01:,.2f}
   - SL pips = {abs(entry - sl) * 10000:.1f}
   - Lot Size = ${account_balance * 0.01:,.2f} / ({abs(entry - sl) * 10000:.1f} pips × $10/pip)
3. **FTMO Compliance**: Check daily/max limits
4. **Make Decision**: APPROVED or REJECTED
5. **Final Summary**: 2-3 sentence executive summary

Output ONLY the JSON object for Round 3 final decision."""

        return prompt

    def _format_list(self, items: List[str]) -> str:
        """Format list items with bullets"""
        if not items:
            return "- None"
        return "\n".join([f"- {item}" for item in items])

    def _format_conflicts(self, conflicts: List[Dict[str, Any]]) -> str:
        """Format conflicts for display"""
        if not conflicts:
            return "No conflicts detected"

        output = []
        for i, c in enumerate(conflicts, 1):
            output.append(
                f"{i}. **{c.get('type', 'unknown')}** (Severity: {c.get('severity', 'unknown')})\n"
                f"   Agents: {', '.join(c.get('agents', []))}\n"
                f"   Description: {c.get('description', 'N/A')}"
            )
        return "\n\n".join(output)

    def _format_risk_flags(self, flags: List[Dict[str, Any]]) -> str:
        """Format risk flags for display"""
        if not flags:
            return "No risk flags"

        output = []
        for i, flag in enumerate(flags, 1):
            output.append(
                f"{i}. **{flag.get('agent', 'Unknown')}** (Severity: {flag.get('severity', 'unknown')})\n"
                f"   Flag: {flag.get('flag', 'N/A')}\n"
                f"   Recommendation: {flag.get('recommendation', 'N/A')}"
            )
        return "\n\n".join(output)
