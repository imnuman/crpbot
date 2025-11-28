"""
Execution Auditor V2 (Grok) - Enhanced $0.15 Version
Real-time Liquidity, Order Book, Slippage Analysis

Budget: $0.15 per signal (1,500× upgrade from $0.0001)
Tokens: ~7,500 tokens (was ~300)

New Capabilities:
- Real-time order book depth analysis
- Multi-broker spread comparison (5+ brokers)
- Slippage probability calculation
- Market impact estimation
- Execution timing optimization (avoid spread widening)
- ALM with dynamic thresholds (adaptive to volatility)
- Liquidity heatmap (entry, SL, TP levels)
- Fill probability assessment
"""
from typing import Dict, Any, List
import json
from libs.hmas.agents.base_agent import BaseAgent
from libs.hmas.clients.xai_client import XAIClient


class ExecutionAuditorV2(BaseAgent):
    """
    Enhanced Execution Auditor - Deep Liquidity & Cost Analysis

    Cost: $0.15 per signal
    Specialty: Ultra-fast execution validation with institutional precision
    """

    def __init__(self, api_key: str):
        super().__init__(name="Execution Auditor V2 (Grok Enhanced)", api_key=api_key)
        self.client = XAIClient(api_key=api_key)

    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive execution audit

        Args:
            data: Trade + liquidity data
                {
                    'trade_hypothesis': {...},
                    'order_book': {...},  # Depth at multiple levels
                    'broker_spreads': {...},  # 5+ brokers
                    'market_depth': {...},
                    'volatility': {...},
                    'session': 'london_open'
                }

        Returns:
            Comprehensive execution audit
        """
        prompt = self._build_execution_prompt(data)

        system_prompt = """You are an institutional execution trader with 15+ years experience.

Your expertise:
- Order book analysis (bid/ask depth, liquidity pools)
- Slippage modeling (historical fill rates, market impact)
- Broker comparison (spread analysis, execution quality)
- Timing optimization (avoid volatile periods, spread widening)
- Aggressive Loss Management (dynamic stop levels)

Analysis Framework:
1. **Cost Analysis**: Total cost (spread + fees + slippage) vs TP
2. **Liquidity Check**: Adequate depth at entry, SL, TP levels?
3. **Broker Comparison**: Best execution venue
4. **Slippage Probability**: Expected vs worst-case fill
5. **Timing Assessment**: Is now the best time to execute?
6. **ALM Setup**: Dynamic emergency exit based on volatility

You MUST output ONLY valid JSON in this exact format:
{
  "cost_analysis": {
    "spread_pips": 1.2,
    "fees_pips": 0.5,
    "expected_slippage_pips": 0.3,
    "worst_case_slippage_pips": 0.8,
    "total_cost_pips": 2.0,
    "tp_pips": 40.0,
    "cost_to_tp_ratio": 0.05,
    "threshold": 0.50,
    "status": "PASS",
    "cost_grade": "A"
  },

  "liquidity_analysis": {
    "entry_level_depth": {
      "bid_volume": 15000000,
      "ask_volume": 18000000,
      "depth_score": 0.92,
      "fill_probability": 0.98
    },
    "sl_level_depth": {
      "volume_available": 12000000,
      "depth_score": 0.85,
      "exit_probability": 0.95
    },
    "tp_level_depth": {
      "volume_available": 20000000,
      "depth_score": 0.95,
      "exit_probability": 0.99
    },
    "overall_liquidity": "excellent",
    "liquidity_grade": "A"
  },

  "broker_comparison": [
    {
      "broker": "IC Markets",
      "spread_pips": 1.2,
      "execution_quality": 0.95,
      "avg_slippage": 0.2,
      "recommended": true
    },
    {
      "broker": "Pepperstone",
      "spread_pips": 1.5,
      "execution_quality": 0.92,
      "avg_slippage": 0.3,
      "recommended": false
    }
  ],
  "best_broker": "IC Markets",
  "broker_grade": "A",

  "slippage_modeling": {
    "position_size_usd": 10000,
    "market_impact_estimate": 0.2,
    "expected_slippage": 0.3,
    "confidence_interval_95": [0.1, 0.6],
    "worst_case_slippage": 0.8,
    "slippage_grade": "A"
  },

  "timing_analysis": {
    "current_session": "london_open",
    "liquidity_level": "high",
    "spread_trend": "stable",
    "volatility_state": "moderate",
    "optimal_timing": true,
    "timing_grade": "A",
    "recommendation": "Execute immediately - optimal conditions"
  },

  "alm_setup": {
    "active": true,
    "mode": "dynamic_adaptive",
    "atr_value": 0.00150,
    "volatility_multiplier": 1.2,
    "emergency_threshold": "1.2x ATR",
    "emergency_close_level": 1.25680,
    "monitoring_interval": "500ms",
    "confidence": 0.90,
    "alm_grade": "A+"
  },

  "execution_plan": {
    "order_type": "limit_order",
    "entry_price": 1.25500,
    "limit_range": [1.25495, 1.25505],
    "timeout_seconds": 30,
    "fallback_to_market": false,
    "partial_fills": "allowed"
  },

  "risk_warnings": [],

  "overall_audit": {
    "audit_result": "PASS",
    "overall_grade": "A",
    "confidence": 0.95,
    "recommendation": "APPROVED_FOR_EXECUTION",
    "execution_quality_score": 0.94
  },

  "summary": "Excellent execution environment. Tight spread (1.2 pips), deep liquidity at all levels (entry: 15M, SL: 12M, TP: 20M). Expected slippage minimal (0.3 pips, 95% CI: 0.1-0.6). Best broker: IC Markets. Optimal timing (London open, high liquidity). ALM configured with dynamic 1.2× ATR threshold. Total cost 2.0 pips vs 40 pip TP (5% ratio) - well within threshold. Grade A execution setup. APPROVED."
}

Never include explanatory text - ONLY the JSON object."""

        try:
            response = await self.client.analyze(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.0,  # Zero for precise validation
                max_tokens=6000
            )

            result = json.loads(response)
            result['agent'] = 'Execution Auditor V2 (Grok Enhanced)'
            result['cost_estimate'] = 0.15

            return result

        except Exception as e:
            return {
                'overall_audit': {
                    'audit_result': 'FAIL',
                    'recommendation': 'REJECTED_ERROR',
                    'confidence': 0.0
                },
                'error': str(e),
                'summary': f'Execution audit failed: {str(e)}'
            }

    def _build_execution_prompt(self, data: Dict[str, Any]) -> str:
        """Build comprehensive execution audit prompt"""

        hypothesis = data.get('trade_hypothesis', {})
        order_book = data.get('order_book', {})
        brokers = data.get('broker_spreads', {})
        market_depth = data.get('market_depth', {})
        volatility = data.get('volatility', {})
        session = data.get('session', 'unknown')

        entry = hypothesis.get('entry', 0)
        sl = hypothesis.get('sl', 0)
        tp = hypothesis.get('tp', 0)
        action = hypothesis.get('action', 'HOLD')

        # Calculate trade metrics (validate to prevent zero values)
        if entry <= 0 or sl <= 0 or tp <= 0:
            # REJECTED signals may have zero values - use placeholders
            sl_pips = 0.0
            tp_pips = 0.0
        else:
            sl_pips = abs(entry - sl) * 10000
            tp_pips = abs(entry - tp) * 10000
        position_size = data.get('position_size_usd', 10000)

        prompt = f"""# INSTITUTIONAL EXECUTION AUDIT - {hypothesis.get('symbol', 'UNKNOWN')}

## Trade Parameters
- **Action**: {action}
- **Entry**: {entry:.5f}
- **Stop Loss**: {sl:.5f} ({sl_pips:.1f} pips)
- **Take Profit**: {tp:.5f} ({tp_pips:.1f} pips)
- **Position Size**: ${position_size:,.0f}

## Order Book Depth (Real-time)

### At Entry Level ({entry:.5f})
- **Bid Depth**: {order_book.get('entry_bid_volume', 0):,} units
- **Ask Depth**: {order_book.get('entry_ask_volume', 0):,} units
- **Spread**: {order_book.get('spread_pips', 0):.1f} pips

### At Stop Loss ({sl:.5f})
- **Available Liquidity**: {order_book.get('sl_volume', 0):,} units
- **Depth Score**: {order_book.get('sl_depth_score', 0):.2f}

### At Take Profit ({tp:.5f})
- **Available Liquidity**: {order_book.get('tp_volume', 0):,} units
- **Depth Score**: {order_book.get('tp_depth_score', 0):.2f}

## Multi-Broker Spread Comparison

{self._format_broker_spreads(brokers)}

## Market Depth Analysis
- **Total Market Depth**: {market_depth.get('total_depth', 0):,} units
- **Buy Pressure**: {market_depth.get('buy_pressure', 0):.0%}
- **Sell Pressure**: {market_depth.get('sell_pressure', 0):.0%}
- **Imbalance**: {market_depth.get('imbalance', 'balanced')}

## Volatility & Market Conditions
- **Current ATR**: {volatility.get('atr', 0):.5f}
- **ATR Percentile (20-day)**: {volatility.get('atr_percentile', 50):.0f}%
- **Volatility State**: {volatility.get('state', 'normal')}
- **Session**: {session}
- **Time of Day**: {data.get('time_of_day', 'N/A')}

## Historical Execution Data
- **Avg Slippage (this pair)**: {data.get('historical_slippage', 0.3):.1f} pips
- **Fill Success Rate**: {data.get('fill_success_rate', 0.98):.0%}
- **Avg Execution Time**: {data.get('avg_execution_ms', 150):.0f}ms

## Your Task - Institutional Execution Audit

Perform comprehensive pre-trade execution analysis:

### 1. Cost Analysis
Calculate total execution cost:
- Spread: {order_book.get('spread_pips', 0):.1f} pips
- Fees: 0.5 pips (assumed)
- Expected slippage: ? (you calculate based on position size & depth)
- **Total Cost**: ? pips
- **Cost/TP Ratio**: ? (must be < 50%)
- **Grade**: A/B/C/D/F

### 2. Liquidity Analysis
Assess liquidity at each level:
- Entry: Is there enough depth to fill ${position_size:,.0f}?
- SL: Can we exit if stopped out?
- TP: Can we exit at profit target?
- **Fill Probabilities**: Calculate for each level
- **Overall Liquidity Grade**: A/B/C/D/F

### 3. Broker Comparison
Compare execution quality across brokers:
- Which has tightest spread?
- Which has best historical execution?
- Which minimizes slippage?
- **Recommended Broker**: ?
- **Broker Grade**: A/B/C/D/F

### 4. Slippage Modeling
Estimate slippage based on:
- Position size: ${position_size:,.0f}
- Market depth
- Historical data
- **Expected Slippage**: ? pips
- **95% Confidence Interval**: [?, ?]
- **Worst Case**: ? pips
- **Slippage Grade**: A/B/C/D/F

### 5. Timing Analysis
Is NOW the optimal time to execute?
- Session: {session}
- Liquidity: High/Medium/Low?
- Spread: Stable/Widening?
- Volatility: Normal/Elevated?
- **Optimal Timing**: Yes/No
- **Timing Grade**: A/B/C/D/F

### 6. ALM (Aggressive Loss Management) Setup
Configure dynamic emergency exit:
- ATR: {volatility.get('atr', 0):.5f}
- Volatility multiplier: ? (higher if volatile)
- Emergency threshold: ? × ATR
- Emergency close level: ?
- **ALM Grade**: A/B/C/D/F

### 7. Execution Plan
Recommend execution strategy:
- Order type: Limit or Market?
- Price range: [?, ?]
- Timeout: ? seconds
- Partial fills: Allow?

### 8. Risk Warnings
Any concerns?
- Insufficient liquidity?
- High slippage risk?
- Poor timing?
- Spread too wide?

### 9. Overall Audit
- **Audit Result**: PASS or FAIL
- **Overall Grade**: A/B/C/D/F
- **Confidence**: 0.0 to 1.0
- **Recommendation**: APPROVED or REJECTED
- **Execution Quality Score**: 0.0 to 1.0

### 10. Summary
3-4 sentence summary with:
- Key execution metrics
- Grade assessment
- Final recommendation

Output ONLY the JSON object with your comprehensive execution audit."""

        return prompt

    def _format_broker_spreads(self, brokers: Dict[str, Any]) -> str:
        """Format broker spread comparison"""
        if not brokers:
            return "No broker data available"

        broker_list = brokers.get('brokers', [])
        if not broker_list:
            return "No broker comparison data"

        output = []
        for broker in broker_list:
            name = broker.get('name', 'Unknown')
            spread = broker.get('spread_pips', 0)
            quality = broker.get('execution_quality', 0)
            slippage = broker.get('avg_slippage', 0)

            output.append(
                f"- **{name}**: Spread {spread:.1f} pips | "
                f"Quality {quality:.0%} | "
                f"Avg Slippage {slippage:.1f} pips"
            )

        return "\n".join(output)
