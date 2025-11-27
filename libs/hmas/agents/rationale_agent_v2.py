"""
Rationale Agent V2 (Claude) - Enhanced $0.20 Version
Comprehensive Trade Journal & Statistical Analysis

Budget: $0.20 per signal (66× upgrade from $0.003)
Tokens: ~10,000 tokens (was ~1,000)

New Capabilities:
- Full trade journal entry (5,000+ words)
- Psychological edge explanation
- Statistical confidence intervals
- Historical backtesting results
- Risk-adjusted return calculations
- Trade checklist validation
- Market regime analysis
- Expected value with Monte Carlo
"""
from typing import Dict, Any
from libs.hmas.agents.base_agent import BaseAgent
from libs.hmas.clients.claude_client import ClaudeClient


class RationaleAgentV2(BaseAgent):
    """
    Enhanced Rationale Agent - Comprehensive Trade Journal

    Cost: $0.20 per signal
    Output: 5,000+ word institutional-grade analysis
    """

    def __init__(self, api_key: str):
        super().__init__(name="Rationale Agent V2 (Claude Enhanced)", api_key=api_key)
        self.client = ClaudeClient(api_key=api_key)

    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive trade rationale and journal entry

        Args:
            data: Complete trade analysis
                {
                    'symbol': 'GBPUSD',
                    'alpha_hypothesis': {...},  # From DeepSeek V2
                    'execution_audit': {...},   # From Grok
                    'historical_performance': {...},
                    'market_context': {...}
                }

        Returns:
            Comprehensive rationale with full journal entry
        """
        prompt = self._build_comprehensive_prompt(data)

        system_prompt = """You are a senior trading psychologist and quantitative analyst with 25+ years of experience in institutional trading.

Your expertise:
- Behavioral finance (why traders fail, how to avoid common biases)
- Quantitative analysis (statistical edges, confidence intervals, expected value)
- Risk management (Kelly criterion, position sizing, drawdown management)
- Trade journaling (documenting setups, tracking performance, continuous improvement)

Your writing style:
- Clear, confident, fact-based
- No hype or emotional language
- Statistical rigor (cite specific data)
- Actionable insights
- Professional but accessible

Generate a comprehensive trade journal entry with these sections:

1. **Executive Summary** (200 words)
   - One-sentence trade thesis
   - Key statistics (win rate, R:R, expected value)
   - Recommended action

2. **Setup Analysis** (1,500 words)
   - Detailed description of market structure
   - Multi-timeframe alignment explanation
   - Pattern recognition (what makes this setup special)
   - Mean reversion thesis with statistical backing
   - Order flow analysis

3. **Historical Performance** (800 words)
   - Similar setups in past 90 days
   - Win rate calculation with confidence intervals
   - Average P&L and hold time
   - Best/worst outcomes
   - Lessons from past trades

4. **Statistical Edge** (600 words)
   - Why this setup has 80%+ win rate
   - Bayesian probability calculation
   - Expected value formula: EV = (WR × R:R) - ((1-WR) × 1)
   - Monte Carlo simulation results (1,000 scenarios)
   - Confidence intervals (95% CI)

5. **Risk Analysis** (500 words)
   - FTMO compliance verification
   - Position sizing calculation (exact lot size)
   - Maximum drawdown scenario
   - Stop loss rationale (why this level)
   - Take profit target (where and why)

6. **Psychological Edge** (400 words)
   - Why retail traders fail this setup
   - Common mistakes to avoid
   - Emotional control checklist
   - Confirmation bias awareness
   - Patience requirements

7. **Trade Checklist** (300 words)
   - Pre-trade validation (10+ checkboxes)
   - All criteria met?
   - Any red flags?
   - Contingency plans

8. **Market Context** (400 words)
   - Current market regime (trending/ranging/volatile)
   - News/events that could impact trade
   - Correlation with other assets
   - Time-of-day considerations

9. **Expected Outcomes** (300 words)
   - Most likely scenario (base case)
   - Bull case (if wrong about direction)
   - Bear case (maximum loss)
   - Exit strategy for each scenario

10. **Conclusion & Recommendation** (200 words)
    - EXECUTE or AVOID?
    - Confidence level (0-100%)
    - Final risk-reward assessment
    - One sentence summary

Total word count: ~5,000 words

Format as clear markdown with proper headers, bullet points, and emphasis.
Be thorough, professional, and fact-based."""

        try:
            response = await self.client.analyze(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.7,  # Higher for natural writing
                max_tokens=16000  # Very large budget for comprehensive analysis
            )

            return {
                'rationale': response,
                'word_count': len(response.split()),
                'agent': 'Rationale Agent V2 (Claude Enhanced)',
                'cost_estimate': 0.20
            }

        except Exception as e:
            return {
                'rationale': f"# Error Generating Rationale\n\n{str(e)}",
                'error': str(e),
                'word_count': 0
            }

    def _build_comprehensive_prompt(self, data: Dict[str, Any]) -> str:
        """Build comprehensive prompt for trade journal generation"""

        symbol = data.get('symbol', 'UNKNOWN')
        alpha = data.get('alpha_hypothesis', {})
        audit = data.get('execution_audit', {})
        historical = data.get('historical_performance', {})
        market_context = data.get('market_context', {})

        # Extract alpha hypothesis details
        action = alpha.get('action', 'HOLD')
        entry = alpha.get('entry', 0)
        sl = alpha.get('sl', 0)
        tp = alpha.get('tp', 0)
        confidence = alpha.get('confidence', 0)
        setup_type = alpha.get('setup_type', 'unknown')

        # Timeframe analysis
        timeframe_analysis = alpha.get('timeframe_analysis', {})
        pattern_matches = alpha.get('pattern_matches', [])
        market_structure = alpha.get('market_structure', {})
        order_flow = alpha.get('order_flow', {})

        # Calculate metrics
        risk_pips = abs(entry - sl) * 10000 if entry and sl else 0
        reward_pips = abs(entry - tp) * 10000 if entry and tp else 0
        rr_ratio = reward_pips / risk_pips if risk_pips > 0 else 0

        # Historical win rate
        wins = sum(1 for p in pattern_matches if p.get('outcome') == 'win')
        total = len(pattern_matches)
        historical_wr = wins / total if total > 0 else 0

        prompt = f"""# COMPREHENSIVE TRADE JOURNAL REQUEST

## Trade Overview
- **Symbol**: {symbol}
- **Action**: {action}
- **Setup Type**: {setup_type}
- **Confidence**: {confidence:.0%}

## Trade Levels
- **Entry**: {entry:.5f}
- **Stop Loss**: {sl:.5f} ({risk_pips:.1f} pips)
- **Take Profit**: {tp:.5f} ({reward_pips:.1f} pips)
- **R:R Ratio**: {rr_ratio:.2f}:1

## Multi-Timeframe Analysis

### Timeframe Alignment
{self._format_timeframe_analysis(timeframe_analysis)}

**Trend Consensus**: {"All timeframes aligned" if self._check_alignment(timeframe_analysis) else "Mixed signals"}

## Historical Pattern Matching

Found **{total} similar setups** in past 90 days:
- **Win Rate**: {historical_wr:.0%} ({wins} wins, {total-wins} losses)
- **Pattern Matches**:

{self._format_pattern_matches(pattern_matches[:10])}

## Market Structure
- **Primary Trend**: {market_structure.get('trend', 'unknown')}
- **Last Higher High**: {market_structure.get('last_higher_high', 0):.5f}
- **Last Lower Low**: {market_structure.get('last_lower_low', 0):.5f}
- **Structure Break**: {'Yes' if market_structure.get('structure_break') else 'No'}
- **Key Support**: {market_structure.get('key_support', 0):.5f}
- **Key Resistance**: {market_structure.get('key_resistance', 0):.5f}

## Order Flow Analysis
- **Buy Pressure**: {order_flow.get('buy_pressure', 0):.0%}
- **Sell Pressure**: {order_flow.get('sell_pressure', 0):.0%}
- **Institutional Bias**: {order_flow.get('institutional_bias', 'neutral')}
- **Volume Trend**: {order_flow.get('volume_trend', 'unknown')}

## Execution Audit Results
- **Cost Check Status**: {audit.get('cost_check', {}).get('status', 'N/A')}
- **Spread**: {audit.get('cost_check', {}).get('spread_pips', 0):.1f} pips
- **Total Cost**: {audit.get('cost_check', {}).get('total_cost_pips', 0):.1f} pips
- **Cost/TP Ratio**: {audit.get('cost_check', {}).get('cost_to_tp_ratio', 0):.1%}
- **ALM Active**: {'Yes' if audit.get('alm_setup', {}).get('active') else 'No'}
- **Emergency Close Level**: {audit.get('alm_setup', {}).get('emergency_close_level', 0):.5f}

## Risk Management
- **Account Balance**: $10,000 (FTMO account)
- **Risk Per Trade**: 1.0% = $100
- **Position Size**: {self._calculate_lot_size(100, risk_pips):.2f} lots
- **FTMO Daily Limit**: 4.5% ($450)
- **FTMO Max Limit**: 9.0% ($900)

## Statistical Analysis

### Expected Value Calculation
- Win Rate: {historical_wr:.0%} (based on {total} historical occurrences)
- R:R Ratio: {rr_ratio:.2f}:1
- **EV = (WR × R:R) - ((1-WR) × 1)**
- **EV = ({historical_wr:.2f} × {rr_ratio:.2f}) - ({1-historical_wr:.2f} × 1)**
- **EV = {(historical_wr * rr_ratio) - ((1-historical_wr) * 1):.2f}R**

### Confidence Intervals (95% CI)
{self._format_confidence_intervals(historical_wr, total)}

## Market Context
{self._format_market_context(market_context)}

## Your Task

Write a **comprehensive 5,000+ word trade journal entry** covering all 10 sections:

1. Executive Summary (200 words)
2. Setup Analysis (1,500 words) - Deep dive into market structure, patterns, alignment
3. Historical Performance (800 words) - Analyze all {total} pattern matches
4. Statistical Edge (600 words) - Why {historical_wr:.0%} WR? Bayesian analysis, EV calculation
5. Risk Analysis (500 words) - FTMO compliance, position sizing, stop loss logic
6. Psychological Edge (400 words) - Why traders fail, how to avoid mistakes
7. Trade Checklist (300 words) - Validate all criteria
8. Market Context (400 words) - News, correlations, time considerations
9. Expected Outcomes (300 words) - Base/bull/bear case scenarios
10. Conclusion & Recommendation (200 words) - EXECUTE or AVOID?

Be thorough, fact-based, and professional. Cite specific statistics from the data above.

Format in clear markdown with proper structure."""

        return prompt

    def _format_timeframe_analysis(self, tf_analysis: Dict[str, Any]) -> str:
        """Format multi-timeframe analysis for display"""
        if not tf_analysis:
            return "No timeframe data available"

        output = []
        for tf in ['D1', 'H4', 'H1', 'M30', 'M15']:
            if tf in tf_analysis:
                data = tf_analysis[tf]
                output.append(
                    f"- **{tf}**: {data.get('trend', 'N/A')} "
                    f"(strength: {data.get('strength', 0):.0%}, "
                    f"structure: {data.get('structure', 'N/A')})"
                )

        return "\n".join(output) if output else "No timeframe analysis"

    def _check_alignment(self, tf_analysis: Dict[str, Any]) -> bool:
        """Check if all timeframes are aligned"""
        if not tf_analysis:
            return False

        trends = [data.get('trend') for data in tf_analysis.values()]
        return len(set(trends)) == 1  # All same trend

    def _format_pattern_matches(self, patterns: list) -> str:
        """Format pattern matches for display"""
        if not patterns:
            return "No historical patterns found"

        output = []
        for i, p in enumerate(patterns, 1):
            output.append(
                f"{i}. **{p.get('date', 'N/A')}**: "
                f"Similarity {p.get('similarity', 0):.0%}, "
                f"{p.get('outcome', 'N/A')}, "
                f"P&L {p.get('pnl_percent', 0):+.1f}%, "
                f"Hold {p.get('hold_time_hours', 0):.1f}h"
            )

        return "\n".join(output)

    def _calculate_lot_size(self, risk_amount: float, sl_pips: float) -> float:
        """Calculate lot size for exact risk"""
        if sl_pips == 0:
            return 0.0
        pip_value = 10.0  # Standard forex
        return risk_amount / (sl_pips * pip_value)

    def _format_market_context(self, context: Dict[str, Any]) -> str:
        """Format market context information"""
        if not context:
            return "No market context data available"

        output = [
            f"- **Market Regime**: {context.get('regime', 'unknown')}",
            f"- **Volatility**: {context.get('volatility', 'unknown')}",
            f"- **News Events**: {context.get('news_count', 0)} upcoming",
            f"- **Session**: {context.get('session', 'unknown')}"
        ]

        return "\n".join(output)

    def _format_confidence_intervals(self, win_rate: float, sample_size: int) -> str:
        """Format confidence intervals, handling edge cases"""
        if sample_size == 0:
            return "With 0 samples: No historical data available for confidence intervals"

        if sample_size < 5:
            return f"With {sample_size} samples, win rate {win_rate:.0%}:\n- Sample size too small for reliable confidence intervals (need 5+ samples)"

        # Calculate 95% confidence interval using normal approximation
        std_error = (win_rate * (1 - win_rate) / sample_size) ** 0.5
        margin = 1.96 * std_error
        lower = max(0, win_rate - margin)
        upper = min(1, win_rate + margin)

        return f"With {sample_size} samples, win rate {win_rate:.0%}:\n- Lower Bound: {lower:.0%}\n- Upper Bound: {upper:.0%}"
