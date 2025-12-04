"""
Macro Analysis Agent - $0.07 Version
Economic Data, Central Banks, Correlations, Market Regime

Budget: $0.07 per signal
Tokens: ~3,500 tokens

Capabilities:
- Economic calendar (GDP, CPI, NFP, employment data)
- Central bank policy (rate expectations, QE/QT)
- Cross-asset correlations (DXY, Gold, Oil, Bonds)
- Market regime detection (risk-on/risk-off)
- Seasonal patterns
- Geopolitical events
"""
from typing import Dict, Any, List
import json
from libs.hmas.agents.base_agent import BaseAgent
from libs.hmas.clients.deepseek_client import DeepSeekClient


class MacroAgent(BaseAgent):
    """
    Macro Analysis Agent - Economics & Market Regime

    Cost: $0.07 per signal
    Specialty: Top-down macro analysis and correlations
    """

    def __init__(self, api_key: str):
        super().__init__(name="Macro Analysis Agent (DeepSeek)", api_key=api_key)
        self.client = DeepSeekClient(api_key=api_key)

    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform macro-economic analysis

        Args:
            data: Macro data including calendar, correlations, regime
                {
                    'symbol': 'GBPUSD',
                    'economic_calendar': [...],
                    'central_bank_policy': {...},
                    'correlations': {...},
                    'market_regime': {...}
                }

        Returns:
            Comprehensive macro analysis
        """
        prompt = self._build_macro_prompt(data)

        system_prompt = """You are a macro-economic analyst with expertise in:
- Central bank policy (Fed, ECB, BoE, BoJ rate decisions)
- Economic indicators (GDP, CPI, unemployment, PMI)
- Cross-asset correlations (currencies, commodities, bonds)
- Market regime analysis (risk-on vs risk-off)
- Geopolitical risk assessment

Analysis framework:
1. **Economic Calendar**: Upcoming high-impact events
2. **Central Bank Policy**: Current stance and expectations
3. **Correlations**: How asset moves with DXY, gold, oil
4. **Market Regime**: Risk appetite, volatility environment
5. **Trading Implications**: Macro tailwinds or headwinds

You MUST output ONLY valid JSON in this exact format:
{
  "economic_calendar": [
    {
      "event": "UK CPI",
      "date": "2024-11-27 09:00",
      "impact": "high",
      "forecast": 2.5,
      "previous": 2.3,
      "implication": "Higher than expected CPI could strengthen GBP",
      "risk_level": "high"
    }
  ],

  "central_bank_policy": {
    "bank_of_england": {
      "current_rate": 5.25,
      "next_meeting": "2024-12-15",
      "rate_expectations": "hold or cut 25bps",
      "policy_stance": "dovish_pivot",
      "market_pricing": "35% chance of cut",
      "implication": "Dovish sentiment weighing on GBP"
    },
    "federal_reserve": {
      "current_rate": 5.50,
      "policy_stance": "on_hold",
      "implication": "Rate differential favors USD"
    }
  },

  "correlations": {
    "dxy": {
      "correlation": 0.82,
      "strength": "strong_positive",
      "implication": "GBP moves with dollar - if DXY rallies, GBP/USD falls",
      "recent_dxy_trend": "uptrend"
    },
    "gold": {
      "correlation": -0.45,
      "strength": "moderate_negative",
      "implication": "Gold falling suggests risk-off, negative for GBP"
    },
    "oil": {
      "correlation": 0.35,
      "strength": "weak_positive",
      "implication": "Oil prices have limited impact on GBP"
    }
  },

  "market_regime": {
    "type": "risk_off",
    "vix_level": 22.5,
    "volatility": "elevated",
    "equity_trend": "down",
    "bond_yields": "falling",
    "safe_haven_flow": "active",
    "interpretation": "Risk-off environment. Flight to safety favors USD over GBP.",
    "confidence": 0.80
  },

  "seasonal_patterns": {
    "month": "November",
    "historical_tendency": "GBP weakness in late November",
    "win_rate": 0.65,
    "avg_move": -0.8,
    "reliability": "moderate"
  },

  "geopolitical_risk": {
    "level": "medium",
    "factors": ["UK political uncertainty", "US election aftermath"],
    "implication": "Moderate uncertainty, slight GBP headwind",
    "impact_on_trade": "minor_negative"
  },

  "overall_macro_assessment": {
    "macro_bias": "bearish_gbp",
    "confidence": 0.75,
    "key_drivers": [
      "BoE dovish pivot vs Fed hawkish hold",
      "Risk-off environment favors USD",
      "Seasonal weakness for GBP in November"
    ],
    "macro_tailwind_or_headwind": "headwind_for_long_gbp",
    "timing_considerations": "CPI release in 2 days - potential volatility"
  },

  "summary": "Macro environment bearish for GBP. BoE dovish pivot vs Fed on hold creates rate differential favoring USD. Risk-off regime with elevated VIX (22.5) supports safe-haven flows to USD. Strong DXY correlation (0.82) confirms weakness. UK CPI in 2 days could spike volatility. Seasonal patterns show GBP weakness in late Nov (65% WR). Overall: Macro headwinds support GBP short bias."
}

Never include explanatory text - ONLY the JSON object."""

        try:
            response = await self.client.analyze(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.2,
                max_tokens=2500
            )

            result = json.loads(response)
            result['agent'] = 'Macro Analysis Agent'
            result['cost_estimate'] = 0.07

            return result

        except Exception as e:
            return {
                'error': str(e),
                'overall_macro_assessment': {
                    'macro_bias': 'neutral',
                    'confidence': 0.0
                },
                'summary': f'Macro analysis failed: {str(e)}'
            }

    def _build_macro_prompt(self, data: Dict[str, Any]) -> str:
        """Build macro analysis prompt"""

        symbol = data.get('symbol', 'UNKNOWN')
        calendar = data.get('economic_calendar', [])
        cb_policy = data.get('central_bank_policy', {})
        correlations = data.get('correlations', {})
        regime = data.get('market_regime', {})

        # Parse currency pair
        base_currency = symbol[:3] if len(symbol) >= 6 else 'GBP'
        quote_currency = symbol[3:6] if len(symbol) >= 6 else 'USD'

        prompt = f"""# MACRO ANALYSIS REQUEST - {symbol}

## Currency Pair
- **Base Currency**: {base_currency}
- **Quote Currency**: {quote_currency}
- **Pair**: {base_currency}/{quote_currency}

## Economic Calendar (Next 7 Days)

{self._format_economic_calendar(calendar)}

## Central Bank Policy

### {base_currency} Central Bank
- Current Policy Rate: {cb_policy.get(f'{base_currency.lower()}_rate', 'N/A')}%
- Next Meeting: {cb_policy.get(f'{base_currency.lower()}_next_meeting', 'N/A')}
- Policy Stance: {cb_policy.get(f'{base_currency.lower()}_stance', 'N/A')}
- Market Expectations: {cb_policy.get(f'{base_currency.lower()}_expectations', 'N/A')}

### {quote_currency} Central Bank
- Current Policy Rate: {cb_policy.get(f'{quote_currency.lower()}_rate', 'N/A')}%
- Next Meeting: {cb_policy.get(f'{quote_currency.lower()}_next_meeting', 'N/A')}
- Policy Stance: {cb_policy.get(f'{quote_currency.lower()}_stance', 'N/A')}
- Market Expectations: {cb_policy.get(f'{quote_currency.lower()}_expectations', 'N/A')}

**Rate Differential**: {cb_policy.get('rate_differential', 'N/A')} bps

## Cross-Asset Correlations

### US Dollar Index (DXY)
- Correlation with {symbol}: {correlations.get('dxy', 0):.2f}
- Recent DXY Trend: {correlations.get('dxy_trend', 'N/A')}
- DXY Level: {correlations.get('dxy_level', 'N/A')}

### Gold (XAU/USD)
- Correlation with {symbol}: {correlations.get('gold', 0):.2f}
- Recent Gold Trend: {correlations.get('gold_trend', 'N/A')}

### Oil (WTI)
- Correlation with {symbol}: {correlations.get('oil', 0):.2f}
- Recent Oil Trend: {correlations.get('oil_trend', 'N/A')}

### 10-Year Treasury Yields
- Correlation with {symbol}: {correlations.get('yields', 0):.2f}
- Recent Yield Trend: {correlations.get('yields_trend', 'N/A')}

## Market Regime Analysis

- **VIX Level**: {regime.get('vix', 'N/A')}
- **Regime Type**: {regime.get('type', 'neutral')}
  - Risk-On: Equities up, safe havens down, high-beta currencies strong
  - Risk-Off: Equities down, safe havens up (USD, JPY, CHF strong)
- **Equity Markets**: {regime.get('equities', 'N/A')}
- **Bond Yields**: {regime.get('bonds', 'N/A')}
- **Safe Haven Flows**: {regime.get('safe_haven_flows', 'N/A')}

## Seasonal Analysis
- **Current Month**: {regime.get('month', 'November')}
- **Historical Patterns**: {regime.get('seasonal_pattern', 'No data')}

## Geopolitical Context
- **Risk Level**: {regime.get('geopolitical_risk', 'low')}
- **Key Events**: {', '.join(regime.get('geopolitical_events', ['None']))}

## Your Task

Perform comprehensive macro analysis:

1. **Economic Calendar Impact**
   - Which events could move the market?
   - When are high-risk periods?
   - What are consensus expectations?

2. **Central Bank Policy**
   - Which central bank is more hawkish/dovish?
   - Rate differential trend (widening/narrowing)?
   - Policy divergence trading opportunity?

3. **Cross-Asset Correlations**
   - What are key correlations?
   - Are correlated assets confirming or diverging?
   - Which correlation is most reliable?

4. **Market Regime**
   - Risk-on or risk-off?
   - How does this affect {symbol}?
   - Safe haven flows active?

5. **Seasonal Patterns**
   - Any seasonal tendency?
   - Historical win rate?
   - Reliability?

6. **Geopolitical Risk**
   - Any major risks?
   - Impact on currencies?

7. **Overall Macro Assessment**
   - Bullish or bearish macro bias?
   - Key macro drivers
   - Tailwind or headwind for trade?
   - Timing considerations

8. **Summary**
   - 3-4 sentence summary with actionable insight

Output ONLY the JSON object with your comprehensive macro analysis."""

        return prompt

    def _format_economic_calendar(self, calendar: List[Dict[str, Any]]) -> str:
        """Format economic calendar events"""
        if not calendar:
            return "No major economic events in next 7 days"

        output = []
        for i, event in enumerate(calendar[:10], 1):
            date = event.get('date', 'N/A')
            name = event.get('event', 'Unknown')
            impact = event.get('impact', 'medium')
            forecast = event.get('forecast', 'N/A')
            previous = event.get('previous', 'N/A')

            output.append(
                f"{i}. **{name}**\n"
                f"   Date: {date} | Impact: {impact.upper()}\n"
                f"   Forecast: {forecast} | Previous: {previous}"
            )

        return "\n\n".join(output)
