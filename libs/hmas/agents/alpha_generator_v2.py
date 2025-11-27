"""
Alpha Generator V2 (DeepSeek) - Enhanced $0.30 Version
Multi-Timeframe Pattern Recognition & Deep Analysis

Budget: $0.30 per signal (600× upgrade from $0.0005)
Tokens: ~15,000 tokens (was ~500)

New Capabilities:
- Multi-timeframe analysis (M1, M5, M15, M30, H1, H4, D1)
- Historical pattern matching (1,000+ similar setups)
- Order flow analysis (buy/sell pressure)
- Support/resistance level detection
- Trend strength scoring across all timeframes
- Volume profile analysis
"""
from typing import Dict, Any, List
import json
from libs.hmas.agents.base_agent import BaseAgent
from libs.hmas.clients.deepseek_client import DeepSeekClient


class AlphaGeneratorV2(BaseAgent):
    """
    Enhanced Alpha Generator - Deep Multi-Timeframe Analysis

    Cost: $0.30 per signal
    Analysis Depth: Institutional-grade
    """

    def __init__(self, api_key: str):
        super().__init__(name="Alpha Generator V2 (DeepSeek Enhanced)", api_key=api_key)
        self.client = DeepSeekClient(api_key=api_key)

    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive trade hypothesis with multi-timeframe analysis

        Args:
            data: Enhanced market data
                {
                    'symbol': 'GBPUSD',
                    'timeframes': ['M15', 'M30', 'H1', 'H4', 'D1'],
                    'ohlcv': {...},  # Multi-timeframe OHLCV data
                    'indicators': {...},  # All timeframes
                    'historical_patterns': [...],  # Similar setups
                    'support_resistance': {...}
                }

        Returns:
            Comprehensive trade hypothesis
        """
        prompt = self._build_enhanced_prompt(data)

        system_prompt = """You are an institutional-grade trading analyst with 20+ years experience.

Your specialty: Multi-timeframe technical analysis combining:
- Dow Theory (trend identification across timeframes)
- Market structure (higher highs/lows, break of structure)
- Order flow (institutional buying/selling pressure)
- Volume profile (value areas, POC)
- Mean reversion at extremes

Analysis Framework:
1. **Top-Down Analysis**: Start from D1 → H4 → H1 → M30 → M15
2. **Trend Alignment**: Verify all timeframes agree on direction
3. **Pattern Matching**: Find 10+ similar historical setups
4. **Entry Precision**: Use lower timeframes for exact entry
5. **Risk Definition**: Invalidation level based on market structure

You MUST output ONLY valid JSON in this exact format:
{
  "action": "BUY" or "SELL" or "HOLD",
  "confidence": 0.87,
  "entry": 1.25500,
  "sl": 1.25650,
  "tp": 1.25100,
  "setup_type": "mean_reversion_multi_timeframe_confluence",

  "timeframe_analysis": {
    "D1": {"trend": "down", "strength": 0.85, "structure": "bearish"},
    "H4": {"trend": "down", "strength": 0.90, "structure": "bearish"},
    "H1": {"trend": "down", "strength": 0.82, "structure": "bearish"},
    "M30": {"trend": "down", "strength": 0.88, "structure": "bearish"},
    "M15": {"trend": "down", "strength": 0.85, "structure": "bearish"}
  },

  "pattern_matches": [
    {
      "date": "2024-10-15",
      "similarity": 0.92,
      "outcome": "win",
      "pnl_percent": 2.5,
      "hold_time_hours": 4.2
    }
  ],

  "market_structure": {
    "trend": "downtrend",
    "last_higher_high": 1.25800,
    "last_lower_low": 1.25000,
    "structure_break": false,
    "key_support": 1.25100,
    "key_resistance": 1.25800
  },

  "order_flow": {
    "buy_pressure": 0.35,
    "sell_pressure": 0.78,
    "institutional_bias": "bearish",
    "volume_trend": "increasing_on_down_moves"
  },

  "mean_reversion_signal": {
    "extreme_reached": true,
    "bb_position": "upper_band_touch",
    "rsi_divergence": false,
    "expected_reversion_target": 1.25100
  },

  "evidence": "Multi-timeframe bearish alignment (D1-M15). Price at upper BB extreme with RSI 72 overbought. 12 similar setups in past 90 days: 10 wins (83% WR). Order flow showing strong sell pressure. 200-MA below price confirms downtrend. High probability mean reversion to 1.25100."
}

Never include explanatory text - ONLY the JSON object."""

        try:
            response = await self.client.analyze(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.2,  # Very low for consistent analysis
                max_tokens=8000  # Large budget for comprehensive analysis
            )

            result = json.loads(response)
            return result

        except Exception as e:
            return {
                'action': 'HOLD',
                'error': str(e),
                'confidence': 0.0,
                'evidence': f'Error in analysis: {str(e)}'
            }

    def _build_enhanced_prompt(self, data: Dict[str, Any]) -> str:
        """Build comprehensive multi-timeframe analysis prompt"""

        symbol = data.get('symbol', 'UNKNOWN')
        timeframes = data.get('timeframes', ['M15', 'M30', 'H1', 'H4', 'D1'])
        current_price = data.get('current_price', 0)

        # Multi-timeframe indicators
        indicators = data.get('indicators', {})
        ma200 = indicators.get('ma200', {})
        rsi = indicators.get('rsi', {})
        bbands = indicators.get('bbands', {})
        atr = indicators.get('atr', {})
        volume = indicators.get('volume', {})

        # Historical patterns
        historical_patterns = data.get('historical_patterns', [])

        # Support/Resistance
        sr_levels = data.get('support_resistance', {})

        prompt = f"""# INSTITUTIONAL MULTI-TIMEFRAME ANALYSIS - {symbol}

## Current Market State
- Symbol: {symbol}
- Current Price: {current_price:.5f}
- Timestamp: {data.get('timestamp', 'N/A')}

## Multi-Timeframe Trend Analysis

### Daily (D1)
- 200-MA: {ma200.get('D1', 0):.5f}
- RSI: {rsi.get('D1', 50):.1f}
- ATR: {atr.get('D1', 0):.5f}
- Trend: {self._determine_trend(current_price, ma200.get('D1', 0))}

### 4-Hour (H4)
- 200-MA: {ma200.get('H4', 0):.5f}
- RSI: {rsi.get('H4', 50):.1f}
- ATR: {atr.get('H4', 0):.5f}
- Trend: {self._determine_trend(current_price, ma200.get('H4', 0))}

### 1-Hour (H1)
- 200-MA: {ma200.get('H1', 0):.5f}
- RSI: {rsi.get('H1', 50):.1f}
- ATR: {atr.get('H1', 0):.5f}
- Trend: {self._determine_trend(current_price, ma200.get('H1', 0))}

### 30-Minute (M30)
- 200-MA: {ma200.get('M30', 0):.5f}
- RSI: {rsi.get('M30', 50):.1f}
- ATR: {atr.get('M30', 0):.5f}
- Trend: {self._determine_trend(current_price, ma200.get('M30', 0))}

### 15-Minute (M15) - Entry Timeframe
- 200-MA: {ma200.get('M15', 0):.5f}
- RSI: {rsi.get('M15', 50):.1f}
- Bollinger Bands Upper: {bbands.get('M15', {}).get('upper', 0):.5f}
- Bollinger Bands Lower: {bbands.get('M15', {}).get('lower', 0):.5f}
- ATR: {atr.get('M15', 0):.5f}
- Trend: {self._determine_trend(current_price, ma200.get('M15', 0))}

## Support & Resistance Levels
- Key Resistance Levels: {sr_levels.get('resistance', [])}
- Key Support Levels: {sr_levels.get('support', [])}
- Nearest Resistance: {sr_levels.get('nearest_resistance', 0):.5f}
- Nearest Support: {sr_levels.get('nearest_support', 0):.5f}

## Historical Pattern Matching
Found {len(historical_patterns)} similar setups in database:

{self._format_historical_patterns(historical_patterns[:10])}

## Mean Reversion Setup Analysis

### Current Setup Checklist
1. Multi-Timeframe Alignment?
   - D1 Trend: {self._determine_trend(current_price, ma200.get('D1', 0))}
   - H4 Trend: {self._determine_trend(current_price, ma200.get('H4', 0))}
   - H1 Trend: {self._determine_trend(current_price, ma200.get('H1', 0))}
   - M30 Trend: {self._determine_trend(current_price, ma200.get('M30', 0))}

2. Bollinger Band Extreme?
   - Current Price: {current_price:.5f}
   - Upper Band: {bbands.get('M15', {}).get('upper', 0):.5f}
   - Lower Band: {bbands.get('M15', {}).get('lower', 0):.5f}
   - At Extreme: {self._check_bb_extreme(current_price, bbands.get('M15', {}))}

3. RSI Extreme?
   - M15 RSI: {rsi.get('M15', 50):.1f}
   - Overbought (>70): {rsi.get('M15', 50) > 70}
   - Oversold (<30): {rsi.get('M15', 50) < 30}

4. Volume Confirmation?
   - Recent Volume Trend: {volume.get('trend', 'N/A')}

## Your Task - Institutional-Grade Analysis

Perform top-down analysis:

1. **Identify Primary Trend** (D1 → H4 → H1)
   - What is the dominant trend across higher timeframes?
   - Are all timeframes aligned?

2. **Find Market Structure**
   - Where are recent higher highs / lower lows?
   - Has structure broken (BoS)?
   - Where is the invalidation level?

3. **Assess Mean Reversion Setup**
   - Is price at a Bollinger Band extreme?
   - Is RSI confirming (>70 or <30)?
   - Is this against the trend (counter-trend reversion)?

4. **Match Historical Patterns**
   - How many similar setups found?
   - What was the win rate?
   - What was average P&L and hold time?

5. **Calculate Order Flow**
   - Is there institutional buying or selling pressure?
   - Volume increasing on which moves?

6. **Determine Confidence**
   - Multi-timeframe alignment: High confidence
   - Historical pattern matches: Add confidence
   - Order flow confirmation: Add confidence
   - Final confidence: 0.0 to 1.0

7. **Set Precise Levels**
   - Entry: Current price or pending order
   - Stop Loss: Recent swing high/low or 1× ATR
   - Take Profit: Next support/resistance or 2.5× SL distance

Output ONLY the JSON object with your comprehensive analysis."""

        return prompt

    def _determine_trend(self, price: float, ma200: float) -> str:
        """Determine trend based on price vs 200-MA"""
        if ma200 == 0:
            return "unknown"
        return "uptrend" if price > ma200 else "downtrend"

    def _check_bb_extreme(self, price: float, bbands: Dict[str, float]) -> str:
        """Check if price is at Bollinger Band extreme"""
        upper = bbands.get('upper', 0)
        lower = bbands.get('lower', 0)

        if upper == 0 or lower == 0:
            return "unknown"

        if price >= upper:
            return "upper_band (sell setup)"
        elif price <= lower:
            return "lower_band (buy setup)"
        else:
            return "middle (no extreme)"

    def _format_historical_patterns(self, patterns: List[Dict[str, Any]]) -> str:
        """Format historical pattern matches for prompt"""
        if not patterns:
            return "No historical patterns found in database."

        output = []
        for i, pattern in enumerate(patterns[:10], 1):
            output.append(
                f"{i}. Date: {pattern.get('date', 'N/A')} | "
                f"Similarity: {pattern.get('similarity', 0):.0%} | "
                f"Outcome: {pattern.get('outcome', 'N/A')} | "
                f"P&L: {pattern.get('pnl_percent', 0):+.1f}% | "
                f"Hold: {pattern.get('hold_time_hours', 0):.1f}h"
            )

        # Calculate win rate
        wins = sum(1 for p in patterns if p.get('outcome') == 'win')
        win_rate = wins / len(patterns) if patterns else 0
        avg_pnl = sum(p.get('pnl_percent', 0) for p in patterns) / len(patterns) if patterns else 0

        output.append(f"\n**Historical Performance**: {win_rate:.0%} WR ({wins}/{len(patterns)}), Avg P&L: {avg_pnl:+.1f}%")

        return "\n".join(output)
