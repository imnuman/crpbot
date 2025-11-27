"""
Technical Analysis Agent - $0.10 Version
Elliott Wave, Fibonacci, Chart Patterns, Harmonics

Budget: $0.10 per signal
Tokens: ~5,000 tokens

Capabilities:
- Elliott Wave count (primary + alternate scenarios)
- Fibonacci retracement & extension levels
- Chart pattern recognition (H&S, triangles, flags, wedges)
- Harmonic patterns (Gartley, Butterfly, Bat, Crab)
- Wyckoff analysis (accumulation/distribution phases)
- Key support/resistance identification
"""
from typing import Dict, Any, List
import json
from libs.hmas.agents.base_agent import BaseAgent
from libs.hmas.clients.deepseek_client import DeepSeekClient  # Using DeepSeek for technical analysis


class TechnicalAgent(BaseAgent):
    """
    Technical Analysis Agent - Elliott Wave & Advanced Patterns

    Cost: $0.10 per signal
    Specialty: Classical technical analysis with institutional precision
    """

    def __init__(self, api_key: str):
        super().__init__(name="Technical Analysis Agent (DeepSeek)", api_key=api_key)
        self.client = DeepSeekClient(api_key=api_key)

    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive technical analysis

        Args:
            data: Market data with price history
                {
                    'symbol': 'GBPUSD',
                    'price_history': [...],  # 500+ candles
                    'current_price': 1.25500,
                    'swing_highs': [...],
                    'swing_lows': [...]
                }

        Returns:
            Technical analysis results with Elliott Wave, Fibonacci, patterns
        """
        prompt = self._build_technical_prompt(data)

        system_prompt = """You are a master technical analyst specializing in:
- Elliott Wave Theory (R.N. Elliott's original work)
- Fibonacci analysis (retracements, extensions, time zones)
- Chart patterns (classical patterns from Edwards & Magee)
- Harmonic patterns (Scott Carney's work)
- Wyckoff methodology (price/volume analysis)

Your analysis is:
- Precise (exact price levels)
- Rule-based (follow Elliott Wave rules strictly)
- Probabilistic (provide confidence for each scenario)
- Actionable (clear trading implications)

You MUST output ONLY valid JSON in this exact format:
{
  "elliott_wave": {
    "primary_count": "Wave 5 of (C) complete, reversal expected",
    "alternate_count": "Wave 4 correction ongoing, one more high",
    "wave_degree": "intermediate",
    "confidence": 0.75,
    "invalidation_level": 1.25800,
    "projection": {
      "target_1": 1.25100,
      "target_2": 1.24850,
      "target_3": 1.24600
    }
  },

  "fibonacci": {
    "swing_high": 1.25800,
    "swing_low": 1.25000,
    "retracement_levels": {
      "0.236": 1.25189,
      "0.382": 1.25305,
      "0.500": 1.25400,
      "0.618": 1.25494,
      "0.786": 1.25629
    },
    "extension_levels": {
      "1.272": 1.25982,
      "1.618": 1.26295
    },
    "current_position": "at 0.618 retracement (resistance)",
    "key_level": 1.25494
  },

  "chart_patterns": [
    {
      "type": "double_top",
      "confidence": 0.85,
      "neckline": 1.25200,
      "target": 1.25100,
      "status": "confirmed",
      "implication": "bearish reversal"
    }
  ],

  "harmonic_patterns": [
    {
      "type": "bearish_bat",
      "confidence": 0.70,
      "completion_point": 1.25550,
      "target": 1.25100,
      "stop": 1.25700,
      "status": "near_completion"
    }
  ],

  "wyckoff_analysis": {
    "phase": "distribution",
    "stage": "markdown",
    "supply_demand": "supply_exceeds_demand",
    "composite_operator": "selling",
    "confidence": 0.80
  },

  "support_resistance": {
    "major_resistance": [1.25800, 1.26000],
    "major_support": [1.25000, 1.24800],
    "nearest_resistance": 1.25600,
    "nearest_support": 1.25200,
    "key_pivot": 1.25400
  },

  "summary": "Elliott Wave suggests wave 5 complete. Double top confirmed. Bearish Bat pattern near completion. Wyckoff distribution phase. Multiple bearish confluences at 1.25500-1.25600 zone. High probability reversal to 1.25100."
}

Never include explanatory text - ONLY the JSON object."""

        try:
            response = await self.client.analyze(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.1,  # Very low for precise technical analysis
                max_tokens=4000
            )

            result = json.loads(response)
            result['agent'] = 'Technical Analysis Agent'
            result['cost_estimate'] = 0.10

            return result

        except Exception as e:
            return {
                'error': str(e),
                'elliott_wave': {'primary_count': 'Error in analysis'},
                'summary': f'Technical analysis failed: {str(e)}'
            }

    def _build_technical_prompt(self, data: Dict[str, Any]) -> str:
        """Build comprehensive technical analysis prompt"""

        symbol = data.get('symbol', 'UNKNOWN')
        current_price = data.get('current_price', 0)
        price_history = data.get('price_history', [])
        swing_highs = data.get('swing_highs', [])
        swing_lows = data.get('swing_lows', [])

        # Get recent price action
        recent_high = max(swing_highs) if swing_highs else current_price
        recent_low = min(swing_lows) if swing_lows else current_price

        prompt = f"""# TECHNICAL ANALYSIS REQUEST - {symbol}

## Current Market State
- **Symbol**: {symbol}
- **Current Price**: {current_price:.5f}
- **Recent High**: {recent_high:.5f}
- **Recent Low**: {recent_low:.5f}
- **Price Range**: {(recent_high - recent_low) * 10000:.1f} pips

## Price History
{len(price_history)} candles provided for analysis.

### Recent Swing Points
**Swing Highs** (most recent 10):
{self._format_swing_points(swing_highs[-10:] if len(swing_highs) > 10 else swing_highs)}

**Swing Lows** (most recent 10):
{self._format_swing_points(swing_lows[-10:] if len(swing_lows) > 10 else swing_lows)}

## Analysis Required

### 1. Elliott Wave Analysis
Count the waves using Elliott Wave principles:
- **Impulse Waves**: 5-wave structure (1-2-3-4-5)
- **Corrective Waves**: 3-wave structure (A-B-C)
- **Rules**:
  - Wave 2 never retraces more than 100% of wave 1
  - Wave 3 is never the shortest
  - Wave 4 never enters wave 1 territory

Determine:
- Primary wave count (most likely scenario)
- Alternate wave count (backup scenario)
- Wave degree (minor, intermediate, primary)
- Invalidation level (where count becomes invalid)
- Price projections (wave targets)

### 2. Fibonacci Analysis
Calculate Fibonacci levels from recent swing high ({recent_high:.5f}) to swing low ({recent_low:.5f}):
- Retracement levels: 23.6%, 38.2%, 50.0%, 61.8%, 78.6%
- Extension levels: 127.2%, 161.8%
- Where is current price relative to these levels?
- Which level is acting as support/resistance?

### 3. Chart Pattern Recognition
Scan for classical patterns:
- **Reversal**: Head & Shoulders, Double Top/Bottom, Triple Top/Bottom
- **Continuation**: Triangles, Flags, Pennants, Wedges
- **Measured Moves**: AB=CD, ABCD patterns

For each pattern found:
- Pattern type
- Completion status
- Target price
- Confidence level

### 4. Harmonic Pattern Detection
Search for harmonic patterns:
- **Bullish**: Gartley, Bat, Butterfly, Crab (bullish versions)
- **Bearish**: Gartley, Bat, Butterfly, Crab (bearish versions)

Verify Fibonacci ratios:
- XA leg
- AB retracement of XA
- BC retracement of AB
- CD extension

### 5. Wyckoff Analysis
Determine market phase:
- **Accumulation**: Spring, test, sign of strength
- **Markup**: Uptrend with increasing volume
- **Distribution**: Upthrust, test, sign of weakness
- **Markdown**: Downtrend with increasing volume

Assess:
- Supply vs demand
- Composite operator activity
- Volume characteristics

### 6. Support & Resistance
Identify key levels:
- Major support zones (tested 3+ times)
- Major resistance zones (tested 3+ times)
- Nearest support below current price
- Nearest resistance above current price
- Key pivot point

## Your Task

Perform comprehensive technical analysis and provide:
1. Elliott Wave count with projections
2. Fibonacci levels and current position
3. Chart patterns (if any)
4. Harmonic patterns (if any)
5. Wyckoff phase
6. Support/Resistance levels
7. Summary with trading implications

Output ONLY the JSON object with your analysis."""

        return prompt

    def _format_swing_points(self, points: List[float]) -> str:
        """Format swing points for display"""
        if not points:
            return "None"

        return ", ".join([f"{p:.5f}" for p in points[-10:]])
