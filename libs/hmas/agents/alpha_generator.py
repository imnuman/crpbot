"""
Alpha Generator (DeepSeek) - Layer 2
Pattern Recognition & Trade Hypothesis Generation

Responsibilities:
1. Scan M15/M30 timeframes for mean reversion setups
2. Check Bollinger Bands (price touching outer band)
3. Confirm RSI oversold (<30) or overbought (>70)
4. Verify 200-MA trend alignment
5. Generate initial Trade Hypothesis
"""
from typing import Dict, Any
from libs.hmas.agents.base_agent import BaseAgent
from libs.hmas.clients.deepseek_client import DeepSeekClient


class AlphaGenerator(BaseAgent):
    """
    Alpha Generator - Pattern Recognition (Layer 2)

    Uses DeepSeek for fast, cost-effective pattern analysis.
    """

    def __init__(self, api_key: str):
        super().__init__(name="Alpha Generator (DeepSeek)", api_key=api_key)
        self.client = DeepSeekClient(api_key=api_key)

    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trade hypothesis from market data

        Args:
            data: Market data including OHLCV, indicators
                {
                    'symbol': 'GBPUSD',
                    'timeframe': 'M15',
                    'current_price': 1.25500,
                    'ma200': 1.25200,
                    'rsi': 72,
                    'bbands_upper': 1.25600,
                    'bbands_lower': 1.25000,
                    'atr': 0.00150
                }

        Returns:
            Trade hypothesis dictionary
        """
        prompt = self._build_analysis_prompt(data)

        system_prompt = """You are an expert trading analyst specializing in mean reversion strategies.

Strategy: Mean Reversion + 200-MA Trend Filter
- BUY when price touches lower Bollinger Band + RSI < 30 + price ABOVE 200-MA (uptrend)
- SELL when price touches upper Bollinger Band + RSI > 70 + price BELOW 200-MA (downtrend)

You MUST output ONLY valid JSON in this exact format:
{
  "action": "BUY" or "SELL" or "HOLD",
  "entry": 1.25500,
  "sl": 1.25650,
  "tp": 1.25100,
  "confidence": 0.85,
  "setup_type": "mean_reversion_bbands_rsi",
  "ma200_alignment": true,
  "ma200_value": 1.25200,
  "current_price": 1.25500,
  "bbands_signal": "upper_band_touch" or "lower_band_touch" or "none",
  "rsi_value": 72,
  "evidence": "Price at upper BB with RSI overbought, 200-MA below = bearish trend"
}

Never include explanatory text - ONLY the JSON object."""

        try:
            response = await self.client.analyze(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,  # Low for consistent pattern recognition
                max_tokens=800
            )

            # Parse JSON
            import json
            result = json.loads(response)

            return result

        except Exception as e:
            return {
                'action': 'HOLD',
                'error': str(e),
                'confidence': 0.0
            }

    def _build_analysis_prompt(self, data: Dict[str, Any]) -> str:
        """Build prompt for alpha generation"""

        symbol = data.get('symbol', 'UNKNOWN')
        timeframe = data.get('timeframe', 'M15')
        current_price = data.get('current_price', 0)
        ma200 = data.get('ma200', 0)
        rsi = data.get('rsi', 50)
        bbands_upper = data.get('bbands_upper', 0)
        bbands_lower = data.get('bbands_lower', 0)
        atr = data.get('atr', 0.001)

        prompt = f"""# MARKET ANALYSIS - {symbol}

## Current Market State
- Symbol: {symbol}
- Timeframe: {timeframe}
- Current Price: {current_price:.5f}
- 200-MA: {ma200:.5f}

## Indicators
- RSI (14): {rsi:.1f}
- Bollinger Bands Upper: {bbands_upper:.5f}
- Bollinger Bands Lower: {bbands_lower:.5f}
- ATR (14): {atr:.5f}

## Mean Reversion Setup Rules

### SELL Setup (Bearish Mean Reversion)
✓ Price touches/exceeds upper Bollinger Band
✓ RSI > 70 (overbought)
✓ Price BELOW 200-MA (confirms downtrend)
→ Entry: At current price (near upper BB)
→ SL: 1× ATR above entry
→ TP: Middle BB or 200-MA

### BUY Setup (Bullish Mean Reversion)
✓ Price touches/falls below lower Bollinger Band
✓ RSI < 30 (oversold)
✓ Price ABOVE 200-MA (confirms uptrend)
→ Entry: At current price (near lower BB)
→ SL: 1× ATR below entry
→ TP: Middle BB or 200-MA

## Your Task
Analyze the current market state and determine:
1. Is there a valid mean reversion setup?
2. What is the trade direction (BUY/SELL/HOLD)?
3. Calculate entry, SL, and TP levels
4. Assess confidence (0.0 to 1.0)
5. Verify 200-MA alignment

Output ONLY the JSON object, no explanatory text."""

        return prompt
