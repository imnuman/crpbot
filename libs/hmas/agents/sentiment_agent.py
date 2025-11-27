"""
Sentiment Analysis Agent - $0.08 Version
News, Social Media, Market Sentiment, COT Analysis

Budget: $0.08 per signal
Tokens: ~4,000 tokens

Capabilities:
- News headline sentiment (Bloomberg, Reuters, FX Street)
- Social media analysis (Twitter/X, Reddit)
- Fear & Greed index
- COT (Commitment of Traders) positioning
- Market sentiment scoring
- Contrarian indicators
"""
from typing import Dict, Any, List
import json
from libs.hmas.agents.base_agent import BaseAgent
from libs.hmas.clients.deepseek_client import DeepSeekClient


class SentimentAgent(BaseAgent):
    """
    Sentiment Analysis Agent - News, Social, COT Data

    Cost: $0.08 per signal
    Specialty: Gauging market psychology and positioning
    """

    def __init__(self, api_key: str):
        super().__init__(name="Sentiment Analysis Agent (DeepSeek)", api_key=api_key)
        self.client = DeepSeekClient(api_key=api_key)

    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market sentiment from multiple sources

        Args:
            data: Sentiment data from various sources
                {
                    'symbol': 'GBPUSD',
                    'news_headlines': [...],
                    'social_mentions': {...},
                    'cot_data': {...},
                    'fear_greed_index': 65
                }

        Returns:
            Comprehensive sentiment analysis
        """
        prompt = self._build_sentiment_prompt(data)

        system_prompt = """You are a market sentiment analyst specializing in behavioral finance.

Your expertise:
- News sentiment analysis (parsing headlines for market impact)
- Social media sentiment (Twitter/X, Reddit crowd psychology)
- COT analysis (institutional vs retail positioning)
- Contrarian indicators (when crowd is too bullish/bearish)
- Fear & Greed psychology

Analysis framework:
1. **News Sentiment**: Positive, neutral, or negative for currency
2. **Social Sentiment**: Crowd bullishness/bearishness (0-100 scale)
3. **COT Positioning**: Are speculators overleveraged?
4. **Contrarian Signal**: Is sentiment extreme (fade the crowd)?
5. **Overall Bias**: Bullish, bearish, or neutral

You MUST output ONLY valid JSON in this exact format:
{
  "news_sentiment": {
    "score": -0.65,
    "bias": "bearish",
    "confidence": 0.80,
    "key_headlines": [
      "UK GDP misses expectations, growth slows",
      "BoE hints at potential rate cuts in Q1",
      "Inflation concerns ease, GBP weakens"
    ],
    "impact_assessment": "Moderately bearish for GBP. Economic data disappointing, central bank dovish."
  },

  "social_sentiment": {
    "twitter_score": -0.55,
    "reddit_score": -0.48,
    "overall_score": -0.52,
    "bias": "bearish",
    "crowd_position": "net short GBP",
    "confidence": 0.65,
    "notable_mentions": "Retail traders heavily short GBP, expecting further weakness"
  },

  "cot_positioning": {
    "commercial_net": -15000,
    "large_speculators_net": 8000,
    "small_speculators_net": -22000,
    "interpretation": "Commercials (smart money) short, large specs long, small specs heavily short",
    "contrarian_signal": "Small specs capitulation suggests bottom near",
    "bias": "contrarian_bullish"
  },

  "fear_greed": {
    "index": 35,
    "level": "fear",
    "interpretation": "Market in fear mode, potential buying opportunity",
    "historical_context": "Fear readings often precede rebounds"
  },

  "sentiment_extremes": {
    "extreme_detected": true,
    "type": "bearish_extreme",
    "contrarian_implication": "When everyone is bearish, often time to buy",
    "confidence": 0.75
  },

  "overall_assessment": {
    "sentiment_bias": "bearish",
    "contrarian_bias": "bullish",
    "recommended_stance": "contrarian_buy",
    "confidence": 0.70,
    "reasoning": "News and social sentiment bearish, but COT shows small spec capitulation and fear index elevated. Classic contrarian setup - fade the crowd."
  },

  "summary": "News bearish (-0.65), social bearish (-0.52), but COT shows small speculators capitulation. Fear index at 35 (fear). Contrarian signal: Crowd too bearish, potential reversal setup. Recommended stance: Fade bearish sentiment, look for long entries."
}

Never include explanatory text - ONLY the JSON object."""

        try:
            response = await self.client.analyze(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=3000
            )

            result = json.loads(response)
            result['agent'] = 'Sentiment Analysis Agent'
            result['cost_estimate'] = 0.08

            return result

        except Exception as e:
            return {
                'error': str(e),
                'overall_assessment': {
                    'sentiment_bias': 'neutral',
                    'confidence': 0.0
                },
                'summary': f'Sentiment analysis failed: {str(e)}'
            }

    def _build_sentiment_prompt(self, data: Dict[str, Any]) -> str:
        """Build sentiment analysis prompt"""

        symbol = data.get('symbol', 'UNKNOWN')
        news_headlines = data.get('news_headlines', [])
        social_mentions = data.get('social_mentions', {})
        cot_data = data.get('cot_data', {})
        fear_greed = data.get('fear_greed_index', 50)

        prompt = f"""# SENTIMENT ANALYSIS REQUEST - {symbol}

## News Headlines (Past 24 Hours)

{self._format_news_headlines(news_headlines)}

## Social Media Sentiment

### Twitter/X Mentions
- Total Mentions: {social_mentions.get('twitter_total', 0)}
- Bullish: {social_mentions.get('twitter_bullish', 0)}
- Bearish: {social_mentions.get('twitter_bearish', 0)}
- Neutral: {social_mentions.get('twitter_neutral', 0)}
- Notable Influencers: {', '.join(social_mentions.get('twitter_influencers', []))}

### Reddit Discussions
- Total Posts: {social_mentions.get('reddit_posts', 0)}
- Upvotes Bullish: {social_mentions.get('reddit_bullish_upvotes', 0)}
- Upvotes Bearish: {social_mentions.get('reddit_bearish_upvotes', 0)}
- Top Subreddits: {', '.join(social_mentions.get('reddit_subs', ['r/forex', 'r/algotrading']))}

## COT (Commitment of Traders) Report

### Commercial Traders (Smart Money)
- Long Contracts: {cot_data.get('commercial_long', 0):,}
- Short Contracts: {cot_data.get('commercial_short', 0):,}
- **Net Position**: {cot_data.get('commercial_net', 0):,}

### Large Speculators (Hedge Funds)
- Long Contracts: {cot_data.get('large_spec_long', 0):,}
- Short Contracts: {cot_data.get('large_spec_short', 0):,}
- **Net Position**: {cot_data.get('large_spec_net', 0):,}

### Small Speculators (Retail)
- Long Contracts: {cot_data.get('small_spec_long', 0):,}
- Short Contracts: {cot_data.get('small_spec_short', 0):,}
- **Net Position**: {cot_data.get('small_spec_net', 0):,}

**COT Interpretation Guide**:
- Commercials often right (hedgers, know the market)
- Small speculators often wrong (retail, emotional)
- Extreme positioning = potential reversal

## Fear & Greed Index
- **Current Level**: {fear_greed}
- **Interpretation**:
  - 0-25: Extreme Fear (contrarian buy signal)
  - 25-45: Fear (potential buy)
  - 45-55: Neutral
  - 55-75: Greed (potential sell)
  - 75-100: Extreme Greed (contrarian sell signal)

## Your Task

Analyze all sentiment data and determine:

1. **News Sentiment**
   - Overall bias (bullish/bearish/neutral)
   - Key themes from headlines
   - Impact assessment (low/medium/high)

2. **Social Sentiment**
   - Crowd positioning (net long/short)
   - Sentiment score (-1.0 to +1.0)
   - Notable patterns (FOMO, panic, etc.)

3. **COT Analysis**
   - Who's positioned where?
   - Any extreme positioning?
   - Contrarian signals?

4. **Fear & Greed Assessment**
   - Current level
   - Historical context
   - Trading implications

5. **Sentiment Extremes**
   - Is sentiment at extreme?
   - Contrarian opportunity?
   - Crowd capitulation?

6. **Overall Assessment**
   - Sentiment bias
   - Contrarian bias (if different)
   - Recommended stance
   - Confidence level

7. **Summary**
   - 2-3 sentence summary with actionable insight

Output ONLY the JSON object with your comprehensive sentiment analysis."""

        return prompt

    def _format_news_headlines(self, headlines: List[Dict[str, Any]]) -> str:
        """Format news headlines for display"""
        if not headlines:
            return "No recent news headlines available"

        output = []
        for i, headline in enumerate(headlines[:10], 1):
            time = headline.get('time', 'N/A')
            source = headline.get('source', 'Unknown')
            text = headline.get('text', '')
            impact = headline.get('impact', 'medium')

            output.append(
                f"{i}. **[{source}]** ({time}) - {text}\n"
                f"   Impact: {impact.upper()}"
            )

        return "\n\n".join(output)
