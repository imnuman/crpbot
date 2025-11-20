"""
CryptoCompare News API Client

Fetches crypto news with sentiment scoring.
News sentiment can be a leading indicator for price moves.

API: https://min-api.cryptocompare.com/
Cost: FREE tier (100 calls/day) or Paid ($30-300/month)
"""

import requests
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from loguru import logger
import os


class CryptoCompareClient:
    """Client for CryptoCompare News API"""

    BASE_URL = "https://min-api.cryptocompare.com/data/v2"

    # Symbol mapping
    SYMBOL_MAP = {
        'BTC-USD': 'BTC',
        'ETH-USD': 'ETH',
        'SOL-USD': 'SOL'
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize CryptoCompare client

        Args:
            api_key: CryptoCompare API key (or set CRYPTOCOMPARE_API_KEY env var)
                    Free tier: 100 calls/day
        """
        self.api_key = api_key or os.getenv('CRYPTOCOMPARE_API_KEY')
        if not self.api_key:
            logger.warning(
                "No CryptoCompare API key provided. Using free tier (limited). "
                "Get one at https://www.cryptocompare.com/cryptopian/api-keys"
            )

        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({'authorization': f'Apikey {self.api_key}'})

        logger.debug("CryptoCompare API client initialized")

    def _convert_symbol(self, symbol: str) -> str:
        """Convert our symbol format to CryptoCompare format"""
        return self.SYMBOL_MAP.get(symbol, symbol.split('-')[0])

    def get_latest_news(self, symbol: Optional[str] = None, limit: int = 10) -> Optional[List[Dict[str, Any]]]:
        """
        Get latest crypto news

        Args:
            symbol: Specific coin (e.g., 'BTC-USD') or None for general crypto news
            limit: Number of articles (max 100)

        Returns:
            List of news articles with sentiment
        """
        try:
            url = f"{self.BASE_URL}/news/"
            params = {}

            if symbol:
                coin = self._convert_symbol(symbol)
                params['categories'] = coin

            # CryptoCompare uses lang parameter
            params['lang'] = 'EN'

            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()

            data = response.json()

            if data.get('Response') == 'Error':
                logger.error(f"CryptoCompare API error: {data.get('Message')}")
                return None

            if 'Data' not in data:
                logger.error("No data in CryptoCompare response")
                return None

            articles = []
            for item in data['Data'][:limit]:
                articles.append({
                    'id': item.get('id'),
                    'title': item.get('title'),
                    'body': item.get('body', '')[:500],  # First 500 chars
                    'published_on': datetime.fromtimestamp(item.get('published_on', 0)),
                    'source': item.get('source'),
                    'url': item.get('url'),
                    'tags': item.get('tags', '').split('|') if item.get('tags') else [],
                    'categories': item.get('categories', '').split('|') if item.get('categories') else []
                })

            logger.info(f"Fetched {len(articles)} news articles" + (f" for {symbol}" if symbol else ""))

            return articles

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching CryptoCompare news: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing CryptoCompare news: {e}")
            return None

    def analyze_news_sentiment(self, symbol: str, hours: int = 24) -> Optional[Dict[str, Any]]:
        """
        Analyze news sentiment for a symbol over recent hours

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            hours: Hours to look back

        Returns:
            {
                'symbol': str,
                'article_count': int,
                'sentiment_score': float (-1 to +1),
                'sentiment': str (bullish, bearish, neutral),
                'recent_headlines': list,
                'interpretation': str
            }
        """
        articles = self.get_latest_news(symbol, limit=50)
        if not articles:
            return None

        # Filter to recent hours
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [a for a in articles if a['published_on'] > cutoff]

        if len(recent) == 0:
            logger.warning(f"No recent news for {symbol} in last {hours} hours")
            return None

        # Simple sentiment analysis based on keywords
        # (In production, you'd use the CryptoCompare sentiment API or NLP)
        sentiment_score = self._calculate_sentiment_from_headlines(recent)

        # Determine sentiment
        if sentiment_score > 0.2:
            sentiment = "bullish"
            interpretation = (
                f"POSITIVE NEWS SENTIMENT ({sentiment_score:+.2f}) - "
                f"{len(recent)} bullish articles in {hours}h. "
                "News could support price rally."
            )
        elif sentiment_score < -0.2:
            sentiment = "bearish"
            interpretation = (
                f"NEGATIVE NEWS SENTIMENT ({sentiment_score:+.2f}) - "
                f"{len(recent)} bearish articles in {hours}h. "
                "News could pressure price lower."
            )
        else:
            sentiment = "neutral"
            interpretation = (
                f"NEUTRAL NEWS SENTIMENT ({sentiment_score:+.2f}) - "
                f"{len(recent)} mixed articles in {hours}h."
            )

        recent_headlines = [
            {
                'title': a['title'],
                'source': a['source'],
                'time': a['published_on'].strftime('%H:%M')
            }
            for a in recent[:5]
        ]

        logger.info(f"{symbol} News Sentiment: {sentiment.upper()} ({sentiment_score:+.2f})")

        return {
            'symbol': symbol,
            'article_count': len(recent),
            'sentiment_score': sentiment_score,
            'sentiment': sentiment,
            'recent_headlines': recent_headlines,
            'interpretation': interpretation
        }

    def _calculate_sentiment_from_headlines(self, articles: List[Dict]) -> float:
        """
        Calculate sentiment score from headlines using keyword analysis

        This is a simple approach. CryptoCompare has a dedicated sentiment API
        that provides more sophisticated analysis.

        Returns:
            Sentiment score (-1 to +1)
        """
        bullish_keywords = [
            'surge', 'rally', 'bull', 'gain', 'rise', 'soar', 'climb',
            'breakout', 'pump', 'moon', 'bullish', 'positive', 'upgrade',
            'adoption', 'partnership', 'institutional', 'accumulation'
        ]
        bearish_keywords = [
            'crash', 'dump', 'bear', 'drop', 'fall', 'plunge', 'decline',
            'breakdown', 'sell', 'selloff', 'bearish', 'negative', 'downgrade',
            'regulation', 'ban', 'hack', 'scam', 'lawsuit', 'fear'
        ]

        total_score = 0
        for article in articles:
            text = (article['title'] + ' ' + article['body']).lower()

            bullish_count = sum(1 for word in bullish_keywords if word in text)
            bearish_count = sum(1 for word in bearish_keywords if word in text)

            # Score: +1 for bullish, -1 for bearish, 0 for neutral
            if bullish_count > bearish_count:
                total_score += 1
            elif bearish_count > bullish_count:
                total_score -= 1

        # Normalize to -1 to +1 range
        if len(articles) > 0:
            return total_score / len(articles)
        else:
            return 0

    def get_social_stats(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get social media stats for a symbol

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')

        Returns:
            Social media metrics (if available in paid tier)
        """
        try:
            coin = self._convert_symbol(symbol)
            url = f"{self.BASE_URL}/social/coin/latest"

            response = self.session.get(
                url,
                params={'coinId': self._get_coin_id(coin)},
                timeout=15
            )

            # This endpoint requires paid tier, might fail on free tier
            if response.status_code == 401 or response.status_code == 403:
                logger.debug(f"Social stats not available (requires paid tier)")
                return None

            response.raise_for_status()
            data = response.json()

            if data.get('Response') == 'Error':
                return None

            # Parse social data (structure depends on CryptoCompare response)
            return data.get('Data', {})

        except Exception as e:
            logger.debug(f"Could not fetch social stats: {e}")
            return None

    def _get_coin_id(self, coin: str) -> int:
        """Map coin symbol to CryptoCompare coin ID (simplified)"""
        # In production, you'd fetch this from the API
        coin_ids = {
            'BTC': 1182,
            'ETH': 7605,
            'SOL': 898822
        }
        return coin_ids.get(coin, 0)


# Test the client if run directly
if __name__ == "__main__":
    client = CryptoCompareClient()

    for symbol in ['BTC-USD', 'ETH-USD', 'SOL-USD']:
        print(f"\n{'='*80}")
        print(f"CRYPTOCOMPARE NEWS ANALYSIS: {symbol}")
        print(f"{'='*80}")

        sentiment = client.analyze_news_sentiment(symbol, hours=24)
        if sentiment:
            print(f"\n📰 News Sentiment (24h):")
            print(f"   Articles: {sentiment['article_count']}")
            print(f"   Sentiment: {sentiment['sentiment'].upper()}")
            print(f"   Score: {sentiment['sentiment_score']:+.2f}")
            print(f"   {sentiment['interpretation']}")

            print(f"\n📋 Recent Headlines:")
            for i, headline in enumerate(sentiment['recent_headlines'], 1):
                print(f"   {i}. [{headline['time']}] {headline['title'][:80]}...")
                print(f"      Source: {headline['source']}")
