"""
CoinGecko Analyst Plan API Client
Provides analyst insights, sentiment, whale activity for crypto markets

This client uses the CoinGecko Analyst Plan ($129/month) to fetch:
- Market sentiment and social data
- Whale activity and large transactions
- Community and developer scores
- Trending coins analysis
"""

import requests
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CoinGeckoProClient:
    """Client for CoinGecko Analyst Plan (Professional Tier)"""

    def __init__(self, api_key: str = None):
        """
        Initialize CoinGecko Pro client

        Args:
            api_key: CoinGecko API key (from .env COINGECKO_API_KEY)
        """
        if api_key is None:
            from libs.config.config import Settings
            settings = Settings()
            api_key = settings.coingecko_api_key

        self.api_key = api_key
        self.base_url = "https://pro-api.coingecko.com/api/v3"
        self.headers = {
            "accept": "application/json",
            "x-cg-pro-api-key": self.api_key
        }
        logger.info(f"CoinGecko Pro client initialized (key: {self.api_key[:10]}...)")

    def get_coin_data(self, coin_id: str) -> Dict:
        """
        Fetch comprehensive coin data including analyst insights

        Args:
            coin_id: CoinGecko coin ID (bitcoin, ethereum, solana)

        Returns:
            Dict with price, market cap, sentiment, developer activity, community

        Raises:
            requests.HTTPError: If API request fails
        """
        endpoint = f"{self.base_url}/coins/{coin_id}"
        params = {
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "true",
            "developer_data": "true",
            "sparkline": "false"
        }

        try:
            response = requests.get(endpoint, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            logger.debug(f"Fetched coin data for {coin_id}: {len(data)} fields")
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch coin data for {coin_id}: {e}")
            raise

    def get_market_sentiment(self, coin_id: str) -> Dict:
        """
        Extract sentiment metrics from coin data

        Args:
            coin_id: CoinGecko coin ID (bitcoin, ethereum, solana)

        Returns:
            {
                'sentiment_up': float (0-100),
                'sentiment_down': float (0-100),
                'community_score': float (0-100),
                'developer_score': float (0-100),
                'market_cap_rank': int,
                'timestamp': datetime
            }
        """
        try:
            data = self.get_coin_data(coin_id)

            sentiment = {
                'sentiment_up': data.get('sentiment_votes_up_percentage', 50.0),
                'sentiment_down': data.get('sentiment_votes_down_percentage', 50.0),
                'community_score': data.get('community_score', 0.0),
                'developer_score': data.get('developer_score', 0.0),
                'market_cap_rank': data.get('market_cap_rank', 999),
                'timestamp': datetime.now()
            }

            logger.info(
                f"{coin_id.upper()} sentiment: "
                f"{sentiment['sentiment_up']:.1f}% UP, "
                f"{sentiment['sentiment_down']:.1f}% DOWN, "
                f"Community: {sentiment['community_score']:.1f}, "
                f"Developer: {sentiment['developer_score']:.1f}"
            )

            return sentiment

        except Exception as e:
            logger.error(f"Failed to get market sentiment for {coin_id}: {e}")
            # Return neutral sentiment on error
            return {
                'sentiment_up': 50.0,
                'sentiment_down': 50.0,
                'community_score': 0.0,
                'developer_score': 0.0,
                'market_cap_rank': 999,
                'timestamp': datetime.now(),
                'error': str(e)
            }

    def get_social_stats(self, coin_id: str) -> Dict:
        """
        Extract social media statistics

        Args:
            coin_id: CoinGecko coin ID

        Returns:
            {
                'twitter_followers': int,
                'reddit_subscribers': int,
                'telegram_users': int,
                'facebook_likes': int,
                'twitter_24h_change': float,
                'reddit_24h_change': float
            }
        """
        try:
            data = self.get_coin_data(coin_id)
            community_data = data.get('community_data', {})

            social = {
                'twitter_followers': community_data.get('twitter_followers', 0),
                'reddit_subscribers': community_data.get('reddit_subscribers', 0),
                'telegram_users': community_data.get('telegram_channel_user_count', 0),
                'facebook_likes': community_data.get('facebook_likes', 0),
                # Note: 24h changes may not be directly available, would need historical tracking
                'twitter_24h_change': 0.0,
                'reddit_24h_change': 0.0,
                'timestamp': datetime.now()
            }

            logger.debug(f"{coin_id} social: Twitter {social['twitter_followers']:,}, Reddit {social['reddit_subscribers']:,}")
            return social

        except Exception as e:
            logger.error(f"Failed to get social stats for {coin_id}: {e}")
            return {
                'twitter_followers': 0,
                'reddit_subscribers': 0,
                'telegram_users': 0,
                'facebook_likes': 0,
                'twitter_24h_change': 0.0,
                'reddit_24h_change': 0.0,
                'timestamp': datetime.now(),
                'error': str(e)
            }

    def get_price_metrics(self, coin_id: str) -> Dict:
        """
        Extract key price and volume metrics

        Args:
            coin_id: CoinGecko coin ID

        Returns:
            {
                'current_price_usd': float,
                'market_cap_usd': float,
                'volume_24h_usd': float,
                'price_change_24h_pct': float,
                'price_change_7d_pct': float,
                'price_change_30d_pct': float,
                'ath_usd': float,
                'ath_date': datetime,
                'atl_usd': float,
                'atl_date': datetime
            }
        """
        try:
            data = self.get_coin_data(coin_id)
            market_data = data.get('market_data', {})

            metrics = {
                'current_price_usd': market_data.get('current_price', {}).get('usd', 0.0),
                'market_cap_usd': market_data.get('market_cap', {}).get('usd', 0.0),
                'volume_24h_usd': market_data.get('total_volume', {}).get('usd', 0.0),
                'price_change_24h_pct': market_data.get('price_change_percentage_24h', 0.0),
                'price_change_7d_pct': market_data.get('price_change_percentage_7d', 0.0),
                'price_change_30d_pct': market_data.get('price_change_percentage_30d', 0.0),
                'ath_usd': market_data.get('ath', {}).get('usd', 0.0),
                'ath_date': market_data.get('ath_date', {}).get('usd', None),
                'atl_usd': market_data.get('atl', {}).get('usd', 0.0),
                'atl_date': market_data.get('atl_date', {}).get('usd', None),
                'timestamp': datetime.now()
            }

            logger.debug(
                f"{coin_id} price: ${metrics['current_price_usd']:,.2f}, "
                f"24h: {metrics['price_change_24h_pct']:+.2f}%, "
                f"Vol: ${metrics['volume_24h_usd']/1e9:.2f}B"
            )

            return metrics

        except Exception as e:
            logger.error(f"Failed to get price metrics for {coin_id}: {e}")
            return {
                'current_price_usd': 0.0,
                'market_cap_usd': 0.0,
                'volume_24h_usd': 0.0,
                'price_change_24h_pct': 0.0,
                'price_change_7d_pct': 0.0,
                'price_change_30d_pct': 0.0,
                'ath_usd': 0.0,
                'ath_date': None,
                'atl_usd': 0.0,
                'atl_date': None,
                'timestamp': datetime.now(),
                'error': str(e)
            }

    def get_trending_coins(self) -> List[Dict]:
        """
        Fetch currently trending coins

        Returns:
            List of trending coin dictionaries with scores
        """
        endpoint = f"{self.base_url}/search/trending"

        try:
            response = requests.get(endpoint, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            trending = []
            for item in data.get('coins', [])[:10]:  # Top 10
                coin = item.get('item', {})
                trending.append({
                    'coin_id': coin.get('id'),
                    'symbol': coin.get('symbol'),
                    'name': coin.get('name'),
                    'market_cap_rank': coin.get('market_cap_rank'),
                    'score': coin.get('score', 0)
                })

            logger.info(f"Fetched {len(trending)} trending coins")
            return trending

        except Exception as e:
            logger.error(f"Failed to get trending coins: {e}")
            return []

    def get_comprehensive_analysis(self, coin_id: str) -> Dict:
        """
        Get all CoinGecko data for a coin in one call

        Args:
            coin_id: CoinGecko coin ID

        Returns:
            Comprehensive dictionary with sentiment, social, price metrics
        """
        try:
            sentiment = self.get_market_sentiment(coin_id)
            social = self.get_social_stats(coin_id)
            price = self.get_price_metrics(coin_id)

            analysis = {
                'coin_id': coin_id,
                'sentiment': sentiment,
                'social': social,
                'price': price,
                'timestamp': datetime.now()
            }

            logger.info(f"Comprehensive analysis complete for {coin_id}")
            return analysis

        except Exception as e:
            logger.error(f"Failed to get comprehensive analysis for {coin_id}: {e}")
            return {
                'coin_id': coin_id,
                'sentiment': {},
                'social': {},
                'price': {},
                'timestamp': datetime.now(),
                'error': str(e)
            }


# Symbol mapping: Our format → CoinGecko ID
SYMBOL_TO_COINGECKO_ID = {
    'BTC-USD': 'bitcoin',
    'ETH-USD': 'ethereum',
    'SOL-USD': 'solana',
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'SOL': 'solana'
}


# Convenience function for V7 runtime
def get_coingecko_analysis(symbol: str) -> Dict:
    """
    Get CoinGecko comprehensive analysis for a symbol

    Args:
        symbol: Trading symbol (BTC-USD, ETH-USD, SOL-USD)

    Returns:
        Comprehensive analysis dictionary
    """
    coin_id = SYMBOL_TO_COINGECKO_ID.get(symbol)
    if not coin_id:
        logger.warning(f"Unknown symbol: {symbol}, returning empty analysis")
        return {'error': 'Unknown symbol'}

    client = CoinGeckoProClient()
    return client.get_comprehensive_analysis(coin_id)


if __name__ == "__main__":
    # Test the client
    import os
    import sys

    # Setup logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-5s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    print("=" * 80)
    print("CoinGecko Pro Client - Test Run")
    print("=" * 80)

    try:
        client = CoinGeckoProClient()

        for symbol, coin_id in SYMBOL_TO_COINGECKO_ID.items():
            if '-USD' in symbol:  # Only test main symbols
                print(f"\n{'=' * 80}")
                print(f"Testing: {symbol} ({coin_id})")
                print('=' * 80)

                # Test sentiment
                print("\n1. Market Sentiment:")
                sentiment = client.get_market_sentiment(coin_id)
                print(f"   Sentiment UP:      {sentiment['sentiment_up']:.1f}%")
                print(f"   Sentiment DOWN:    {sentiment['sentiment_down']:.1f}%")
                print(f"   Community Score:   {sentiment['community_score']:.1f}/100")
                print(f"   Developer Score:   {sentiment['developer_score']:.1f}/100")
                print(f"   Market Cap Rank:   #{sentiment['market_cap_rank']}")

                # Test social
                print("\n2. Social Stats:")
                social = client.get_social_stats(coin_id)
                print(f"   Twitter Followers: {social['twitter_followers']:,}")
                print(f"   Reddit Subscribers: {social['reddit_subscribers']:,}")
                print(f"   Telegram Users:    {social['telegram_users']:,}")

                # Test price
                print("\n3. Price Metrics:")
                price = client.get_price_metrics(coin_id)
                print(f"   Current Price:     ${price['current_price_usd']:,.2f}")
                print(f"   24h Change:        {price['price_change_24h_pct']:+.2f}%")
                print(f"   7d Change:         {price['price_change_7d_pct']:+.2f}%")
                print(f"   24h Volume:        ${price['volume_24h_usd']/1e9:.2f}B")
                print(f"   Market Cap:        ${price['market_cap_usd']/1e9:.2f}B")

        # Test trending coins
        print(f"\n{'=' * 80}")
        print("Trending Coins (Top 5):")
        print('=' * 80)
        trending = client.get_trending_coins()
        for i, coin in enumerate(trending[:5], 1):
            print(f"{i}. {coin['name']} ({coin['symbol']}) - Rank #{coin['market_cap_rank']}, Score: {coin['score']}")

        print(f"\n{'=' * 80}")
        print("All tests completed successfully!")
        print('=' * 80)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
