"""
Real-time CoinGecko data fetcher for runtime feature generation.

Fetches live market data (market cap, volume, ATH) with 5-minute caching
to avoid rate limiting during high-frequency trading signal generation.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Dict, Optional

import requests
from loguru import logger

# CoinGecko configuration
COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY')
COINGECKO_BASE_URL = 'https://api.coingecko.com/api/v3'

# Coin ID mapping
COIN_IDS = {
    'BTC-USD': 'bitcoin',
    'ETH-USD': 'ethereum',
    'SOL-USD': 'solana'
}

# Cache duration (5 minutes - CoinGecko data doesn't change rapidly)
CACHE_DURATION_SECONDS = 300


class CoinGeckoFetcher:
    """
    Fetch real-time CoinGecko market data with caching.

    Usage:
        fetcher = CoinGeckoFetcher()
        data = fetcher.get_market_data('BTC-USD')
        features = fetcher.calculate_features(data)
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize CoinGecko fetcher.

        Args:
            api_key: CoinGecko API key (defaults to COINGECKO_API_KEY env var)
        """
        self.api_key = api_key or COINGECKO_API_KEY
        self.cache: Dict[str, tuple[dict, datetime]] = {}  # symbol -> (data, timestamp)

        if not self.api_key:
            logger.warning("⚠️  No CoinGecko API key - will use placeholders")

    def get_market_data(self, symbol: str) -> Optional[dict]:
        """
        Fetch current market data for a symbol.

        Uses 5-minute cache to avoid hitting rate limits.

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')

        Returns:
            dict: Market data or None if fetch fails
        """
        # Check cache first
        if symbol in self.cache:
            cached_data, cached_time = self.cache[symbol]
            age = (datetime.now() - cached_time).total_seconds()

            if age < CACHE_DURATION_SECONDS:
                logger.debug(f"Using cached CoinGecko data for {symbol} (age: {age:.0f}s)")
                return cached_data

        # Map symbol to CoinGecko coin ID
        coin_id = COIN_IDS.get(symbol)
        if not coin_id:
            logger.error(f"Unknown symbol for CoinGecko: {symbol}")
            return None

        # Fetch fresh data
        logger.info(f"Fetching fresh CoinGecko data for {symbol}")

        url = f"{COINGECKO_BASE_URL}/coins/{coin_id}"

        headers = {
            'accept': 'application/json',
            'x-cg-pro-api-key': self.api_key
        }

        params = {
            'localization': 'false',
            'tickers': 'false',
            'market_data': 'true',
            'community_data': 'false',
            'developer_data': 'false',
            'sparkline': 'false'
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            market_data = data.get('market_data', {})

            # Extract relevant fields
            result = {
                'market_cap_usd': market_data.get('market_cap', {}).get('usd', 0),
                'total_volume_usd': market_data.get('total_volume', {}).get('usd', 0),
                'price_usd': market_data.get('current_price', {}).get('usd', 0),
                'price_change_24h_pct': market_data.get('price_change_percentage_24h', 0),
                'price_change_7d_pct': market_data.get('price_change_percentage_7d', 0),
                'market_cap_change_24h_pct': market_data.get('market_cap_change_percentage_24h', 0),
                'ath_usd': market_data.get('ath', {}).get('usd', 0),
                'ath_date': market_data.get('ath_date', {}).get('usd', ''),
                'circulating_supply': market_data.get('circulating_supply', 0),
                'total_supply': market_data.get('total_supply', 0),
            }

            # Cache result
            self.cache[symbol] = (result, datetime.now())

            logger.info(f"✅ Fetched CoinGecko data for {symbol} "
                       f"(market_cap: ${result['market_cap_usd']:,.0f}, "
                       f"price: ${result['price_usd']:,.2f})")

            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch CoinGecko data for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching CoinGecko data: {e}")
            return None

    def calculate_features(self, data: Optional[dict]) -> dict:
        """
        Calculate CoinGecko features from market data.

        Returns 10 features matching the placeholder structure:
        - ath_date (days since ATH)
        - market_cap_change_pct
        - volume_change_pct (placeholder - needs historical data)
        - price_change_pct (24h)
        - ath_distance_pct
        - market_cap_7d_ma (placeholder - needs historical data)
        - market_cap_30d_ma (placeholder - needs historical data)
        - market_cap_change_7d_pct (placeholder - needs historical data)
        - market_cap_trend (placeholder - needs historical data)
        - volume_change_7d_pct (placeholder - needs historical data)

        Args:
            data: Market data dict from get_market_data()

        Returns:
            dict: Feature name -> value
        """
        if not data:
            # Return zeros if no data (same as placeholders)
            return {
                'ath_date': 0,
                'market_cap_change_pct': 0.0,
                'volume_change_pct': 0.0,
                'price_change_pct': 0.0,
                'ath_distance_pct': -50.0,
                'market_cap_7d_ma': 0.0,
                'market_cap_30d_ma': 0.0,
                'market_cap_change_7d_pct': 0.0,
                'market_cap_trend': 0.0,
                'volume_change_7d_pct': 0.0,
            }

        features = {}

        # ATH date (days since ATH)
        if data.get('ath_date'):
            try:
                ath_date = datetime.fromisoformat(data['ath_date'].replace('Z', '+00:00'))
                days_since_ath = (datetime.now(ath_date.tzinfo) - ath_date).days
                features['ath_date'] = days_since_ath
            except Exception:
                features['ath_date'] = 0
        else:
            features['ath_date'] = 0

        # Market cap change (24h)
        features['market_cap_change_pct'] = data.get('market_cap_change_24h_pct', 0.0)

        # Volume change (not available in single API call - use placeholder)
        features['volume_change_pct'] = 0.0  # TODO: needs historical data

        # Price change (24h)
        features['price_change_pct'] = data.get('price_change_24h_pct', 0.0)

        # ATH distance percentage
        ath_usd = data.get('ath_usd', 0)
        price_usd = data.get('price_usd', 0)

        if ath_usd > 0 and price_usd > 0:
            features['ath_distance_pct'] = ((price_usd - ath_usd) / ath_usd) * 100
        else:
            features['ath_distance_pct'] = -50.0  # Default assumption

        # Moving averages and trends (not available in single API call - use placeholders)
        # These would require historical data or additional API calls
        features['market_cap_7d_ma'] = 0.0  # TODO: needs historical data
        features['market_cap_30d_ma'] = 0.0  # TODO: needs historical data
        features['market_cap_change_7d_pct'] = 0.0  # TODO: needs historical data
        features['market_cap_trend'] = 0.0  # TODO: needs historical data
        features['volume_change_7d_pct'] = 0.0  # TODO: needs historical data

        logger.debug(f"Calculated CoinGecko features: "
                    f"ath_distance={features['ath_distance_pct']:.1f}%, "
                    f"price_change_24h={features['price_change_pct']:.2f}%")

        return features

    def get_features(self, symbol: str) -> dict:
        """
        Convenience method: fetch data and calculate features in one call.

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')

        Returns:
            dict: Feature name -> value
        """
        data = self.get_market_data(symbol)
        return self.calculate_features(data)

    def clear_cache(self):
        """Clear cached data (for testing or forced refresh)."""
        self.cache.clear()
        logger.info("Cleared CoinGecko cache")
