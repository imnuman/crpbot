"""
CoinGecko Analyst API Client
Fetches real-time market data from CoinGecko Professional API ($129/month).
"""
import requests
from typing import Dict, Optional
from loguru import logger


class CoinGeckoClient:
    """Client for CoinGecko Analyst Plan (Professional Tier)"""

    # Symbol mapping: Coinbase format → CoinGecko ID
    SYMBOL_MAP = {
        "BTC-USD": "bitcoin",
        "ETH-USD": "ethereum",
        "SOL-USD": "solana"
    }

    def __init__(self, api_key: str):
        """
        Initialize CoinGecko client.

        Args:
            api_key: CoinGecko API key (from .env)
        """
        self.api_key = api_key
        self.base_url = "https://api.coingecko.com/api/v3"
        self.headers = {
            "x-cg-pro-api-key": api_key,
            "Accept": "application/json"
        }

    def get_market_data(self, symbol: str) -> Optional[Dict]:
        """
        Fetch comprehensive market data for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTC-USD")

        Returns:
            Dictionary with market data:
            {
                'market_cap': float,           # Total market capitalization
                'total_volume': float,         # 24h trading volume
                'price_change_24h_pct': float, # 24h price change %
                'ath': float,                  # All-time high price
                'ath_distance_pct': float,     # Distance from ATH (negative = below ATH)
                'circulating_supply': float,   # Coins in circulation
                'max_supply': float,           # Maximum supply (if applicable)
                'market_cap_rank': int         # Rank by market cap
            }
        """
        try:
            # Convert Coinbase symbol to CoinGecko ID
            coin_id = self.SYMBOL_MAP.get(symbol)
            if not coin_id:
                logger.warning(f"Unknown symbol: {symbol}, skipping CoinGecko")
                return None

            # API endpoint: /coins/{id}
            url = f"{self.base_url}/coins/{coin_id}"
            params = {
                "localization": "false",
                "tickers": "false",
                "market_data": "true",
                "community_data": "false",
                "developer_data": "false",
                "sparkline": "false"
            }

            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Extract market data
            market_data = data.get("market_data", {})
            current_price = market_data.get("current_price", {}).get("usd", 0)
            ath = market_data.get("ath", {}).get("usd", 0)

            result = {
                "market_cap": market_data.get("market_cap", {}).get("usd", 0),
                "total_volume": market_data.get("total_volume", {}).get("usd", 0),
                "price_change_24h_pct": market_data.get("price_change_percentage_24h", 0),
                "ath": ath,
                "ath_distance_pct": market_data.get("ath_change_percentage", {}).get("usd", 0),
                "circulating_supply": market_data.get("circulating_supply", 0),
                "max_supply": market_data.get("max_supply", 0),
                "market_cap_rank": data.get("market_cap_rank", 0),
                "current_price_usd": current_price
            }

            logger.info(f"CoinGecko data fetched for {symbol}: "
                       f"MCap ${result['market_cap']/1e9:.1f}B, "
                       f"Vol ${result['total_volume']/1e9:.2f}B, "
                       f"ATH dist {result['ath_distance_pct']:.1f}%")

            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"CoinGecko API error for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing CoinGecko data for {symbol}: {e}")
            return None

    def get_simple_price(self, symbol: str) -> Optional[float]:
        """
        Quick price fetch (lighter endpoint).

        Args:
            symbol: Trading pair (e.g., "BTC-USD")

        Returns:
            Current price in USD
        """
        try:
            coin_id = self.SYMBOL_MAP.get(symbol)
            if not coin_id:
                return None

            url = f"{self.base_url}/simple/price"
            params = {
                "ids": coin_id,
                "vs_currencies": "usd"
            }

            response = requests.get(url, headers=self.headers, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()

            price = data.get(coin_id, {}).get("usd")
            return float(price) if price else None

        except Exception as e:
            logger.error(f"Error fetching CoinGecko price for {symbol}: {e}")
            return None
