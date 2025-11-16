#!/usr/bin/env python3
"""Test CoinGecko API integration and data collection."""

import os
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

# Load environment
load_dotenv()

def test_coingecko_api():
    """Test CoinGecko Pro API connection."""
    from apps.runtime.coingecko_fetcher import CoinGeckoFetcher

    api_key = os.getenv('COINGECKO_API_KEY')

    if not api_key:
        logger.error("❌ No CoinGecko API key found in .env")
        return False

    logger.info(f"Testing CoinGecko API with key: {api_key[:10]}...")

    try:
        fetcher = CoinGeckoFetcher(api_key)

        # Test all 3 symbols
        symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']

        for symbol in symbols:
            logger.info(f"\nTesting {symbol}...")

            features = fetcher.get_features(symbol)

            logger.info(f"✅ {symbol} CoinGecko features:")
            for key, value in features.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.2f}")
                else:
                    logger.info(f"  {key}: {value}")

        logger.info("\n" + "="*60)
        logger.info("🎉 CoinGecko API Test Summary:")
        logger.info("  ✅ API connection successful")
        logger.info("  ✅ All 3 symbols working")
        logger.info("  ✅ Fundamental features available")
        logger.info("="*60)

        return True

    except Exception as e:
        logger.error(f"❌ CoinGecko API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_coingecko_historical():
    """Test fetching historical CoinGecko data."""
    try:
        from pycoingecko import CoinGeckoAPI

        api_key = os.getenv('COINGECKO_API_KEY')

        # Initialize CoinGecko client
        cg = CoinGeckoAPI(api_key=api_key)

        logger.info("\nTesting CoinGecko historical data...")

        # Map symbols to CoinGecko IDs
        coin_map = {
            'BTC-USD': 'bitcoin',
            'ETH-USD': 'ethereum',
            'SOL-USD': 'solana'
        }

        for symbol, coin_id in coin_map.items():
            logger.info(f"\nFetching historical data for {symbol} ({coin_id})...")

            # Get market chart (7 days)
            data = cg.get_coin_market_chart_by_id(
                id=coin_id,
                vs_currency='usd',
                days=7
            )

            prices = data.get('prices', [])
            market_caps = data.get('market_caps', [])
            volumes = data.get('total_volumes', [])

            logger.info(f"✅ {symbol}:")
            logger.info(f"  Prices: {len(prices)} data points")
            logger.info(f"  Market caps: {len(market_caps)} data points")
            logger.info(f"  Volumes: {len(volumes)} data points")

            if prices:
                latest_price = prices[-1][1]
                logger.info(f"  Latest price: ${latest_price:,.2f}")

            if market_caps:
                latest_mcap = market_caps[-1][1]
                logger.info(f"  Market cap: ${latest_mcap:,.0f}")

        logger.info("\n✅ Historical data test complete!")
        return True

    except ImportError:
        logger.warning("pycoingecko not installed. Installing...")
        import subprocess
        subprocess.run(['pip3', 'install', 'pycoingecko'], check=True)
        logger.info("✅ Installed pycoingecko, please run test again")
        return False
    except Exception as e:
        logger.error(f"❌ Historical data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("CoinGecko Integration Test")
    logger.info("="*60)

    # Test current features API
    success1 = test_coingecko_api()

    # Test historical data API
    success2 = test_coingecko_historical()

    if success1 and success2:
        logger.info("\n🎉 All CoinGecko tests passed!")
        logger.info("Ready to integrate into V6 data collection")
    else:
        logger.warning("\n⚠️  Some tests failed, review errors above")
