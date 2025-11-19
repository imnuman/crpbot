#!/usr/bin/env python3
"""Test Kraken authenticated API connection."""

import os
import ccxt
from loguru import logger
from dotenv import load_dotenv

def test_kraken_private_api():
    """Test Kraken private API (requires authentication)."""

    # Load environment variables
    load_dotenv()

    api_key = os.getenv('KRAKEN_API_KEY')
    api_secret = os.getenv('KRAKEN_API_SECRET')

    if not api_key or not api_secret:
        logger.error("❌ Kraken API credentials not found in .env file")
        logger.info("Please add the following to your .env file:")
        logger.info("KRAKEN_API_KEY=your_api_key_here")
        logger.info("KRAKEN_API_SECRET=your_api_secret_here")
        return False

    logger.info("Testing Kraken authenticated API connection...")
    logger.info(f"API Key (first 10 chars): {api_key[:10]}...")

    try:
        # Initialize Kraken with authentication
        kraken = ccxt.kraken({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'timeout': 30000
        })

        # Test 1: Check account balance
        logger.info("Test 1: Fetching account balance...")
        balance = kraken.fetch_balance()

        logger.info("✅ Successfully connected to Kraken private API!")
        logger.info(f"Account balances (non-zero):")

        for currency, amount in balance['total'].items():
            if amount > 0:
                logger.info(f"  {currency}: {amount}")

        # Test 2: Check open orders
        logger.info("\nTest 2: Fetching open orders...")
        open_orders = kraken.fetch_open_orders()
        logger.info(f"✅ Open orders: {len(open_orders)}")

        # Test 3: Check trading fees
        logger.info("\nTest 3: Fetching trading fees...")
        try:
            # Fetch trading fee for BTC/USD
            markets = kraken.load_markets()
            if 'BTC/USD' in markets:
                market = markets['BTC/USD']
                logger.info(f"✅ BTC/USD trading fee:")
                logger.info(f"  Maker: {market.get('maker', 'N/A')}%")
                logger.info(f"  Taker: {market.get('taker', 'N/A')}%")
        except Exception as e:
            logger.warning(f"Could not fetch trading fees: {e}")

        # Test 4: Check API permissions
        logger.info("\nTest 4: Checking API permissions...")
        logger.info("✅ Your API key has the following permissions:")
        logger.info("  - Query Funds: YES (balance check successful)")
        logger.info("  - Query Open Orders: YES")

        logger.info("\n" + "="*60)
        logger.info("🎉 Kraken Private API Test Summary:")
        logger.info("  ✅ Authentication successful")
        logger.info("  ✅ Account balance accessible")
        logger.info("  ✅ Order queries working")
        logger.info("  ✅ Ready for authenticated data fetching")
        logger.info("="*60)

        return True

    except ccxt.AuthenticationError as e:
        logger.error(f"❌ Authentication failed: {e}")
        logger.error("Please verify your API key and secret are correct")
        return False

    except ccxt.PermissionDenied as e:
        logger.error(f"❌ Permission denied: {e}")
        logger.error("Your API key may not have the required permissions")
        logger.info("Required permissions: Query Funds, Query Open Orders")
        return False

    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False


def test_kraken_historical_data_with_auth():
    """Test fetching historical data with authenticated connection."""

    load_dotenv()

    api_key = os.getenv('KRAKEN_API_KEY')
    api_secret = os.getenv('KRAKEN_API_SECRET')

    if not api_key or not api_secret:
        logger.warning("Skipping authenticated data fetch test (no credentials)")
        return False

    try:
        kraken = ccxt.kraken({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'timeout': 30000
        })

        logger.info("\nTesting historical data fetch with authentication...")

        # Fetch recent OHLCV data
        symbol = 'BTC/USD'
        timeframe = '1h'
        limit = 10

        ohlcv = kraken.fetch_ohlcv(symbol, timeframe, limit=limit)

        if ohlcv:
            logger.info(f"✅ Successfully fetched {len(ohlcv)} candles for {symbol}")
            latest = ohlcv[-1]
            from datetime import datetime
            timestamp = datetime.fromtimestamp(latest[0] / 1000)
            logger.info(f"Latest candle: {timestamp} | Close: ${latest[4]:,.2f}")

        return True

    except Exception as e:
        logger.error(f"❌ Data fetch failed: {e}")
        return False


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("Kraken Authenticated API Test")
    logger.info("="*60)
    logger.info("")

    # Test private API
    success = test_kraken_private_api()

    if success:
        # Test historical data fetch
        test_kraken_historical_data_with_auth()

    logger.info("")
    logger.info("Test complete!")
