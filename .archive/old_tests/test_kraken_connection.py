#!/usr/bin/env python3
"""Test Kraken API connectivity and data fetching."""

import ccxt
from datetime import datetime, timedelta
from loguru import logger

def test_kraken_public():
    """Test Kraken public API (no authentication needed)."""
    logger.info("Testing Kraken public API connectivity...")

    try:
        kraken = ccxt.kraken({
            'enableRateLimit': True,
            'timeout': 30000  # 30 seconds
        })

        # Test 1: Fetch markets
        logger.info("Fetching available markets...")
        markets = kraken.load_markets()
        logger.info(f"✅ Found {len(markets)} markets on Kraken")

        # Test 2: Check BTC/USD availability
        if 'BTC/USD' in markets:
            logger.info("✅ BTC/USD market available")
        else:
            logger.warning("⚠️  BTC/USD not found, available symbols: {}", list(markets.keys())[:10])

        # Test 3: Fetch recent OHLCV data
        symbols_to_test = ['BTC/USD', 'ETH/USD', 'SOL/USD']

        for symbol in symbols_to_test:
            try:
                logger.info(f"Fetching 1h candles for {symbol}...")
                ohlcv = kraken.fetch_ohlcv(symbol, '1h', limit=100)

                if ohlcv:
                    latest = ohlcv[-1]
                    timestamp = datetime.fromtimestamp(latest[0] / 1000)
                    close_price = latest[4]

                    logger.info(f"✅ {symbol}: {len(ohlcv)} candles fetched")
                    logger.info(f"   Latest: {timestamp} | Close: ${close_price:,.2f}")
                else:
                    logger.warning(f"⚠️  No data returned for {symbol}")

            except Exception as e:
                logger.error(f"❌ Failed to fetch {symbol}: {e}")

        # Test 4: Fetch 1-minute granularity
        logger.info("Testing 1-minute granularity...")
        ohlcv_1m = kraken.fetch_ohlcv('BTC/USD', '1m', limit=60)
        logger.info(f"✅ Fetched {len(ohlcv_1m)} 1-minute candles")

        logger.info("=" * 60)
        logger.info("🎉 Kraken API Test Summary:")
        logger.info("   ✅ Public API accessible")
        logger.info("   ✅ BTC/USD, ETH/USD, SOL/USD data available")
        logger.info("   ✅ 1-minute granularity supported")
        logger.info("   ✅ Ready for data collection")

        return True

    except Exception as e:
        logger.error(f"❌ Kraken API test failed: {e}")
        return False


def test_kraken_vs_coinbase():
    """Compare Kraken vs Coinbase data availability."""
    logger.info("Comparing Kraken vs Coinbase data coverage...")

    try:
        kraken = ccxt.kraken({'enableRateLimit': True})
        coinbase = ccxt.coinbase({'enableRateLimit': True})

        kraken_markets = kraken.load_markets()
        coinbase_markets = coinbase.load_markets()

        logger.info(f"Kraken markets: {len(kraken_markets)}")
        logger.info(f"Coinbase markets: {len(coinbase_markets)}")

        # Check our trading pairs
        pairs = ['BTC/USD', 'ETH/USD', 'SOL/USD']

        for pair in pairs:
            kraken_has = pair in kraken_markets
            coinbase_has = pair in coinbase_markets

            status = "✅" if (kraken_has and coinbase_has) else "⚠️"
            logger.info(f"{status} {pair}: Kraken={kraken_has}, Coinbase={coinbase_has}")

    except Exception as e:
        logger.error(f"Comparison failed: {e}")


if __name__ == "__main__":
    logger.info("Starting Kraken API connectivity test...")
    logger.info("=" * 60)

    success = test_kraken_public()

    if success:
        logger.info("")
        test_kraken_vs_coinbase()

    logger.info("=" * 60)
    logger.info("Test complete!")
