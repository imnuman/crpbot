#!/usr/bin/env python3
"""
Test CoinGecko Premium API integration.

Verifies that we can fetch real-time data and that features
are no longer placeholders (all zeros).
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from apps.runtime.coingecko_fetcher import CoinGeckoFetcher
from loguru import logger


def test_api_key():
    """Test that API key is available."""
    api_key = os.getenv('COINGECKO_API_KEY')

    if not api_key:
        logger.error("❌ COINGECKO_API_KEY not found in environment")
        logger.info("Export it: export COINGECKO_API_KEY=CG-VQhq64e59sGxchtK8mRgdxXW")
        return False

    logger.info(f"✅ API Key found: {api_key[:10]}...")
    return True


def test_fetch_market_data():
    """Test fetching market data for all symbols."""
    logger.info("\n📊 Testing Market Data Fetch")
    logger.info("=" * 60)

    fetcher = CoinGeckoFetcher()
    symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']

    results = {}

    for symbol in symbols:
        logger.info(f"\nFetching {symbol}...")
        data = fetcher.get_market_data(symbol)

        if not data:
            logger.error(f"❌ Failed to fetch {symbol}")
            results[symbol] = False
            continue

        logger.info(f"✅ {symbol} data:")
        logger.info(f"   Market Cap: ${data['market_cap_usd']:,.0f}")
        logger.info(f"   Price: ${data['price_usd']:,.2f}")
        logger.info(f"   24h Change: {data['price_change_24h_pct']:.2f}%")
        logger.info(f"   ATH: ${data['ath_usd']:,.2f}")
        logger.info(f"   Volume: ${data['total_volume_usd']:,.0f}")

        results[symbol] = True

    logger.info("\n" + "=" * 60)
    success_count = sum(results.values())
    logger.info(f"Results: {success_count}/{len(symbols)} successful")

    return all(results.values())


def test_calculate_features():
    """Test feature calculation from market data."""
    logger.info("\n🔢 Testing Feature Calculation")
    logger.info("=" * 60)

    fetcher = CoinGeckoFetcher()
    symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']

    all_passed = True

    for symbol in symbols:
        logger.info(f"\nCalculating features for {symbol}...")
        features = fetcher.get_features(symbol)

        if not features:
            logger.error(f"❌ Failed to calculate features for {symbol}")
            all_passed = False
            continue

        logger.info(f"✅ {symbol} features:")
        logger.info(f"   ath_date: {features['ath_date']} days")
        logger.info(f"   market_cap_change_pct: {features['market_cap_change_pct']:.2f}%")
        logger.info(f"   price_change_pct: {features['price_change_pct']:.2f}%")
        logger.info(f"   ath_distance_pct: {features['ath_distance_pct']:.2f}%")

        # Check that features are NOT all zeros (i.e., not placeholders)
        non_zero_count = sum(1 for v in features.values() if v != 0.0)

        if non_zero_count == 0:
            logger.error(f"❌ All features are zero (placeholders) for {symbol}")
            all_passed = False
        else:
            logger.info(f"   ✅ {non_zero_count}/{len(features)} features have non-zero values")

    logger.info("\n" + "=" * 60)
    return all_passed


def test_cache_behavior():
    """Test that caching works correctly."""
    logger.info("\n⏱️  Testing Cache Behavior")
    logger.info("=" * 60)

    fetcher = CoinGeckoFetcher()

    # First fetch (should hit API)
    logger.info("\n1st fetch (should hit API)...")
    import time
    start = time.time()
    data1 = fetcher.get_market_data('BTC-USD')
    time1 = time.time() - start
    logger.info(f"   Took: {time1:.2f}s")

    # Second fetch (should use cache)
    logger.info("\n2nd fetch (should use cache)...")
    start = time.time()
    data2 = fetcher.get_market_data('BTC-USD')
    time2 = time.time() - start
    logger.info(f"   Took: {time2:.2f}s")

    # Cache should be much faster
    if time2 < time1 / 2:
        logger.info(f"✅ Cache is working ({time2:.2f}s vs {time1:.2f}s)")
        return True
    else:
        logger.warning(f"⚠️  Cache may not be working ({time2:.2f}s vs {time1:.2f}s)")
        return False


def test_runtime_integration():
    """Test integration with runtime feature pipeline."""
    logger.info("\n🔗 Testing Runtime Integration")
    logger.info("=" * 60)

    import pandas as pd
    from apps.runtime.runtime_features import add_coingecko_features

    # Create dummy DataFrame (like runtime would have)
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-11-15', periods=100, freq='1min'),
        'open': [50000.0] * 100,
        'high': [50100.0] * 100,
        'low': [49900.0] * 100,
        'close': [50000.0] * 100,
        'volume': [1000000.0] * 100,
    })

    logger.info(f"Created dummy DataFrame: {len(df)} rows")
    logger.info(f"Columns before: {list(df.columns)}")

    # Add CoinGecko features
    df = add_coingecko_features(df, 'BTC-USD')

    logger.info(f"Columns after: {list(df.columns)}")

    # Check that CoinGecko features were added
    expected_features = [
        'ath_date', 'market_cap_change_pct', 'volume_change_pct',
        'price_change_pct', 'ath_distance_pct', 'market_cap_7d_ma',
        'market_cap_30d_ma', 'market_cap_change_7d_pct',
        'market_cap_trend', 'volume_7d_ma', 'volume_change_7d_pct'
    ]

    missing_features = [f for f in expected_features if f not in df.columns]

    if missing_features:
        logger.error(f"❌ Missing features: {missing_features}")
        return False

    logger.info("✅ All expected features present")

    # Check that at least some features have non-zero values
    non_zero_features = []
    for feature in expected_features:
        if df[feature].iloc[0] != 0.0:
            non_zero_features.append(feature)
            logger.info(f"   {feature}: {df[feature].iloc[0]:.2f}")

    if len(non_zero_features) == 0:
        logger.error("❌ All features are zero (placeholders)")
        return False

    logger.info(f"✅ {len(non_zero_features)}/{len(expected_features)} features have real data")

    logger.info("\n" + "=" * 60)
    return True


def main():
    """Run all tests."""
    logger.info("🧪 CoinGecko Premium API Integration Test")
    logger.info("=" * 60)

    tests = [
        ("API Key", test_api_key),
        ("Market Data Fetch", test_fetch_market_data),
        ("Feature Calculation", test_calculate_features),
        ("Cache Behavior", test_cache_behavior),
        ("Runtime Integration", test_runtime_integration),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running: {test_name}")
            logger.info(f"{'='*60}")
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False

    # Summary
    logger.info("\n\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {test_name}")

    passed_count = sum(results.values())
    total_count = len(results)

    logger.info("=" * 60)
    logger.info(f"Results: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        logger.info("\n✅ ALL TESTS PASSED - CoinGecko integration working!")
        logger.info("Next steps:")
        logger.info("  1. Restart runtime bot to use real CoinGecko data")
        logger.info("  2. Monitor for improved predictions (>50%)")
        return 0
    else:
        logger.error("\n❌ SOME TESTS FAILED - Fix issues before deploying")
        return 1


if __name__ == '__main__':
    sys.exit(main())
