#!/usr/bin/env python3
"""Test multi-TF data pipeline with timezone fix."""

import os
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

# Load environment
load_dotenv()

def test_multi_tf_timezone_alignment():
    """Test multi-TF data fetching and alignment with timezone fix."""
    from apps.runtime.data_fetcher import get_data_fetcher
    from apps.runtime.multi_tf_fetcher import fetch_multi_tf_data, align_multi_tf_to_base

    logger.info("Testing Multi-TF Data Pipeline with Timezone Fix")
    logger.info("="*60)

    # Initialize data fetcher (uses Settings from config)
    fetcher = get_data_fetcher()

    symbol = "BTC-USD"

    try:
        # Step 1: Fetch base 1m data
        logger.info(f"\n1. Fetching base 1m data for {symbol}...")
        base_df = fetcher.fetch_latest_candles(symbol=symbol, num_candles=150)

        if base_df is None or len(base_df) == 0:
            logger.error("Failed to fetch base 1m data")
            return False

        logger.info(f"   ✅ Fetched {len(base_df)} 1m candles")
        logger.info(f"   Timestamp dtype: {base_df['timestamp'].dtype}")
        logger.info(f"   Timezone: {base_df['timestamp'].dt.tz}")

        # Step 2: Fetch multi-TF data
        logger.info(f"\n2. Fetching multi-TF data (5m, 15m, 1h)...")
        multi_tf_data = fetch_multi_tf_data(
            data_fetcher=fetcher,
            symbol=symbol,
            intervals=["5m", "15m", "1h"],
            num_candles=150
        )

        if not multi_tf_data:
            logger.error("Failed to fetch multi-TF data")
            return False

        logger.info(f"   ✅ Fetched {len(multi_tf_data)} timeframes")
        for interval, df in multi_tf_data.items():
            logger.info(f"   {interval}: {len(df)} candles, dtype={df['timestamp'].dtype}, tz={df['timestamp'].dt.tz}")

        # Step 3: Align multi-TF data to base
        logger.info(f"\n3. Aligning multi-TF data to base 1m...")
        aligned_df = align_multi_tf_to_base(
            base_df=base_df,
            multi_tf_data=multi_tf_data
        )

        if aligned_df is None or len(aligned_df) == 0:
            logger.error("Failed to align multi-TF data")
            return False

        logger.info(f"   ✅ Aligned DataFrame created: {len(aligned_df)} rows")
        logger.info(f"   Timestamp dtype: {aligned_df['timestamp'].dtype}")
        logger.info(f"   Timezone: {aligned_df['timestamp'].dt.tz}")

        # Step 4: Check for multi-TF columns
        logger.info(f"\n4. Checking multi-TF columns...")
        tf_columns = [col for col in aligned_df.columns if any(tf in col for tf in ['5m_', '15m_', '1h_'])]
        logger.info(f"   ✅ Found {len(tf_columns)} multi-TF columns:")
        for col in tf_columns[:10]:  # Show first 10
            logger.info(f"      {col}")
        if len(tf_columns) > 10:
            logger.info(f"      ... and {len(tf_columns) - 10} more")

        # Step 5: Check for missing values
        logger.info(f"\n5. Checking data quality...")
        missing_counts = aligned_df[tf_columns].isnull().sum()
        total_missing = missing_counts.sum()

        if total_missing > 0:
            logger.warning(f"   ⚠️  Found {total_missing} missing values in multi-TF columns")
            logger.info(f"   Missing by column:")
            for col, count in missing_counts[missing_counts > 0].items():
                logger.info(f"      {col}: {count}")
        else:
            logger.info(f"   ✅ No missing values in multi-TF columns")

        # Step 6: Display sample data
        logger.info(f"\n6. Sample aligned data (last 5 rows):")
        sample_cols = ['timestamp', 'close', '5m_close', '15m_close', '1h_close']
        logger.info(aligned_df[sample_cols].tail().to_string())

        logger.info("\n" + "="*60)
        logger.info("✅ Multi-TF Data Pipeline Test PASSED!")
        logger.info("="*60)
        logger.info("\nTimezone fix verification:")
        logger.info(f"  • Base timestamp: {aligned_df['timestamp'].dtype} (tz={aligned_df['timestamp'].dt.tz})")
        logger.info(f"  • No merge errors encountered")
        logger.info(f"  • Multi-TF alignment successful")

        return True

    except Exception as e:
        logger.error(f"❌ Multi-TF test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_multi_tf_timezone_alignment()

    if success:
        logger.info("\n🎉 Test completed successfully!")
        logger.info("Timezone fix is working correctly.")
        exit(0)
    else:
        logger.error("\n❌ Test failed!")
        logger.error("Review errors above.")
        exit(1)
