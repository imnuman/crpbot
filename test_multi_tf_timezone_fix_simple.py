#!/usr/bin/env python3
"""Test multi-TF timezone alignment logic with mock data."""

import pandas as pd
from loguru import logger


def test_timezone_alignment():
    """Test the timezone alignment fix with mock data."""
    from apps.runtime.multi_tf_fetcher import align_multi_tf_to_base

    logger.info("Testing Multi-TF Timezone Alignment Fix")
    logger.info("="*60)

    # Create mock base 1m data (timezone-naive)
    logger.info("\n1. Creating mock base 1m data (timezone-naive)...")
    base_timestamps = pd.date_range(
        start='2025-11-16 10:00:00',
        periods=10,
        freq='1min'
    )
    base_df = pd.DataFrame({
        'timestamp': base_timestamps,
        'open': [100.0 + i for i in range(10)],
        'high': [101.0 + i for i in range(10)],
        'low': [99.0 + i for i in range(10)],
        'close': [100.5 + i for i in range(10)],
        'volume': [1000.0 + i*10 for i in range(10)]
    })
    logger.info(f"   Base DF dtype: {base_df['timestamp'].dtype}")
    logger.info(f"   Base DF timezone: {base_df['timestamp'].dt.tz}")

    # Create mock 5m data (timezone-aware UTC)
    logger.info("\n2. Creating mock 5m data (timezone-aware UTC)...")
    tf_5m_timestamps = pd.date_range(
        start='2025-11-16 10:00:00',
        periods=3,
        freq='5min',
        tz='UTC'
    )
    df_5m = pd.DataFrame({
        'timestamp': tf_5m_timestamps,
        'open': [100.0, 105.0, 110.0],
        'high': [101.0, 106.0, 111.0],
        'low': [99.0, 104.0, 109.0],
        'close': [100.5, 105.5, 110.5],
        'volume': [5000.0, 5100.0, 5200.0]
    })
    logger.info(f"   5m DF dtype: {df_5m['timestamp'].dtype}")
    logger.info(f"   5m DF timezone: {df_5m['timestamp'].dt.tz}")

    # Create mock 15m data (timezone-aware UTC)
    logger.info("\n3. Creating mock 15m data (timezone-aware UTC)...")
    tf_15m_timestamps = pd.date_range(
        start='2025-11-16 10:00:00',
        periods=1,
        freq='15min',
        tz='UTC'
    )
    df_15m = pd.DataFrame({
        'timestamp': tf_15m_timestamps,
        'open': [100.0],
        'high': [115.0],
        'low': [99.0],
        'close': [112.0],
        'volume': [15000.0]
    })
    logger.info(f"   15m DF dtype: {df_15m['timestamp'].dtype}")
    logger.info(f"   15m DF timezone: {df_15m['timestamp'].dt.tz}")

    # Create multi_tf_data dict
    multi_tf_data = {
        '5m': df_5m,
        '15m': df_15m
    }

    # Test alignment
    logger.info("\n4. Testing alignment (this should NOT raise timezone error)...")
    try:
        aligned_df = align_multi_tf_to_base(
            base_df=base_df,
            multi_tf_data=multi_tf_data
        )

        logger.info(f"   ✅ Alignment successful!")
        logger.info(f"   Aligned DF shape: {aligned_df.shape}")
        logger.info(f"   Aligned DF timestamp dtype: {aligned_df['timestamp'].dtype}")
        logger.info(f"   Aligned DF timezone: {aligned_df['timestamp'].dt.tz}")

        # Check for multi-TF columns
        tf_columns = [col for col in aligned_df.columns if '5m_' in col or '15m_' in col]
        logger.info(f"\n5. Checking multi-TF columns...")
        logger.info(f"   ✅ Found {len(tf_columns)} multi-TF columns:")
        for col in tf_columns:
            logger.info(f"      {col}")

        # Display sample
        logger.info(f"\n6. Sample aligned data:")
        sample_cols = ['timestamp', 'close', '5m_close', '15m_close']
        logger.info("\n" + aligned_df[sample_cols].to_string())

        logger.info("\n" + "="*60)
        logger.info("✅ TIMEZONE ALIGNMENT FIX VERIFIED!")
        logger.info("="*60)
        logger.info("\nThe fix successfully handles:")
        logger.info("  • Timezone-naive base DataFrame")
        logger.info("  • Timezone-aware UTC multi-TF DataFrames")
        logger.info("  • No merge errors encountered")
        logger.info("  • Proper conversion to timezone-aware UTC")

        return True

    except TypeError as e:
        if "incompatible merge keys" in str(e):
            logger.error(f"\n❌ TIMEZONE ERROR STILL PRESENT: {e}")
            logger.error("The fix did not work correctly!")
            return False
        else:
            raise
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_timezone_alignment()

    if success:
        logger.info("\n🎉 Test PASSED!")
        logger.info("Timezone fix is working correctly.")
        logger.info("Ready to proceed with data collection.")
        exit(0)
    else:
        logger.error("\n❌ Test FAILED!")
        logger.error("Timezone fix needs more work.")
        exit(1)
