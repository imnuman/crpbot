"""
Engineer Features for All Symbols

Applies 35+ technical indicators to 2 years of historical data for all 10 symbols.
"""
import sys
_this_file = Path(__file__).resolve()
_project_root = _this_file.parent.parent
sys.path.insert(0, str(_project_root))

import pandas as pd
import logging
from pathlib import Path
from libs.features.technical_indicators import TechnicalIndicators

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# All 10 production symbols
SYMBOLS = [
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'DOGE-USD',
    'ADA-USD', 'AVAX-USD', 'LINK-USD', 'POL-USD', 'LTC-USD'
]


def engineer_features_for_symbol(symbol: str, ti: TechnicalIndicators) -> bool:
    """Engineer features for a single symbol"""

    # Convert symbol format for filename
    symbol_clean = symbol.replace('-', '_')

    # Input path
    input_path = Path(f'data/historical/{symbol_clean}_3600s_730d.parquet')

    if not input_path.exists():
        logger.error(f"❌ Historical data not found: {input_path}")
        return False

    logger.info(f"📊 Processing {symbol}...")

    # Load historical data
    df = pd.read_parquet(input_path)
    logger.info(f"  Loaded {len(df)} candles")

    # Add all technical indicators
    df_with_features = ti.add_all_indicators(df.copy())

    # Count features
    original_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    feature_cols = [c for c in df_with_features.columns if c not in original_cols]

    logger.info(f"  Added {len(feature_cols)} features ({len(df_with_features.columns)} total columns)")

    # Check data quality
    null_counts = df_with_features[feature_cols].isnull().sum()
    total_nulls = null_counts.sum()
    null_pct = (total_nulls / (len(df_with_features) * len(feature_cols))) * 100

    logger.info(f"  Nulls: {total_nulls:,} / {len(df_with_features) * len(feature_cols):,} ({null_pct:.2f}%)")

    # Save enriched data
    output_path = Path(f'data/historical/{symbol_clean}_3600s_730d_features.parquet')
    df_with_features.to_parquet(output_path, compression='snappy', index=False)

    file_size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"  ✅ Saved: {output_path} ({file_size_mb:.2f} MB)")

    return True


def main():
    """Engineer features for all symbols"""

    print("=" * 70)
    print("FEATURE ENGINEERING - ALL SYMBOLS")
    print("=" * 70)
    print(f"\nSymbols: {len(SYMBOLS)}")
    print(f"Features: 35+ technical indicators")
    print(f"Timeframe: 2 years (730 days) @ 1 hour candles\n")

    # Initialize technical indicators
    ti = TechnicalIndicators()

    # Process each symbol
    success_count = 0
    failed_symbols = []

    for i, symbol in enumerate(SYMBOLS, 1):
        print(f"\n[{i}/{len(SYMBOLS)}] {symbol}")
        print("-" * 70)

        success = engineer_features_for_symbol(symbol, ti)

        if success:
            success_count += 1
        else:
            failed_symbols.append(symbol)

    # Summary
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 70)
    print(f"\n✅ Successfully processed: {success_count}/{len(SYMBOLS)} symbols")

    if failed_symbols:
        print(f"❌ Failed symbols: {', '.join(failed_symbols)}")

    # Calculate total storage
    total_size_mb = 0
    for symbol in SYMBOLS:
        symbol_clean = symbol.replace('-', '_')
        output_path = Path(f'data/historical/{symbol_clean}_3600s_730d_features.parquet')
        if output_path.exists():
            total_size_mb += output_path.stat().st_size / 1024 / 1024

    print(f"\n💾 Total storage: {total_size_mb:.2f} MB")
    print(f"📁 Location: data/historical/*_features.parquet")

    print("\n" + "=" * 70)
    print("✅ Feature Engineering Complete!")
    print("=" * 70)

    return success_count == len(SYMBOLS)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
