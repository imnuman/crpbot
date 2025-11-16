"""Fix training data dtypes by converting categorical columns to numeric.

This script converts session, volatility_regime, and ath_date to numeric dtypes
to match what the V5 FIXED models expect during inference.
"""
import sys
from pathlib import Path

import pandas as pd
from loguru import logger


def convert_categoricals_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert categorical columns to numeric codes to match V5 FIXED model training.

    The V5 FIXED models were trained with categorical columns converted to numeric:
    - session: tokyo=0, london=1, new_york=2
    - volatility_regime: low=0, medium=1, high=2
    - ath_date: converted to 0 (placeholder) or days since epoch

    Args:
        df: DataFrame with features

    Returns:
        DataFrame with categorical columns converted to numeric
    """
    df = df.copy()

    # Convert session to numeric
    if 'session' in df.columns and df['session'].dtype == 'object':
        session_map = {'tokyo': 0, 'london': 1, 'new_york': 2}
        logger.info(f"Converting session from {df['session'].dtype} to int")
        df['session'] = df['session'].map(session_map).fillna(0).astype(int)

    # Convert volatility_regime to numeric
    if 'volatility_regime' in df.columns and df['volatility_regime'].dtype == 'object':
        volatility_map = {'low': 0, 'medium': 1, 'high': 2}
        logger.info(f"Converting volatility_regime from {df['volatility_regime'].dtype} to int")
        df['volatility_regime'] = df['volatility_regime'].map(volatility_map).fillna(1).astype(int)

    # Convert ath_date to numeric (use 0 as placeholder)
    if 'ath_date' in df.columns and df['ath_date'].dtype == 'object':
        logger.info(f"Converting ath_date from {df['ath_date'].dtype} to int")
        # Set all to 0 (placeholder since we don't have real ATH data)
        df['ath_date'] = 0

    return df


def fix_training_data(symbol: str, data_dir: Path = Path("data/training")):
    """
    Fix training data for a specific symbol.

    Args:
        symbol: Trading symbol (e.g., "BTC-USD")
        data_dir: Directory containing training data
    """
    symbol_dir = data_dir / symbol

    if not symbol_dir.exists():
        logger.error(f"Training data directory not found: {symbol_dir}")
        return False

    # Process each split
    for split in ['train', 'val', 'test']:
        parquet_path = symbol_dir / f"{split}.parquet"

        if not parquet_path.exists():
            logger.warning(f"File not found: {parquet_path}")
            continue

        logger.info(f"Processing {symbol} {split}...")

        # Load data
        df = pd.read_parquet(parquet_path)
        logger.info(f"  Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

        # Check current dtypes
        logger.info(f"  Current dtypes:")
        if 'session' in df.columns:
            logger.info(f"    session: {df['session'].dtype}")
        if 'volatility_regime' in df.columns:
            logger.info(f"    volatility_regime: {df['volatility_regime'].dtype}")
        if 'ath_date' in df.columns:
            logger.info(f"    ath_date: {df['ath_date'].dtype}")

        # Convert categoricals
        df_fixed = convert_categoricals_to_numeric(df)

        # Verify conversions
        logger.info(f"  Updated dtypes:")
        if 'session' in df_fixed.columns:
            logger.info(f"    session: {df_fixed['session'].dtype}")
        if 'volatility_regime' in df_fixed.columns:
            logger.info(f"    volatility_regime: {df_fixed['volatility_regime'].dtype}")
        if 'ath_date' in df_fixed.columns:
            logger.info(f"    ath_date: {df_fixed['ath_date'].dtype}")

        # Count numeric features (excluding timestamp)
        exclude = ['timestamp']
        all_features = [c for c in df_fixed.columns if c not in exclude]
        numeric_features = [c for c in all_features if df_fixed[c].dtype in ['float64', 'float32', 'int64', 'int32']]

        logger.info(f"  Total columns: {len(df_fixed.columns)}")
        logger.info(f"  Numeric features (excl timestamp): {len(numeric_features)}")

        # Save fixed data
        backup_path = parquet_path.with_suffix('.parquet.backup')
        logger.info(f"  Creating backup: {backup_path.name}")
        parquet_path.rename(backup_path)

        logger.info(f"  Saving fixed data: {parquet_path.name}")
        df_fixed.to_parquet(parquet_path, index=False)

        logger.success(f"✅ Fixed {symbol} {split}: {len(numeric_features)} numeric features")

    return True


def main():
    """Main entry point."""
    logger.info("Starting training data dtype fix...")

    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]

    for symbol in symbols:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {symbol}")
        logger.info(f"{'='*60}")

        success = fix_training_data(symbol)

        if not success:
            logger.error(f"Failed to fix {symbol}")
            sys.exit(1)

    logger.success("\n✅ All training data fixed!")
    logger.info("\nNext steps:")
    logger.info("1. Verify numeric feature counts match model expectations (80 for BTC/SOL, 61 for ETH)")
    logger.info("2. Retrain models if needed")
    logger.info("3. Test runtime predictions")


if __name__ == "__main__":
    main()
