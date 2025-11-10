# Feature Engineering Workflow - Phase 6.5

**Status**: Prepared and Ready
**Date**: 2025-11-10
**Dataset**: 2 years (2023-11-10 to 2025-11-10)

---

## Overview

The feature engineering pipeline generates **39 features** from raw OHLCV data:
- **5** OHLCV base columns (not used as features)
- **31** numeric features for model training
- **3** categorical features (session, volatility_regime) - encoded

---

## Feature Categories

### 1. Session Features (6 features)
**Purpose**: Capture trading session patterns (Tokyo/London/NY)

| Feature | Type | Description |
|---------|------|-------------|
| `session_tokyo` | Binary | Tokyo session (00:00-08:00 UTC) |
| `session_london` | Binary | London session (08:00-16:00 UTC) |
| `session_new_york` | Binary | New York session (16:00-00:00 UTC) |
| `day_of_week` | Numeric | 0=Monday, 6=Sunday |
| `is_weekend` | Binary | Weekend indicator |
| `session` | Categorical | Session name (excluded from model) |

### 2. Spread & Execution Features (4 features)
**Purpose**: Price spread and execution cost indicators

| Feature | Type | Description |
|---------|------|-------------|
| `spread` | Numeric | high - low (absolute) |
| `spread_pct` | Numeric | (high - low) / close × 100 |
| `atr` | Numeric | Average True Range (14-period) |
| `spread_atr_ratio` | Numeric | spread / ATR (normalized) |

### 3. Volume Features (3 features)
**Purpose**: Trading volume patterns

| Feature | Type | Description |
|---------|------|-------------|
| `volume_ma` | Numeric | 20-period moving average |
| `volume_ratio` | Numeric | current / MA (relative volume) |
| `volume_trend` | Numeric | Linear slope of volume |

### 4. Moving Average Features (8 features)
**Purpose**: Price trends and momentum

| Feature | Type | Description |
|---------|------|-------------|
| `sma_7` | Numeric | 7-period simple moving average |
| `sma_14` | Numeric | 14-period SMA |
| `sma_21` | Numeric | 21-period SMA |
| `sma_50` | Numeric | 50-period SMA |
| `price_sma_7_ratio` | Numeric | close / SMA_7 |
| `price_sma_14_ratio` | Numeric | close / SMA_14 |
| `price_sma_21_ratio` | Numeric | close / SMA_21 |
| `price_sma_50_ratio` | Numeric | close / SMA_50 |

### 5. Technical Indicators (8 features)
**Purpose**: Advanced momentum and volatility indicators (via `ta` library)

| Feature | Type | Description |
|---------|------|-------------|
| `rsi` | Numeric | Relative Strength Index (14-period) |
| `macd` | Numeric | MACD line |
| `macd_signal` | Numeric | MACD signal line |
| `macd_diff` | Numeric | MACD histogram |
| `bb_high` | Numeric | Bollinger Band upper |
| `bb_low` | Numeric | Bollinger Band lower |
| `bb_width` | Numeric | (BB_high - BB_low) / close |
| `bb_position` | Numeric | Relative position in BB (0-1) |

### 6. Volatility Regime (4 features)
**Purpose**: Market volatility classification

| Feature | Type | Description |
|---------|------|-------------|
| `volatility_low` | Binary | Low volatility regime |
| `volatility_medium` | Binary | Medium volatility regime |
| `volatility_high` | Binary | High volatility regime |
| `volatility_regime` | Categorical | Regime name (excluded from model) |

---

## Performance Optimization

### Memory Management
**Problem**: 2 years × 3 coins = ~3M rows, ~39 columns = **~450 MB in memory**

**Optimizations**:
1. ✅ **Parquet format**: Compressed storage (~50% smaller than CSV)
2. ✅ **Chunked processing**: Process per-symbol (not all at once)
3. ✅ **Forward/backward fill**: Efficient NaN handling
4. ✅ **Rolling windows**: Use pandas optimized rolling functions

### Processing Time Estimates
Per symbol (2 years, ~1M rows):
- Raw data load: ~2-3 seconds
- Feature engineering: ~30-60 seconds
- Save to parquet: ~5-10 seconds
- **Total per symbol: ~60 seconds**

All 3 symbols: **~3-5 minutes total**

---

## Command Reference

### Basic Feature Engineering (Per Symbol)
```bash
# Process raw data file
uv run python scripts/engineer_features.py \
    --input data/raw/BTC-USD_1m_2023-11-10_2025-11-10.parquet \
    --symbol BTC-USD \
    --interval 1m
```

### Batch Processing (All Symbols)
```bash
# BTC-USD
uv run python scripts/engineer_features.py \
    --input data/raw/BTC-USD_1m_2023-11-10_2025-11-10.parquet

# ETH-USD
uv run python scripts/engineer_features.py \
    --input data/raw/ETH-USD_1m_2023-11-10_2025-11-10.parquet

# BNB-USD
uv run python scripts/engineer_features.py \
    --input data/raw/BNB-USD_1m_2023-11-10_2025-11-10.parquet
```

### With Normalization (Training-Ready)
```bash
# Generate features + normalize (NOT recommended - do normalization during training)
uv run python scripts/engineer_features.py \
    --input data/raw/BTC-USD_1m_2023-11-10_2025-11-10.parquet \
    --normalize \
    --normalize-method standard
```

**⚠️ Note**: Normalization should be done **during training** (fit on train, apply to val/test) to avoid data leakage. Don't use `--normalize` flag.

### Selective Features (Optional)
```bash
# Skip certain feature categories
uv run python scripts/engineer_features.py \
    --input data/raw/BTC-USD_1m_2023-11-10_2025-11-10.parquet \
    --no-session      # Skip session features
    --no-technical    # Skip technical indicators
    --no-spread       # Skip spread features
    --no-volume       # Skip volume features
    --no-volatility   # Skip volatility regime
```

---

## Validation Checks

After feature engineering, verify:

### 1. Feature Count
```bash
uv run python - <<'EOF'
import pandas as pd
df = pd.read_parquet('data/features/features_BTC-USD_1m_latest.parquet')
print(f"Columns: {len(df.columns)}")
print(f"Rows: {len(df)}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"\nFeature columns:")
exclude = ["timestamp", "open", "high", "low", "close", "volume", "session", "volatility_regime"]
features = [col for col in df.columns if col not in exclude]
print(f"  Total: {len(features)}")
print(f"  First 10: {features[:10]}")
EOF
```

Expected output:
- **Columns**: 39
- **Rows**: ~1,051,200 (2 years × 1440 min/day)
- **Features**: ~31 numeric

### 2. Data Quality
```bash
uv run python scripts/validate_data_quality.py --symbol BTC-USD
```

Checks:
- ✅ No NaN values
- ✅ No data leakage (future → past)
- ✅ Monotonic timestamps
- ✅ Feature distributions

### 3. Memory Usage
```bash
ls -lh data/features/features_BTC-USD_1m_*.parquet
```

Expected: ~150-200 MB per file (compressed)

---

## Automated Batch Script

Create `batch_engineer_features.sh`:

```bash
#!/bin/bash
# Batch feature engineering for all symbols

SYMBOLS=("BTC-USD" "ETH-USD" "BNB-USD")
INTERVAL="1m"
START="2023-11-10"
END="2025-11-10"

for SYMBOL in "${SYMBOLS[@]}"; do
    echo "========================================="
    echo "Processing $SYMBOL..."
    echo "========================================="

    INPUT_FILE="data/raw/${SYMBOL}_${INTERVAL}_${START}_${END}.parquet"

    if [ ! -f "$INPUT_FILE" ]; then
        echo "❌ Error: $INPUT_FILE not found!"
        continue
    fi

    uv run python scripts/engineer_features.py \
        --input "$INPUT_FILE" \
        --symbol "$SYMBOL" \
        --interval "$INTERVAL"

    if [ $? -eq 0 ]; then
        echo "✅ $SYMBOL features completed"
    else
        echo "❌ $SYMBOL features failed"
        exit 1
    fi

    echo ""
done

echo "========================================="
echo "✅ All feature engineering complete!"
echo "========================================="
```

Usage:
```bash
chmod +x batch_engineer_features.sh
./batch_engineer_features.sh
```

---

## Output Format

Generated files:
```
data/features/
├── features_BTC-USD_1m_2025-11-10.parquet
├── features_BTC-USD_1m_latest.parquet → features_BTC-USD_1m_2025-11-10.parquet
├── features_ETH-USD_1m_2025-11-10.parquet
├── features_ETH-USD_1m_latest.parquet → features_ETH-USD_1m_2025-11-10.parquet
├── features_BNB-USD_1m_2025-11-10.parquet
└── features_BNB-USD_1m_latest.parquet → features_BNB-USD_1m_2025-11-10.parquet
```

**Note**: `*_latest.parquet` symlinks always point to the most recent version.

---

## Troubleshooting

### Issue: `ta` library not found
```bash
uv pip install ta
```

### Issue: Memory error (large dataset)
Solution: Process in chunks or increase system memory.

### Issue: NaN values in features
Check: Initial rows may have NaN due to rolling windows (e.g., SMA_50 needs 50 rows). The script applies `ffill().bfill()` to handle this.

### Issue: Feature engineering too slow
- Check disk I/O (SSD vs HDD)
- Reduce feature complexity (disable some feature categories)
- Use smaller dataset for testing

---

## Next Steps

After feature engineering:

1. ✅ Validate data quality
2. ✅ Review feature distributions
3. ✅ Proceed to model training (LSTM → Transformer)
4. ✅ Evaluate models with promotion gates

---

## References

- Feature implementation: `apps/trainer/features.py`
- Main script: `scripts/engineer_features.py`
- Data quality checks: `scripts/validate_data_quality.py`
- Phase 6.5 plan: `PHASE6_5_RESTART_PLAN.md`
