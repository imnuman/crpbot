# Phase 2.4: Data Quality Checks

## ✅ Implementation Complete

Comprehensive data quality validation system is fully implemented and tested.

## Overview

The data quality system provides robust validation for both raw data and engineered features, ensuring data integrity, preventing leakage, and maintaining high-quality inputs for model training.

### Key Features

- **Leakage Detection**: Checks for features using future data (T+1)
- **Completeness Validation**: Detects missing periods and gaps
- **Missing Value Detection**: Identifies and reports missing values
- **Data Type Validation**: Ensures correct data types
- **Data Range Validation**: Checks OHLCV logic and valid ranges
- **Comprehensive Reporting**: Detailed quality reports with pass/fail status

## Quality Checks

### 1. Leakage Detection

**Purpose**: Ensure no features use future data (prevents look-ahead bias)

**Checks**:
- Timestamp ordering (must be monotonic increasing)
- Feature distribution analysis across train/test splits
- Temporal overlap detection

**Usage**:
```python
from libs.data.quality import check_data_leakage

# Check for leakage
leakage_check = check_data_leakage(
    df=df_features,
    feature_columns=feature_cols,
    split_timestamp=split_time  # Optional: for train/test split validation
)
```

### 2. Data Completeness

**Purpose**: Validate data completeness (no missing periods)

**Checks**:
- Expected vs actual row count
- Missing period detection
- Large gap identification (> max_gap_minutes)

**Usage**:
```python
from libs.data.quality import check_data_completeness

completeness_check = check_data_completeness(
    df=df,
    interval="1h",
    max_gap_minutes=60,
    min_completeness_pct=95.0
)
```

### 3. Missing Values

**Purpose**: Detect and report missing values

**Checks**:
- Missing value count per column
- Missing percentage per column
- Maximum allowed missing percentage (default: 5%)

**Usage**:
```python
from libs.data.quality import check_missing_values

missing_checks = check_missing_values(
    df=df,
    feature_columns=feature_cols,
    max_missing_pct=5.0
)
```

### 4. Data Types

**Purpose**: Validate data types are correct

**Checks**:
- Timestamp column is datetime
- OHLCV columns are numeric
- Feature columns are appropriate types

**Usage**:
```python
from libs.data.quality import check_data_types

type_checks = check_data_types(df)
```

### 5. Data Ranges

**Purpose**: Validate data ranges are reasonable

**Checks**:
- High >= Low (OHLC logic)
- Prices are positive
- Volume is non-negative

**Usage**:
```python
from libs.data.quality import check_data_ranges

range_checks = check_data_ranges(df)
```

## Comprehensive Validation

### Validate Raw Data

```python
from libs.data.quality import validate_data_quality

report = validate_data_quality(
    df=df_raw,
    interval="1h",
    check_leakage=False,  # Raw data doesn't have features
    check_completeness=True,
    check_missing=True,
    check_types=True,
    check_ranges=True
)

if report.is_valid:
    print("✅ Data quality is acceptable")
else:
    print("❌ Data quality issues found")
    print(report)
```

### Validate Features

```python
from libs.data.quality import validate_feature_quality

report = validate_feature_quality(
    df=df_features,
    split_timestamp=split_time  # Optional: for leakage check
)

if report.is_valid:
    print("✅ Feature quality is acceptable")
```

## Scripts

### Validate Data Quality

```bash
# Validate raw data
python scripts/validate_data_quality.py \
    --input data/raw/test_BTC-USD_1h_7d.parquet \
    --type raw \
    --interval 1h

# Validate features
python scripts/validate_data_quality.py \
    --input data/features/features_BTC-USD_1h_latest.parquet \
    --type features

# Validate raw data and engineer features
python scripts/validate_data_quality.py \
    --input data/raw/test_BTC-USD_1h_7d.parquet \
    --type raw \
    --interval 1h \
    --engineer-features
```

### Run Quality Test Suite

```bash
python scripts/test_data_quality.py
```

This runs comprehensive tests:
- Raw data quality checks
- Feature quality checks
- Leakage detection with train/test split
- Data completeness checks
- Missing value detection
- Data type validation
- Data range validation

## Quality Report Format

```
============================================================
Data Quality Report
============================================================
Total checks: 11
Passed: 10
Failed: 1
Warnings: 1

⚠️  WARNINGS:
  ❌ FAIL [WARNING]: Leakage Detection - No feature columns found to check

✅ PASSED CHECKS:
  ✅ PASS [INFO]: Data Completeness - Completeness: 100.00% (168/168 rows)
  ✅ PASS [INFO]: Missing Values - No missing values found (checked 6 columns)
  ...
============================================================
```

## Integration with Pipeline

### Automatic Quality Checks

Quality checks are integrated into the data pipeline:

```python
from apps.trainer.data_pipeline import clean_and_validate_data
from libs.data.quality import validate_data_quality

# Clean and validate returns quality report
df, quality_report = clean_and_validate_data(df, interval="1h")

# Additional comprehensive validation
validation_report = validate_data_quality(df, interval="1h")
```

### Walk-Forward Split Validation

When creating train/test splits, validate for leakage:

```python
from apps.trainer.data_pipeline import create_walk_forward_splits
from libs.data.quality import check_data_leakage

train_df, val_df, test_df = create_walk_forward_splits(df, train_end, val_end)

# Validate no leakage
leakage_check = check_data_leakage(
    train_df,
    split_timestamp=train_end
)
assert leakage_check.passed, "Leakage detected in train set!"
```

## Quality Gates

### For Training

Before training models, ensure:
1. ✅ No leakage detected
2. ✅ Completeness ≥ 95%
3. ✅ No missing values (or handled appropriately)
4. ✅ Data types correct
5. ✅ Data ranges valid

### For Features

Before using engineered features:
1. ✅ No leakage detected
2. ✅ No missing values
3. ✅ Data types correct
4. ✅ Features validated against train/test split

## Best Practices

1. **Always validate after cleaning**: Run quality checks after data cleaning
2. **Check for leakage**: Always validate leakage when using train/test splits
3. **Monitor completeness**: Ensure data completeness ≥ 95% for reliable training
4. **Handle missing values**: Either fill missing values or remove rows with too many missing values
5. **Validate types**: Ensure correct data types before feature engineering
6. **Check ranges**: Validate OHLCV logic to catch data quality issues early

## Troubleshooting

### Leakage Detected

**Problem**: Leakage check fails

**Solutions**:
- Ensure timestamps are sorted
- Check feature engineering code uses only past data (T and earlier)
- Verify walk-forward splits don't overlap
- Review feature calculations for look-ahead bias

### Low Completeness

**Problem**: Completeness < 95%

**Solutions**:
- Check for missing periods in data
- Verify data fetching covers entire date range
- Handle gaps appropriately (fill or exclude)

### Missing Values

**Problem**: Too many missing values

**Solutions**:
- Use forward fill / backward fill
- Remove rows with too many missing values
- Use interpolation for time series
- Check data source for issues

## Next Steps

Phase 2.4 is complete. Ready to proceed to:
- Phase 3: LSTM/Transformer Models (will use quality checks in pipeline)

