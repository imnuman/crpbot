#!/usr/bin/env python3
"""
Comprehensive investigation of 50-feature model failure.

This script checks:
1. Data quality (NaN, Inf, outliers)
2. Target distribution and class balance
3. Feature statistics and correlations
4. Potential data leakage
5. Feature importance baseline
6. Simple model baseline (logistic regression)

Usage:
    uv run python scripts/investigate_50feat_failure.py --symbol BTC-USD
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def check_data_quality(df, symbol):
    """Check for NaN, Inf, and outliers."""
    print(f"\n{'='*70}")
    print(f"1. DATA QUALITY CHECK - {symbol}")
    print('='*70)

    # Exclude metadata columns
    exclude_cols = ['timestamp', 'target', 'symbol', 'interval']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print(f"\nTotal rows: {len(df):,}")
    print(f"Total features: {len(feature_cols)}")

    # Check NaN values
    print(f"\n--- NaN Values ---")
    nan_counts = df[feature_cols].isna().sum()
    has_nans = nan_counts[nan_counts > 0]
    if len(has_nans) > 0:
        print("❌ FOUND NaN values:")
        for col, count in has_nans.items():
            pct = (count / len(df)) * 100
            print(f"  {col}: {count:,} ({pct:.2f}%)")
    else:
        print("✅ No NaN values found")

    # Check Inf values
    print(f"\n--- Infinite Values ---")
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
    inf_counts = df[numeric_cols].isin([np.inf, -np.inf]).sum()
    has_infs = inf_counts[inf_counts > 0]
    if len(has_infs) > 0:
        print("❌ FOUND Inf values:")
        for col, count in has_infs.items():
            pct = (count / len(df)) * 100
            print(f"  {col}: {count:,} ({pct:.2f}%)")
    else:
        print("✅ No Inf values found")

    # Check for constant features
    print(f"\n--- Constant Features ---")
    constant_features = []
    for col in feature_cols:
        if df[col].nunique() == 1:
            constant_features.append(col)

    if constant_features:
        print(f"❌ FOUND constant features (no variance):")
        for col in constant_features:
            print(f"  {col}: all values = {df[col].iloc[0]}")
    else:
        print("✅ No constant features")

    # Check for extreme outliers (>5 std from mean)
    print(f"\n--- Extreme Outliers ---")
    outlier_summary = []
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            outliers = ((df[col] - mean).abs() > 5 * std).sum()
            if outliers > 0:
                pct = (outliers / len(df)) * 100
                outlier_summary.append((col, outliers, pct))

    if outlier_summary:
        print("⚠️  Found extreme outliers (>5σ):")
        for col, count, pct in sorted(outlier_summary, key=lambda x: x[2], reverse=True)[:10]:
            print(f"  {col}: {count:,} ({pct:.2f}%)")
    else:
        print("✅ No extreme outliers")

    return has_nans, has_infs, constant_features


def check_target_distribution(df, symbol):
    """Check target variable distribution."""
    print(f"\n{'='*70}")
    print(f"2. TARGET DISTRIBUTION - {symbol}")
    print('='*70)

    if 'target' not in df.columns:
        print("❌ ERROR: No 'target' column found")
        print("   Attempting to compute target from close prices...")

        # Compute target: 1 if price goes up in 15 minutes, 0 otherwise
        future_close = df['close'].shift(-15)
        df['target'] = (future_close > df['close']).astype(int)
        print("✅ Target computed: 1 if close[t+15] > close[t], else 0")

    target = df['target'].dropna()

    print(f"\nTotal samples: {len(target):,}")
    print(f"Valid targets: {(~target.isna()).sum():,}")
    print(f"NaN targets: {target.isna().sum():,}")

    # Class distribution
    print(f"\n--- Class Distribution ---")
    value_counts = target.value_counts()
    for val, count in value_counts.items():
        pct = (count / len(target)) * 100
        print(f"  Class {val}: {count:,} ({pct:.2f}%)")

    # Balance check
    if len(value_counts) == 2:
        balance = min(value_counts.values) / max(value_counts.values)
        print(f"\nClass balance ratio: {balance:.3f}")
        if balance < 0.4:
            print("⚠️  WARNING: Significant class imbalance (ratio < 0.4)")
        elif balance > 0.45 and balance < 0.55:
            print("✅ Well balanced classes")
        else:
            print("✅ Acceptable class balance")

    # Check if target is predictable at all
    print(f"\n--- Baseline Accuracy ---")
    majority_class_pct = (value_counts.max() / len(target)) * 100
    print(f"Majority class baseline: {majority_class_pct:.2f}%")
    print(f"(Always predicting majority class)")

    if majority_class_pct > 60:
        print("⚠️  WARNING: Majority class >60%, model might just learn to predict one class")

    return target


def check_feature_statistics(df, symbol):
    """Analyze feature statistics and correlations."""
    print(f"\n{'='*70}")
    print(f"3. FEATURE STATISTICS - {symbol}")
    print('='*70)

    exclude_cols = ['timestamp', 'target', 'symbol', 'interval']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Basic statistics
    print(f"\n--- Feature Value Ranges ---")
    stats = df[feature_cols].describe()

    # Check for features with very small variance
    print("\nFeatures with low variance (std < 0.01):")
    low_var = stats.loc['std'][stats.loc['std'] < 0.01]
    if len(low_var) > 0:
        print("⚠️  Found low-variance features:")
        for col in low_var.index:
            print(f"  {col}: std = {low_var[col]:.6f}")
    else:
        print("✅ All features have reasonable variance")

    # Check feature scales
    print(f"\n--- Feature Scales ---")
    print("Features with very different scales:")
    max_vals = stats.loc['max'].abs()
    scale_issues = []
    for col in feature_cols[:20]:  # Check first 20
        if col in stats.columns:
            max_val = stats.loc['max'][col]
            if abs(max_val) > 1000:
                scale_issues.append((col, max_val))

    if scale_issues:
        print("⚠️  Features with large values (>1000):")
        for col, val in scale_issues[:10]:
            print(f"  {col}: max = {val:.2f}")
        print("  → Consider normalization/standardization")
    else:
        print("✅ Feature scales look reasonable")

    # Correlation analysis
    print(f"\n--- Feature Correlations ---")
    print("Computing correlation matrix (may take a moment)...")

    # Sample for speed if dataset is large
    sample_size = min(100000, len(df))
    df_sample = df[feature_cols].sample(n=sample_size, random_state=42)
    corr_matrix = df_sample.corr()

    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val > 0.95:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_val
                ))

    if high_corr_pairs:
        print(f"⚠️  Found {len(high_corr_pairs)} highly correlated feature pairs (>0.95):")
        for col1, col2, corr in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)[:10]:
            print(f"  {col1} <-> {col2}: {corr:.3f}")
        print("  → High correlation may cause multicollinearity")
    else:
        print("✅ No highly correlated features (>0.95)")

    return corr_matrix


def check_data_leakage(df, symbol):
    """Check for potential data leakage."""
    print(f"\n{'='*70}")
    print(f"4. DATA LEAKAGE CHECK - {symbol}")
    print('='*70)

    exclude_cols = ['timestamp', 'target', 'symbol', 'interval']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Check for perfect correlations with target
    print("\n--- Target Correlations ---")
    if 'target' in df.columns:
        target = df['target']

        # Compute correlation with target
        correlations = []
        for col in feature_cols:
            if df[col].dtype in [np.float64, np.int64]:
                corr = df[col].corr(target)
                if not np.isnan(corr):
                    correlations.append((col, abs(corr)))

        # Sort by absolute correlation
        correlations.sort(key=lambda x: x[1], reverse=True)

        print("Top 15 features correlated with target:")
        for col, corr in correlations[:15]:
            print(f"  {col}: {corr:.4f}")

        # Check for suspiciously high correlations
        suspicious = [(col, corr) for col, corr in correlations if corr > 0.5]
        if suspicious:
            print(f"\n⚠️  WARNING: {len(suspicious)} features with correlation >0.5:")
            for col, corr in suspicious[:5]:
                print(f"  {col}: {corr:.4f}")
            print("  → May indicate data leakage")
        else:
            print("\n✅ No suspiciously high correlations with target")

    # Check for look-ahead features
    print(f"\n--- Look-Ahead Feature Check ---")
    lookahead_keywords = ['future', 'next', 'forward', 'ahead', 'target']
    suspicious_names = []
    for col in feature_cols:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in lookahead_keywords):
            suspicious_names.append(col)

    if suspicious_names:
        print("⚠️  WARNING: Features with look-ahead keywords:")
        for col in suspicious_names:
            print(f"  {col}")
    else:
        print("✅ No obvious look-ahead features by name")


def test_simple_baseline(df, symbol):
    """Test simple logistic regression baseline."""
    print(f"\n{'='*70}")
    print(f"5. SIMPLE BASELINE MODEL - {symbol}")
    print('='*70)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score

    exclude_cols = ['timestamp', 'target', 'symbol', 'interval']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    if 'target' not in df.columns:
        print("❌ ERROR: No target column")
        return

    # Remove NaN
    df_clean = df[feature_cols + ['target']].dropna()
    print(f"Clean samples: {len(df_clean):,}")

    # Split data (simple time-based split)
    split_point = int(len(df_clean) * 0.85)
    train_df = df_clean[:split_point]
    test_df = df_clean[split_point:]

    print(f"Train: {len(train_df):,} samples")
    print(f"Test:  {len(test_df):,} samples")

    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['target'].values

    # Standardize
    print("\nTraining Logistic Regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)

    # Evaluate
    train_preds = lr.predict(X_train_scaled)
    test_preds = lr.predict(X_test_scaled)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    test_precision = precision_score(y_test, test_preds, zero_division=0)
    test_recall = recall_score(y_test, test_preds, zero_division=0)

    print(f"\n--- Baseline Results ---")
    print(f"Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")

    # Interpretation
    print(f"\n--- Interpretation ---")
    if test_acc < 0.52:
        print("❌ CRITICAL: Even simple baseline can't beat random (50%)")
        print("   → Problem may be with target definition or data quality")
        print("   → Features may not be predictive at all")
    elif test_acc < 0.55:
        print("⚠️  WARNING: Baseline accuracy only slightly better than random")
        print("   → Problem is difficult, may need better features/architecture")
    elif test_acc < 0.65:
        print("✅ Baseline shows some signal (55-65% accuracy)")
        print("   → LSTM should be able to improve on this")
        print("   → If LSTM doesn't, check architecture/training")
    else:
        print("✅ Strong baseline (>65% accuracy)")
        print("   → LSTM failing at 50% suggests training issue, not data issue")

    # Feature importance
    print(f"\n--- Top 10 Most Important Features ---")
    feature_importance = list(zip(feature_cols, abs(lr.coef_[0])))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    for col, importance in feature_importance[:10]:
        print(f"  {col}: {importance:.4f}")

    return test_acc


def main():
    parser = argparse.ArgumentParser(description='Investigate 50-feature model failure')
    parser.add_argument('--symbol', type=str, default='BTC-USD',
                        help='Symbol to investigate (default: BTC-USD)')
    args = parser.parse_args()

    symbol = args.symbol

    # Load data
    feature_file = f"data/features/features_{symbol}_1m_2025-11-13_50feat.parquet"

    print(f"\n{'='*70}")
    print(f"INVESTIGATING 50-FEATURE MODEL FAILURE")
    print(f"Symbol: {symbol}")
    print(f"File: {feature_file}")
    print('='*70)

    if not Path(feature_file).exists():
        print(f"\n❌ ERROR: File not found: {feature_file}")
        print("\nAvailable files:")
        for f in Path("data/features").glob("*.parquet"):
            print(f"  {f}")
        return

    print(f"\nLoading data...")
    df = pd.read_parquet(feature_file)
    print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Run all checks
    has_nans, has_infs, constant_features = check_data_quality(df, symbol)
    target = check_target_distribution(df, symbol)
    corr_matrix = check_feature_statistics(df, symbol)
    check_data_leakage(df, symbol)
    baseline_acc = test_simple_baseline(df, symbol)

    # Final summary
    print(f"\n{'='*70}")
    print("INVESTIGATION SUMMARY")
    print('='*70)

    issues = []
    if len(has_nans) > 0:
        issues.append(f"❌ {len(has_nans)} features have NaN values")
    if len(has_infs) > 0:
        issues.append(f"❌ {len(has_infs)} features have Inf values")
    if len(constant_features) > 0:
        issues.append(f"❌ {len(constant_features)} constant features")

    if baseline_acc is not None:
        if baseline_acc < 0.52:
            issues.append(f"❌ CRITICAL: Baseline model can't beat random ({baseline_acc*100:.1f}%)")
        elif baseline_acc < 0.55:
            issues.append(f"⚠️  Baseline barely beats random ({baseline_acc*100:.1f}%)")
        elif baseline_acc >= 0.65:
            issues.append(f"✅ Baseline works ({baseline_acc*100:.1f}%) - LSTM training issue likely")

    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ No critical data issues found")

    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print('='*70)

    if baseline_acc and baseline_acc < 0.52:
        print("\n🔴 DATA/TARGET ISSUE")
        print("The problem is NOT the model - even simple baseline can't learn.")
        print("\nRecommended fixes:")
        print("1. Change target definition:")
        print("   - Try longer horizon (30 or 60 minutes)")
        print("   - Try threshold (>0.5% change instead of any change)")
        print("2. Change base timeframe:")
        print("   - Try 5-minute or 15-minute candles instead of 1-minute")
        print("3. Different task:")
        print("   - Try regression (predict price change) instead of classification")
    elif baseline_acc and baseline_acc >= 0.55:
        print("\n🟡 MODEL/TRAINING ISSUE")
        print("Simple baseline works, but LSTM doesn't - training problem.")
        print("\nRecommended fixes:")
        print("1. Check training hyperparameters:")
        print("   - Learning rate may be too high/low")
        print("   - Batch size may be too large/small")
        print("2. Simplify architecture:")
        print("   - Try 2 layers instead of 3")
        print("   - Try 64 hidden units instead of 128")
        print("3. Add regularization:")
        print("   - Increase dropout")
        print("   - Add weight decay")
        print("4. Check for gradient issues:")
        print("   - Add gradient clipping")
        print("   - Try different optimizer (AdamW)")
    else:
        print("\n⚠️  Unable to determine root cause")
        print("Run this script with --symbol for each coin to compare.")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
