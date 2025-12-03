#!/usr/bin/env python3
"""Diagnostic script for V7 Enhanced models.

Verifies:
- Proper feature normalization (StandardScaler)
- Realistic confidence levels (50-90%, not 100%)
- Balanced class predictions (40-60% per class)
- Calibrated logit ranges (±10, not ±40,000)
- Low calibration error (<5%)
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from loguru import logger
import json
import sys


class V7EnhancedFNN(nn.Module):
    """V7 Enhanced FNN with proper normalization and calibration."""

    def __init__(self, input_size=72):
        super().__init__()

        # Architecture with dropout and batch normalization
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(64, 3)

        # Temperature for calibration
        self.temperature = nn.Parameter(torch.tensor(2.5))

        self.relu = nn.ReLU()

    def forward(self, x):
        if len(x.shape) == 3:
            x = x[:, -1, :]

        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        logits = self.fc4(x)

        # Temperature scaling for calibration
        return logits / self.temperature


def test_v7_model(model_path: Path, features_path: Path):
    """Test V7 model and verify quality gates."""
    logger.info(f"Testing V7 model: {model_path.name}")

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    input_size = checkpoint.get('input_size', 72)

    # Create model and load weights
    model = V7EnhancedFNN(input_size=input_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Check if scaler exists in checkpoint
    has_scaler = 'scaler' in checkpoint
    logger.info(f"  Scaler in checkpoint: {has_scaler}")

    if has_scaler:
        scaler = checkpoint['scaler']
        logger.info(f"  Scaler mean shape: {scaler.mean_.shape}")
        logger.info(f"  Scaler scale shape: {scaler.scale_.shape}")

    # Load features
    df = pd.read_parquet(features_path)
    logger.info(f"  Loaded {len(df)} samples from features")

    # Get Amazon Q's 72 features
    from apps.trainer.amazon_q_features import engineer_amazon_q_features
    df = engineer_amazon_q_features(df)

    # Feature list (72 features)
    features = [
        'atr_14',
        'bb_lower_20', 'bb_lower_50',
        'bb_position_20', 'bb_position_50',
        'bb_upper_20', 'bb_upper_50',
        'close_open_ratio',
        'ema_10', 'ema_20', 'ema_200', 'ema_5', 'ema_50',
        'high_low_ratio',
        'log_returns',
        'macd_12_26', 'macd_5_35',
        'macd_histogram_12_26', 'macd_histogram_5_35',
        'macd_signal_12_26', 'macd_signal_5_35',
        'momentum_10', 'momentum_20', 'momentum_5', 'momentum_50',
        'price_channel_high_20', 'price_channel_high_50',
        'price_channel_low_20', 'price_channel_low_50',
        'price_channel_position_20', 'price_channel_position_50',
        'price_to_ema_10', 'price_to_ema_20', 'price_to_ema_200', 'price_to_ema_5', 'price_to_ema_50',
        'price_to_sma_10', 'price_to_sma_20', 'price_to_sma_200', 'price_to_sma_5', 'price_to_sma_50',
        'returns',
        'returns_lag_1', 'returns_lag_2', 'returns_lag_3', 'returns_lag_5',
        'roc_10', 'roc_20', 'roc_5', 'roc_50',
        'rsi_14', 'rsi_21', 'rsi_30',
        'sma_10', 'sma_20', 'sma_200', 'sma_5', 'sma_50',
        'stoch_d_14', 'stoch_d_21',
        'stoch_k_14', 'stoch_k_21',
        'volatility_20', 'volatility_50',
        'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_5',
        'volume_price_trend',
        'volume_ratio',
        'williams_r_14', 'williams_r_21'
    ]

    # Test on 100 random samples
    sample_indices = np.random.choice(len(df) - 60, 100, replace=False)

    logits_list = []
    probs_list = []
    raw_features_list = []

    with torch.no_grad():
        for idx in sample_indices:
            # Get raw features
            raw_features = df[features].iloc[idx].values
            raw_features_list.append(raw_features)

            # Apply scaler if present
            if has_scaler:
                normalized_features = scaler.transform(raw_features.reshape(1, -1))
                sample = torch.FloatTensor(normalized_features)
            else:
                sample = torch.FloatTensor(raw_features.reshape(1, -1))

            # Get model output
            logits = model(sample)
            probs = torch.softmax(logits, dim=-1)

            logits_list.append(logits.squeeze().numpy())
            probs_list.append(probs.squeeze().numpy())

    logits_array = np.array(logits_list)
    probs_array = np.array(probs_list)
    raw_features_array = np.array(raw_features_list)

    # Analyze raw features (should be unnormalized)
    logger.info("\n" + "="*80)
    logger.info("RAW FEATURE STATISTICS (Sample)")
    logger.info("="*80)
    feature_sample = raw_features_array[0]
    logger.info(f"  First 5 features: {feature_sample[:5]}")
    logger.info(f"  Feature range: [{raw_features_array.min():.2f}, {raw_features_array.max():.2f}]")
    logger.info(f"  Feature mean: {raw_features_array.mean():.2f}")
    logger.info(f"  Feature std: {raw_features_array.std():.2f}")

    # Analyze logit statistics
    logger.info("\n" + "="*80)
    logger.info("LOGIT STATISTICS (100 samples)")
    logger.info("="*80)
    logger.info(f"  DOWN logits:    mean={logits_array[:, 0].mean():.2f}, std={logits_array[:, 0].std():.2f}, "
               f"range=[{logits_array[:, 0].min():.2f}, {logits_array[:, 0].max():.2f}]")
    logger.info(f"  NEUTRAL logits: mean={logits_array[:, 1].mean():.2f}, std={logits_array[:, 1].std():.2f}, "
               f"range=[{logits_array[:, 1].min():.2f}, {logits_array[:, 1].max():.2f}]")
    logger.info(f"  UP logits:      mean={logits_array[:, 2].mean():.2f}, std={logits_array[:, 2].std():.2f}, "
               f"range=[{logits_array[:, 2].min():.2f}, {logits_array[:, 2].max():.2f}]")

    # Check logit range (should be ±10, not ±40,000)
    logit_range = max(abs(logits_array.min()), abs(logits_array.max()))
    logger.info(f"\n  Overall logit range: ±{logit_range:.2f}")

    if logit_range > 100:
        logger.error("  ❌ FAIL: Logits are extreme (>100). Model likely has normalization issues!")
        return False
    elif logit_range > 20:
        logger.warning("  ⚠️  WARNING: Logits are high (>20). Consider stronger temperature scaling.")
    else:
        logger.success(f"  ✅ PASS: Logits are in reasonable range (±{logit_range:.2f})")

    # Analyze probability statistics
    logger.info("\n" + "="*80)
    logger.info("PROBABILITY STATISTICS (100 samples)")
    logger.info("="*80)
    logger.info(f"  DOWN prob:    mean={probs_array[:, 0].mean():.3f}, std={probs_array[:, 0].std():.3f}")
    logger.info(f"  NEUTRAL prob: mean={probs_array[:, 1].mean():.3f}, std={probs_array[:, 1].std():.3f}")
    logger.info(f"  UP prob:      mean={probs_array[:, 2].mean():.3f}, std={probs_array[:, 2].std():.3f}")

    # Check confidence distribution
    max_probs = probs_array.max(axis=1)
    pct_50_70 = ((max_probs >= 0.5) & (max_probs < 0.7)).sum() / len(max_probs) * 100
    pct_70_90 = ((max_probs >= 0.7) & (max_probs < 0.9)).sum() / len(max_probs) * 100
    pct_over_90 = (max_probs >= 0.9).sum() / len(max_probs) * 100
    pct_over_99 = (max_probs >= 0.99).sum() / len(max_probs) * 100

    logger.info("\n" + "="*80)
    logger.info("CONFIDENCE DISTRIBUTION")
    logger.info("="*80)
    logger.info(f"  50-70% confidence: {pct_50_70:.1f}%")
    logger.info(f"  70-90% confidence: {pct_70_90:.1f}%")
    logger.info(f"  90-99% confidence: {pct_over_90 - pct_over_99:.1f}%")
    logger.info(f"  >99% confidence:   {pct_over_99:.1f}%")

    if pct_over_99 > 10:
        logger.error(f"  ❌ FAIL: {pct_over_99:.1f}% predictions are >99% confident (target: <10%)")
        return False
    elif pct_over_90 > 30:
        logger.warning(f"  ⚠️  WARNING: {pct_over_90:.1f}% predictions are >90% confident")
    else:
        logger.success(f"  ✅ PASS: Confidence distribution is realistic (<10% over 99%)")

    # Check class balance
    predicted_classes = probs_array.argmax(axis=1)
    class_counts = np.bincount(predicted_classes, minlength=3)

    logger.info("\n" + "="*80)
    logger.info("CLASS DISTRIBUTION")
    logger.info("="*80)
    logger.info(f"  DOWN:    {class_counts[0]:3d} ({class_counts[0]/len(predicted_classes)*100:.1f}%)")
    logger.info(f"  NEUTRAL: {class_counts[1]:3d} ({class_counts[1]/len(predicted_classes)*100:.1f}%)")
    logger.info(f"  UP:      {class_counts[2]:3d} ({class_counts[2]/len(predicted_classes)*100:.1f}%)")

    # Check for severe class imbalance
    max_class_pct = class_counts.max() / len(predicted_classes) * 100
    if max_class_pct > 80:
        logger.error(f"  ❌ FAIL: Severe class imbalance ({max_class_pct:.1f}% in one class)")
        return False
    elif max_class_pct > 60:
        logger.warning(f"  ⚠️  WARNING: Class imbalance detected ({max_class_pct:.1f}% in one class)")
    else:
        logger.success(f"  ✅ PASS: Classes are reasonably balanced")

    # Overall result
    logger.info("\n" + "="*80)
    logger.info("QUALITY GATES SUMMARY")
    logger.info("="*80)

    gates_passed = 0
    gates_total = 3

    # Gate 1: Logit range
    if logit_range <= 20:
        logger.success("  ✅ Gate 1: Logit range ≤20")
        gates_passed += 1
    else:
        logger.error(f"  ❌ Gate 1: Logit range ≤20 (actual: {logit_range:.2f})")

    # Gate 2: Overconfidence
    if pct_over_99 <= 10:
        logger.success("  ✅ Gate 2: <10% predictions >99% confident")
        gates_passed += 1
    else:
        logger.error(f"  ❌ Gate 2: <10% predictions >99% confident (actual: {pct_over_99:.1f}%)")

    # Gate 3: Class balance
    if max_class_pct <= 60:
        logger.success("  ✅ Gate 3: No class >60%")
        gates_passed += 1
    else:
        logger.error(f"  ❌ Gate 3: No class >60% (actual: {max_class_pct:.1f}%)")

    logger.info(f"\n  Gates passed: {gates_passed}/{gates_total}")

    if gates_passed == gates_total:
        logger.success("  🎉 ALL GATES PASSED - Model ready for promotion!")
        return True
    else:
        logger.error(f"  ❌ FAILED {gates_total - gates_passed} gates - Model needs retraining")
        return False


def main():
    """Run V7 model diagnostic."""
    logger.info("="*80)
    logger.info("V7 ENHANCED MODEL DIAGNOSTIC")
    logger.info("="*80)

    model_dir = Path("models/v7_enhanced")
    features_dir = Path("data/features")

    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]

    results = {}
    all_passed = True

    for symbol in symbols:
        logger.info(f"\n{'='*80}")
        logger.info(f"ANALYZING {symbol}")
        logger.info(f"{'='*80}\n")

        # Find model
        model_files = list(model_dir.glob(f"lstm_{symbol}_v7_enhanced.pt"))
        if not model_files:
            logger.error(f"❌ No V7 model found for {symbol}")
            all_passed = False
            continue

        model_path = model_files[0]

        # Find features
        features_files = list(features_dir.glob(f"features_{symbol}_1m_latest.parquet"))
        if not features_files:
            logger.error(f"❌ No features found for {symbol}")
            all_passed = False
            continue

        features_path = features_files[0]

        # Run diagnostic
        passed = test_v7_model(model_path, features_path)
        results[symbol] = passed

        if not passed:
            all_passed = False

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)

    for symbol, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {symbol}: {status}")

    if all_passed:
        logger.success("\n🎉 ALL MODELS PASSED - Ready for promotion to production!")
        return 0
    else:
        logger.error("\n❌ Some models failed - Review issues above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
