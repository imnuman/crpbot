#!/usr/bin/env python3
"""Diagnostic script to identify issues with V6 Enhanced models.

Analyzes:
- Model architecture and weights
- Training data distribution
- Prediction calibration
- Feature distributions
- Output logit ranges
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import json

def analyze_model_weights(model_path: Path):
    """Analyze model weights for anomalies."""
    logger.info(f"Analyzing model: {model_path.name}")

    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']

    # Analyze weight statistics
    weight_stats = {}
    for name, param in state_dict.items():
        weight_stats[name] = {
            'shape': list(param.shape),
            'mean': param.mean().item(),
            'std': param.std().item(),
            'min': param.min().item(),
            'max': param.max().item(),
            'has_nan': torch.isnan(param).any().item(),
            'has_inf': torch.isinf(param).any().item()
        }

    logger.info("Weight Statistics:")
    for name, stats in weight_stats.items():
        logger.info(f"  {name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                   f"range=[{stats['min']:.4f}, {stats['max']:.4f}]")
        if stats['has_nan'] or stats['has_inf']:
            logger.error(f"  ❌ {name} contains NaN or Inf values!")

    return weight_stats


def analyze_training_data(features_path: Path):
    """Analyze training data for class imbalance and feature distributions."""
    logger.info(f"Analyzing training data: {features_path.name}")

    df = pd.read_parquet(features_path)

    # Check for target variable (assuming 'direction' or 'target')
    target_col = None
    for col in ['direction', 'target', 'label', 'close']:
        if col in df.columns:
            target_col = col
            break

    if target_col == 'close':
        # Create binary target from price changes
        df['target'] = (df['close'].shift(-15) > df['close']).astype(int)
        target_col = 'target'

    if target_col:
        logger.info(f"Target variable: {target_col}")
        value_counts = df[target_col].value_counts()
        logger.info(f"  Class distribution: {value_counts.to_dict()}")

        # Calculate class imbalance ratio
        if len(value_counts) == 2:
            majority = value_counts.max()
            minority = value_counts.min()
            imbalance_ratio = majority / minority
            logger.info(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")

            if imbalance_ratio > 1.5:
                logger.warning(f"  ⚠️  Significant class imbalance detected!")
    else:
        logger.warning("  No target variable found in training data")

    # Analyze feature distributions
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    logger.info(f"Numeric features: {len(numeric_cols)}")

    feature_stats = {}
    for col in numeric_cols:
        if col in ['timestamp', 'target', 'direction', 'label']:
            continue

        stats = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'has_nan': df[col].isna().any(),
            'pct_nan': df[col].isna().sum() / len(df) * 100
        }
        feature_stats[col] = stats

        # Flag features with extreme values or high NaN percentage
        if abs(stats['mean']) > 1000 or abs(stats['std']) > 1000:
            logger.warning(f"  ⚠️  {col}: extreme values (mean={stats['mean']:.2f}, std={stats['std']:.2f})")
        if stats['pct_nan'] > 10:
            logger.warning(f"  ⚠️  {col}: high NaN percentage ({stats['pct_nan']:.1f}%)")

    return feature_stats


def test_model_predictions(model_path: Path, features_path: Path):
    """Test model predictions and analyze output distribution."""
    logger.info("Testing model predictions...")

    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    input_size = checkpoint.get('input_size', 72)

    # Create model architecture
    import torch.nn as nn

    class V6EnhancedFNN(nn.Module):
        def __init__(self, input_size=72):
            super().__init__()
            self.fc1 = nn.Linear(input_size, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 64)
            self.fc4 = nn.Linear(64, 3)
            self.relu = nn.ReLU()

        def forward(self, x):
            if len(x.shape) == 3:
                x = x[:, -1, :]
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.relu(self.fc3(x))
            return self.fc4(x)

    model = V6EnhancedFNN(input_size=input_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load features
    df = pd.read_parquet(features_path)

    # Get Amazon Q's 72 features
    from apps.trainer.amazon_q_features import engineer_amazon_q_features
    df = engineer_amazon_q_features(df)

    # Get feature list
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

    with torch.no_grad():
        for idx in sample_indices:
            sample = torch.FloatTensor(df[features].iloc[idx:idx+60].values).unsqueeze(0)
            logits = model(sample)
            probs = torch.softmax(logits, dim=-1)

            logits_list.append(logits.squeeze().numpy())
            probs_list.append(probs.squeeze().numpy())

    logits_array = np.array(logits_list)
    probs_array = np.array(probs_list)

    logger.info("Logit Statistics (100 samples):")
    logger.info(f"  Down logits: mean={logits_array[:, 0].mean():.2f}, std={logits_array[:, 0].std():.2f}, "
               f"range=[{logits_array[:, 0].min():.2f}, {logits_array[:, 0].max():.2f}]")
    logger.info(f"  Neutral logits: mean={logits_array[:, 1].mean():.2f}, std={logits_array[:, 1].std():.2f}, "
               f"range=[{logits_array[:, 1].min():.2f}, {logits_array[:, 1].max():.2f}]")
    logger.info(f"  Up logits: mean={logits_array[:, 2].mean():.2f}, std={logits_array[:, 2].std():.2f}, "
               f"range=[{logits_array[:, 2].min():.2f}, {logits_array[:, 2].max():.2f}]")

    logger.info("\nProbability Statistics (100 samples):")
    logger.info(f"  Down prob: mean={probs_array[:, 0].mean():.3f}, std={probs_array[:, 0].std():.3f}")
    logger.info(f"  Neutral prob: mean={probs_array[:, 1].mean():.3f}, std={probs_array[:, 1].std():.3f}")
    logger.info(f"  Up prob: mean={probs_array[:, 2].mean():.3f}, std={probs_array[:, 2].std():.3f}")

    # Check for extreme overconfidence
    max_probs = probs_array.max(axis=1)
    pct_over_90 = (max_probs > 0.9).sum() / len(max_probs) * 100
    pct_over_99 = (max_probs > 0.99).sum() / len(max_probs) * 100

    logger.info(f"\nConfidence Distribution:")
    logger.info(f"  Predictions > 90% confidence: {pct_over_90:.1f}%")
    logger.info(f"  Predictions > 99% confidence: {pct_over_99:.1f}%")

    if pct_over_90 > 50:
        logger.error("  ❌ Model is severely overconfident!")
    elif pct_over_90 > 30:
        logger.warning("  ⚠️  Model shows high overconfidence")

    # Check prediction distribution
    predicted_classes = probs_array.argmax(axis=1)
    class_counts = np.bincount(predicted_classes, minlength=3)
    logger.info(f"\nPredicted Class Distribution:")
    logger.info(f"  Down: {class_counts[0]} ({class_counts[0]/len(predicted_classes)*100:.1f}%)")
    logger.info(f"  Neutral: {class_counts[1]} ({class_counts[1]/len(predicted_classes)*100:.1f}%)")
    logger.info(f"  Up: {class_counts[2]} ({class_counts[2]/len(predicted_classes)*100:.1f}%)")

    if class_counts[0] > 80 or class_counts[2] > 80:
        logger.error("  ❌ Model is biased towards one class!")

    return {
        'logits': logits_array,
        'probs': probs_array,
        'pct_over_90': pct_over_90,
        'pct_over_99': pct_over_99,
        'class_distribution': class_counts.tolist()
    }


def main():
    """Run full diagnostic."""
    logger.info("=" * 80)
    logger.info("V6 ENHANCED MODEL DIAGNOSTIC")
    logger.info("=" * 80)

    # Paths
    model_dir = Path("models/promoted")
    features_dir = Path("data/features")

    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]

    results = {}

    for symbol in symbols:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"ANALYZING {symbol}")
        logger.info(f"{'=' * 80}\n")

        # Find model
        model_files = list(model_dir.glob(f"lstm_{symbol}_v6_enhanced.pt"))
        if not model_files:
            logger.error(f"No V6 enhanced model found for {symbol}")
            continue

        model_path = model_files[0]

        # Find features
        features_files = list(features_dir.glob(f"features_{symbol}_1m_latest.parquet"))
        if not features_files:
            logger.error(f"No features found for {symbol}")
            continue

        features_path = features_files[0]

        # Run diagnostics
        weight_stats = analyze_model_weights(model_path)
        feature_stats = analyze_training_data(features_path)
        prediction_stats = test_model_predictions(model_path, features_path)

        results[symbol] = {
            'weight_stats': weight_stats,
            'prediction_stats': prediction_stats
        }

    # Save results
    output_file = Path("reports/v6_model_diagnostic.json")
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\n✅ Diagnostic complete. Results saved to: {output_file}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("=" * 80)

    for symbol, result in results.items():
        pred_stats = result['prediction_stats']
        logger.info(f"\n{symbol}:")
        logger.info(f"  Overconfidence (>90%): {pred_stats['pct_over_90']:.1f}%")
        logger.info(f"  Extreme confidence (>99%): {pred_stats['pct_over_99']:.1f}%")
        logger.info(f"  Class bias: {pred_stats['class_distribution']}")


if __name__ == "__main__":
    main()
