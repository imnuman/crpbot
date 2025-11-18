#!/usr/bin/env python3
"""
V6 Temperature Tuning Script
Tests different temperature values to find optimal confidence range (60-75%)
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from loguru import logger


class V6EnhancedFNN(nn.Module):
    """Original V6 model architecture."""
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


def test_temperature(base_model, scaler, features_path, feature_cols, temperature, symbol):
    """Test a specific temperature value."""

    # Load features
    df = pd.read_parquet(features_path)
    from apps.trainer.amazon_q_features import engineer_amazon_q_features
    df = engineer_amazon_q_features(df)

    # Test on 100 random samples
    sample_indices = np.random.choice(len(df) - 60, 100, replace=False)

    logits_list = []
    probs_list = []

    with torch.no_grad():
        for idx in sample_indices:
            # Get features
            features = df[feature_cols].iloc[idx:idx+60].values
            features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

            # Normalize
            if scaler is not None:
                features = scaler.transform(features)

            # Convert to tensor
            sample = torch.FloatTensor(features).unsqueeze(0)

            # Get raw logits
            logits = base_model(sample)

            # Clamp logits
            logits = torch.clamp(logits, -15.0, 15.0)

            # Apply temperature
            logits_scaled = logits / temperature

            # Get probabilities
            probs = torch.softmax(logits_scaled, dim=-1)

            logits_list.append(logits_scaled.squeeze().numpy())
            probs_list.append(probs.squeeze().numpy())

    logits_array = np.array(logits_list)
    probs_array = np.array(probs_list)

    # Calculate metrics
    max_probs = probs_array.max(axis=1)
    avg_conf = max_probs.mean()
    pct_over_60 = (max_probs > 0.6).sum() / len(max_probs) * 100
    pct_over_75 = (max_probs > 0.75).sum() / len(max_probs) * 100
    pct_over_90 = (max_probs > 0.9).sum() / len(max_probs) * 100

    # Class distribution
    predicted_classes = probs_array.argmax(axis=1)
    class_counts = np.bincount(predicted_classes, minlength=3)

    return {
        'temperature': temperature,
        'symbol': symbol,
        'avg_confidence': avg_conf,
        'pct_60_75': pct_over_60 - pct_over_75,  # Sweet spot
        'pct_over_60': pct_over_60,
        'pct_over_75': pct_over_75,
        'pct_over_90': pct_over_90,
        'class_distribution': class_counts.tolist(),
        'logit_mean': logits_array.mean(),
        'logit_std': logits_array.std(),
        'prob_down': probs_array[:, 0].mean(),
        'prob_neutral': probs_array[:, 1].mean(),
        'prob_up': probs_array[:, 2].mean()
    }


def main():
    """Test different temperature values."""

    logger.info("="*80)
    logger.info("V6 TEMPERATURE TUNING")
    logger.info("="*80)

    # Amazon Q's 72 features
    feature_cols = [
        'atr_14', 'bb_lower_20', 'bb_lower_50', 'bb_position_20', 'bb_position_50',
        'bb_upper_20', 'bb_upper_50', 'close_open_ratio', 'ema_10', 'ema_20',
        'ema_200', 'ema_5', 'ema_50', 'high_low_ratio', 'log_returns',
        'macd_12_26', 'macd_5_35', 'macd_histogram_12_26', 'macd_histogram_5_35',
        'macd_signal_12_26', 'macd_signal_5_35', 'momentum_10', 'momentum_20',
        'momentum_5', 'momentum_50', 'price_channel_high_20', 'price_channel_high_50',
        'price_channel_low_20', 'price_channel_low_50', 'price_channel_position_20',
        'price_channel_position_50', 'price_to_ema_10', 'price_to_ema_20',
        'price_to_ema_200', 'price_to_ema_5', 'price_to_ema_50', 'price_to_sma_10',
        'price_to_sma_20', 'price_to_sma_200', 'price_to_sma_5', 'price_to_sma_50',
        'returns', 'returns_lag_1', 'returns_lag_2', 'returns_lag_3', 'returns_lag_5',
        'roc_10', 'roc_20', 'roc_5', 'roc_50', 'rsi_14', 'rsi_21', 'rsi_30',
        'sma_10', 'sma_20', 'sma_200', 'sma_5', 'sma_50', 'stoch_d_14', 'stoch_d_21',
        'stoch_k_14', 'stoch_k_21', 'volatility_20', 'volatility_50', 'volume_lag_1',
        'volume_lag_2', 'volume_lag_3', 'volume_lag_5', 'volume_price_trend',
        'volume_ratio', 'williams_r_14', 'williams_r_21'
    ]

    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
    temperatures = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

    model_dir = Path("models/promoted")
    features_dir = Path("data/features")
    scalers_dir = Path("models/v6_fixed")

    all_results = []

    for symbol in symbols:
        logger.info(f"\n{'='*80}")
        logger.info(f"TESTING {symbol}")
        logger.info(f"{'='*80}\n")

        # Load model
        model_path = model_dir / f"lstm_{symbol}_v6_enhanced.pt"
        features_path = features_dir / f"features_{symbol}_1m_latest.parquet"
        scaler_path = scalers_dir / f"scaler_{symbol}_v6_fixed.pkl"

        if not model_path.exists() or not features_path.exists() or not scaler_path.exists():
            logger.error(f"Missing files for {symbol}")
            continue

        # Load base model
        checkpoint = torch.load(model_path, map_location='cpu')
        base_model = V6EnhancedFNN(input_size=72)
        base_model.load_state_dict(checkpoint['model_state_dict'])
        base_model.eval()

        # Load scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        # Test each temperature
        for temp in temperatures:
            logger.info(f"Testing T={temp:.1f}...")
            result = test_temperature(base_model, scaler, features_path, feature_cols, temp, symbol)
            all_results.append(result)

            logger.info(f"  Avg confidence: {result['avg_confidence']:.1%}")
            logger.info(f"  In sweet spot (60-75%): {result['pct_60_75']:.1f}%")
            logger.info(f"  Over 90% confident: {result['pct_over_90']:.1f}%")

    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEMPERATURE TUNING RESULTS - SUMMARY")
    logger.info("="*80)

    # Group by temperature
    for temp in temperatures:
        temp_results = [r for r in all_results if r['temperature'] == temp]
        avg_conf = np.mean([r['avg_confidence'] for r in temp_results])
        avg_sweet_spot = np.mean([r['pct_60_75'] for r in temp_results])
        avg_over_90 = np.mean([r['pct_over_90'] for r in temp_results])

        logger.info(f"\nT={temp:.1f}:")
        logger.info(f"  Average confidence:     {avg_conf:.1%}")
        logger.info(f"  In sweet spot (60-75%): {avg_sweet_spot:.1f}%")
        logger.info(f"  Overconfident (>90%):   {avg_over_90:.1f}%")

        # Assessment
        if 0.60 <= avg_conf <= 0.75 and avg_over_90 < 10:
            logger.success(f"  ✅ OPTIMAL - Perfect range!")
        elif 0.55 <= avg_conf <= 0.80 and avg_over_90 < 15:
            logger.info(f"  ⚠️  GOOD - Acceptable range")
        elif avg_conf < 0.55:
            logger.warning(f"  ❌ TOO LOW - Increase confidence")
        else:
            logger.warning(f"  ❌ TOO HIGH - Risk of overconfidence")

    # Detailed table
    logger.info("\n" + "="*80)
    logger.info("DETAILED RESULTS BY SYMBOL AND TEMPERATURE")
    logger.info("="*80)

    df_results = pd.DataFrame(all_results)

    for symbol in symbols:
        symbol_results = df_results[df_results['symbol'] == symbol]
        logger.info(f"\n{symbol}:")
        logger.info(f"{'T':>6} {'Avg Conf':>10} {'60-75%':>8} {'>90%':>8} {'Class Dist':>20}")
        logger.info("-" * 60)
        for _, row in symbol_results.iterrows():
            class_str = f"[{row['class_distribution'][0]}, {row['class_distribution'][1]}, {row['class_distribution'][2]}]"
            logger.info(f"{row['temperature']:>6.1f} {row['avg_confidence']:>9.1%} {row['pct_60_75']:>7.1f}% {row['pct_over_90']:>7.1f}% {class_str:>20}")

    # Recommendation
    logger.info("\n" + "="*80)
    logger.info("RECOMMENDATION")
    logger.info("="*80)

    # Find best temperature (closest to 67.5% avg confidence, <10% overconfident)
    best_temp = None
    best_score = float('inf')

    for temp in temperatures:
        temp_results = [r for r in all_results if r['temperature'] == temp]
        avg_conf = np.mean([r['avg_confidence'] for r in temp_results])
        avg_over_90 = np.mean([r['pct_over_90'] for r in temp_results])

        # Score: distance from 67.5% + penalty for overconfidence
        score = abs(avg_conf - 0.675) + (max(0, avg_over_90 - 10) * 0.01)

        if score < best_score:
            best_score = score
            best_temp = temp

    logger.success(f"\n✅ RECOMMENDED TEMPERATURE: {best_temp:.1f}")

    best_results = [r for r in all_results if r['temperature'] == best_temp]
    avg_conf = np.mean([r['avg_confidence'] for r in best_results])
    avg_sweet_spot = np.mean([r['pct_60_75'] for r in best_results])
    avg_over_90 = np.mean([r['pct_over_90'] for r in best_results])

    logger.info(f"\nExpected Performance:")
    logger.info(f"  Average confidence:     {avg_conf:.1%}")
    logger.info(f"  In sweet spot (60-75%): {avg_sweet_spot:.1f}%")
    logger.info(f"  Overconfident (>90%):   {avg_over_90:.1f}%")

    logger.info(f"\nNext Steps:")
    logger.info(f"  1. Update fix_v6_models.py to use temperature={best_temp:.1f}")
    logger.info(f"  2. Re-run: uv run python scripts/fix_v6_models.py")
    logger.info(f"  3. Compare with V8 when training completes")

    logger.success(f"\n✅ Temperature tuning complete!")


if __name__ == "__main__":
    main()
