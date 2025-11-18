#!/usr/bin/env python3
"""
Quick Fix for V6 Models
Addresses critical issues without retraining:
1. Feature normalization (StandardScaler)
2. Logit clamping (prevent explosion)
3. Temperature scaling (calibrate confidence)
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
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


class V6FixedWrapper(nn.Module):
    """Wrapper that adds normalization and temperature scaling to V6 models."""

    def __init__(self, base_model, scaler, temperature=10.0, logit_clip=15.0):
        super().__init__()
        self.base_model = base_model
        self.scaler = scaler
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.logit_clip = logit_clip

    def forward(self, x):
        """
        Forward pass with fixes:
        1. Normalize features (if scaler provided)
        2. Get base model predictions
        3. Clamp logits to prevent explosion
        4. Apply temperature scaling
        """
        # Get raw logits from base model
        logits = self.base_model(x)

        # Clamp logits to prevent numerical overflow
        logits = torch.clamp(logits, -self.logit_clip, self.logit_clip)

        # Apply temperature scaling
        logits = logits / self.temperature

        return logits


def fit_scaler(features_path: Path, feature_cols: list) -> StandardScaler:
    """Fit StandardScaler on training data."""
    logger.info(f"Fitting scaler on {features_path.name}")

    df = pd.read_parquet(features_path)

    # Engineer Amazon Q features
    from apps.trainer.amazon_q_features import engineer_amazon_q_features
    df = engineer_amazon_q_features(df)

    # Extract feature values
    X = df[feature_cols].values

    # Remove NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    # Fit scaler
    scaler = StandardScaler()
    scaler.fit(X)

    logger.info(f"Scaler fitted: mean={scaler.mean_[:5]}, std={scaler.scale_[:5]}")

    return scaler


def load_and_fix_model(model_path: Path, scaler_path: Path, symbol: str):
    """Load V6 model and wrap with fixes."""
    logger.info(f"Loading model: {model_path.name}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    input_size = checkpoint.get('input_size', 72)

    # Create base model
    base_model = V6EnhancedFNN(input_size=input_size)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model.eval()

    # Load or create scaler
    if scaler_path.exists():
        logger.info(f"Loading scaler: {scaler_path.name}")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    else:
        logger.warning(f"Scaler not found, model will use unnormalized features!")
        scaler = None

    # Wrap with fixes
    fixed_model = V6FixedWrapper(
        base_model=base_model,
        scaler=scaler,
        temperature=1.0,  # Optimal temperature from tuning
        logit_clip=15.0
    )
    fixed_model.eval()

    logger.info(f"✅ Fixed model created for {symbol}")
    logger.info(f"  Temperature: {fixed_model.temperature.item():.2f}")
    logger.info(f"  Logit clip: ±{fixed_model.logit_clip}")

    return fixed_model


def test_fixed_model(model, features_path: Path, feature_cols: list, symbol: str):
    """Test fixed model predictions."""
    logger.info(f"\nTesting fixed model for {symbol}...")

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

            # Normalize if scaler available
            if model.scaler is not None:
                features = model.scaler.transform(features)

            # Convert to tensor
            sample = torch.FloatTensor(features).unsqueeze(0)

            # Get predictions
            logits = model(sample)
            probs = torch.softmax(logits, dim=-1)

            logits_list.append(logits.squeeze().numpy())
            probs_list.append(probs.squeeze().numpy())

    logits_array = np.array(logits_list)
    probs_array = np.array(probs_list)

    # Analyze results
    logger.info(f"\n{'='*60}")
    logger.info(f"FIXED MODEL RESULTS - {symbol}")
    logger.info(f"{'='*60}")

    logger.info("\nLogit Statistics (100 samples):")
    logger.info(f"  Down:    mean={logits_array[:, 0].mean():>7.2f}, std={logits_array[:, 0].std():>6.2f}, "
               f"range=[{logits_array[:, 0].min():>7.2f}, {logits_array[:, 0].max():>7.2f}]")
    logger.info(f"  Neutral: mean={logits_array[:, 1].mean():>7.2f}, std={logits_array[:, 1].std():>6.2f}, "
               f"range=[{logits_array[:, 1].min():>7.2f}, {logits_array[:, 1].max():>7.2f}]")
    logger.info(f"  Up:      mean={logits_array[:, 2].mean():>7.2f}, std={logits_array[:, 2].std():>6.2f}, "
               f"range=[{logits_array[:, 2].min():>7.2f}, {logits_array[:, 2].max():>7.2f}]")

    logger.info("\nProbability Statistics:")
    logger.info(f"  Down:    mean={probs_array[:, 0].mean():.3f}, std={probs_array[:, 0].std():.3f}")
    logger.info(f"  Neutral: mean={probs_array[:, 1].mean():.3f}, std={probs_array[:, 1].std():.3f}")
    logger.info(f"  Up:      mean={probs_array[:, 2].mean():.3f}, std={probs_array[:, 2].std():.3f}")

    # Check confidence
    max_probs = probs_array.max(axis=1)
    avg_conf = max_probs.mean()
    pct_over_90 = (max_probs > 0.9).sum() / len(max_probs) * 100
    pct_over_99 = (max_probs > 0.99).sum() / len(max_probs) * 100

    logger.info(f"\nConfidence Distribution:")
    logger.info(f"  Average confidence:  {avg_conf:.1%}")
    logger.info(f"  Predictions > 90%:   {pct_over_90:.1f}%")
    logger.info(f"  Predictions > 99%:   {pct_over_99:.1f}%")

    # Check class distribution
    predicted_classes = probs_array.argmax(axis=1)
    class_counts = np.bincount(predicted_classes, minlength=3)

    logger.info(f"\nPredicted Class Distribution:")
    logger.info(f"  Down (0):    {class_counts[0]:>3d} ({class_counts[0]/len(predicted_classes)*100:>5.1f}%)")
    logger.info(f"  Neutral (1): {class_counts[1]:>3d} ({class_counts[1]/len(predicted_classes)*100:>5.1f}%)")
    logger.info(f"  Up (2):      {class_counts[2]:>3d} ({class_counts[2]/len(predicted_classes)*100:>5.1f}%)")

    # Overall assessment
    logger.info(f"\n{'='*60}")
    if pct_over_90 < 30 and class_counts.min() > 10:
        logger.success(f"✅ {symbol}: Model looks MUCH better!")
    elif pct_over_90 < 70:
        logger.warning(f"⚠️  {symbol}: Improved but still needs work")
    else:
        logger.error(f"❌ {symbol}: Still has issues")
    logger.info(f"{'='*60}\n")

    return {
        'logits': logits_array,
        'probs': probs_array,
        'avg_confidence': avg_conf,
        'pct_over_90': pct_over_90,
        'pct_over_99': pct_over_99,
        'class_distribution': class_counts.tolist()
    }


def main():
    """Fix all V6 models."""
    logger.info("="*80)
    logger.info("V6 MODEL FIX - QUICK PATCH")
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
    model_dir = Path("models/promoted")
    features_dir = Path("data/features")
    output_dir = Path("models/v6_fixed")
    output_dir.mkdir(exist_ok=True)

    results = {}

    for symbol in symbols:
        logger.info(f"\n{'='*80}")
        logger.info(f"PROCESSING {symbol}")
        logger.info(f"{'='*80}\n")

        # Paths
        model_path = model_dir / f"lstm_{symbol}_v6_enhanced.pt"
        features_path = features_dir / f"features_{symbol}_1m_latest.parquet"
        scaler_path = output_dir / f"scaler_{symbol}_v6_fixed.pkl"

        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            continue

        if not features_path.exists():
            logger.error(f"Features not found: {features_path}")
            continue

        # Step 1: Fit scaler
        logger.info("Step 1: Fitting StandardScaler...")
        scaler = fit_scaler(features_path, feature_cols)

        # Save scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Scaler saved: {scaler_path}")

        # Step 2: Load and wrap model
        logger.info("\nStep 2: Loading and wrapping model...")
        fixed_model = load_and_fix_model(model_path, scaler_path, symbol)

        # Step 3: Test fixed model
        logger.info("\nStep 3: Testing fixed model...")
        test_results = test_fixed_model(fixed_model, features_path, feature_cols, symbol)

        # Step 4: Save fixed model
        output_path = output_dir / f"lstm_{symbol}_v6_FIXED.pt"
        torch.save({
            'model_state_dict': fixed_model.state_dict(),
            'base_model_state_dict': fixed_model.base_model.state_dict(),
            'temperature': fixed_model.temperature.item(),
            'logit_clip': fixed_model.logit_clip,
            'input_size': 72,
            'symbol': symbol,
            'version': 'v6_fixed'
        }, output_path)

        logger.success(f"✅ Fixed model saved: {output_path}")

        results[symbol] = test_results

    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY - V6 FIXED MODELS")
    logger.info("="*80)

    for symbol, result in results.items():
        logger.info(f"\n{symbol}:")
        logger.info(f"  Average confidence:  {result['avg_confidence']:.1%}")
        logger.info(f"  Overconfident (>90%): {result['pct_over_90']:.1f}%")
        logger.info(f"  Class distribution:   {result['class_distribution']}")

    logger.success("\n✅ All V6 models fixed!")
    logger.info("Compare with V8 when AWS training completes.")


if __name__ == "__main__":
    main()
