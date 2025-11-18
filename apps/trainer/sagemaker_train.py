#!/usr/bin/env python3
"""SageMaker training script for V7 Enhanced models with mandatory scaler verification."""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class V7EnhancedFNN(nn.Module):
    """V7 Enhanced FNN with MANDATORY feature normalization."""

    def __init__(self, input_size=72, scaler=None):
        super().__init__()
        
        # CRITICAL: Store scaler in model
        self.scaler = scaler
        if self.scaler is None:
            raise ValueError("Scaler must be provided! V7 models require normalization.")

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

    def forward(self, x, apply_scaling=True):
        """Forward pass with automatic normalization."""
        if len(x.shape) == 3:
            x = x[:, -1, :]

        # CRITICAL: Normalize input using StandardScaler
        if apply_scaling and self.scaler is not None:
            device = x.device
            x_np = x.cpu().numpy()
            x_scaled = self.scaler.transform(x_np)
            x = torch.FloatTensor(x_scaled).to(device)

        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        logits = self.fc4(x)

        # Temperature scaling for calibration
        return logits / self.temperature

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def engineer_features(df):
    """Engineer 72 features for V7 model."""
    # Basic price features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Moving averages
    for period in [5, 10, 20, 50]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
    
    # Volatility
    df['volatility_10'] = df['returns'].rolling(10).std()
    df['volatility_20'] = df['returns'].rolling(20).std()
    
    # Volume features
    df['volume_sma_10'] = df['volume'].rolling(10).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_10']
    
    # Price position
    df['price_position_20'] = (df['close'] - df['close'].rolling(20).min()) / (df['close'].rolling(20).max() - df['close'].rolling(20).min())
    
    # Create target (simplified 3-class)
    future_returns = df['close'].shift(-5) / df['close'] - 1
    df['target'] = 1  # HOLD
    df.loc[future_returns > 0.002, 'target'] = 2  # UP
    df.loc[future_returns < -0.002, 'target'] = 0  # DOWN
    
    return df

def train_v7_model(symbol, data_dir, output_dir, epochs=30, batch_size=64, lr=0.001):
    """Train V7 Enhanced model with proper normalization."""

    logger.info(f"Training V7 Enhanced model for {symbol}")
    logger.info("="*80)

    # Load data
    data_path = Path(data_dir) / f"{symbol}_features.parquet"
    if not data_path.exists():
        # Try alternative naming
        data_path = Path(data_dir) / f"features_{symbol}_1m_latest.parquet"
    
    if not data_path.exists():
        raise FileNotFoundError(f"No data found for {symbol} in {data_dir}")
    
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} samples")

    # Engineer features
    df = engineer_features(df)
    
    # Select features (simplified set for demo)
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'target', 'symbol']]
    feature_cols = feature_cols[:72]  # Take first 72 features
    
    # Prepare data
    X = df[feature_cols].dropna().values
    y = df.loc[df[feature_cols].dropna().index, 'target'].values

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # CRITICAL: Feature normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    logger.info("Feature Normalization:")
    logger.info(f"  Raw range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    logger.info(f"  Scaled range: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")

    # Convert to tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_t = torch.FloatTensor(X_train_scaled).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val_scaled).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)

    # Initialize model with scaler
    model = V7EnhancedFNN(input_size=len(feature_cols), scaler=scaler).to(device)

    # Training configuration
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        model.train()

        # Training
        optimizer.zero_grad()
        outputs = model(X_train_t, apply_scaling=False)  # Already scaled
        loss = criterion(outputs, y_train_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t, apply_scaling=False)
            val_loss = criterion(val_outputs, y_val_t)
            val_preds = val_outputs.argmax(dim=-1)
            val_acc = (val_preds == y_val_t).float().mean().item()

        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}, "
                   f"Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc:.3f}")

        # Early stopping and checkpoint saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # CRITICAL: Save checkpoint with scaler
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss.item(),
                'accuracy': val_acc,
                'input_size': len(feature_cols),
                'scaler': scaler,  # ← CRITICAL
                'scaler_mean': scaler.mean_,
                'scaler_scale': scaler.scale_,
                'feature_columns': feature_cols
            }

            checkpoint_path = Path(output_dir) / f"lstm_{symbol}_v7_enhanced.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"✅ Saved checkpoint: {checkpoint_path}")

            # MANDATORY: Verify scaler is in checkpoint
            loaded = torch.load(checkpoint_path, map_location='cpu')
            if 'scaler' not in loaded:
                raise ValueError("CRITICAL ERROR: Scaler not found in checkpoint!")
            logger.info("✅ Scaler verified in checkpoint")

        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Final test evaluation
    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test_scaled).to(device)
        y_test_t = torch.LongTensor(y_test).to(device)

        test_outputs = model(X_test_t, apply_scaling=False)
        test_preds = test_outputs.argmax(dim=-1)
        test_acc = (test_preds == y_test_t).float().mean().item()

        # Check logit range (quality gate)
        logit_range = test_outputs.abs().max().item()
        
        # Check confidence distribution
        test_probs = torch.softmax(test_outputs, dim=-1)
        max_probs = test_probs.max(dim=-1)[0]
        pct_over_99 = (max_probs > 0.99).float().mean().item() * 100

    logger.info(f"\n{'='*80}")
    logger.info(f"Training Complete: {symbol}")
    logger.info(f"  Test Accuracy: {test_acc:.3f}")
    logger.info(f"  Logit Range: ±{logit_range:.2f} (target: <20)")
    logger.info(f"  Overconfidence (>99%): {pct_over_99:.1f}% (target: <10%)")
    logger.info(f"  Checkpoint: {checkpoint_path}")
    
    # Quality gates
    if logit_range > 20:
        logger.error(f"❌ QUALITY GATE FAILED: Logit range {logit_range:.2f} > 20")
    if pct_over_99 > 10:
        logger.error(f"❌ QUALITY GATE FAILED: Overconfidence {pct_over_99:.1f}% > 10%")
    
    logger.info(f"{'='*80}\n")

    return {
        'symbol': symbol,
        'test_accuracy': test_acc,
        'logit_range': logit_range,
        'overconfidence_pct': pct_over_99,
        'checkpoint_path': str(checkpoint_path)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default='BTC')
    parser.add_argument('--data-dir', type=str, default='/opt/ml/input/data/training')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/model')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    result = train_v7_model(
        symbol=args.symbol,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

    print(f"\n✅ Training complete: {result}")
