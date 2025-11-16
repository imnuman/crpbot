# AWS SageMaker Training Setup for V7 Models

## Overview

This document provides complete setup for training V7 Enhanced models on AWS SageMaker with **guaranteed** feature normalization and proper calibration.

---

## Why SageMaker?

- **Managed Training**: Automatic instance provisioning, monitoring, and cleanup
- **Cost Efficient**: Pay only for training time, auto-shutdown when complete
- **Scalable**: Easy to scale to multiple GPUs or distributed training
- **Reproducible**: Containerized environment ensures consistent results
- **Integrated**: Direct S3 integration for data and model artifacts

---

## Instance Recommendation

**Instance Type**: `ml.g5.xlarge`
- **GPU**: 1x NVIDIA A10G (24GB VRAM)
- **vCPUs**: 4
- **RAM**: 16GB
- **Cost**: ~$1.41/hour (on-demand)
- **Training Time**: 6-9 hours for 3 models = ~$12-14 total

---

## Architecture: V7 Enhanced FNN

### Critical Fix: Feature Normalization Integration

```python
class V7EnhancedFNN(nn.Module):
    """V7 Enhanced FNN with MANDATORY feature normalization."""

    def __init__(self, input_size=72, scaler=None):
        super().__init__()

        # CRITICAL: Store scaler in model
        self.scaler = scaler  # ← Must be provided during initialization
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
        """Forward pass with automatic normalization.

        Args:
            x: Input features (raw, unnormalized)
            apply_scaling: Whether to apply StandardScaler (default: True)
        """
        if len(x.shape) == 3:
            x = x[:, -1, :]

        # CRITICAL: Normalize input using StandardScaler
        if apply_scaling and self.scaler is not None:
            # Convert to numpy, scale, convert back to tensor
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


class V7Trainer:
    """Trainer with mandatory feature normalization."""

    def __init__(self, input_size=72, device='cuda'):
        self.input_size = input_size
        self.device = device
        self.scaler = None
        self.model = None

    def prepare_data(self, X_train, y_train, X_val, y_val):
        """Prepare data with feature normalization.

        CRITICAL: This MUST be called before model initialization!
        """
        from sklearn.preprocessing import StandardScaler

        # Fit scaler on training data only
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Log scaling statistics
        logger.info("Feature Normalization Applied:")
        logger.info(f"  Raw feature range: [{X_train.min():.2f}, {X_train.max():.2f}]")
        logger.info(f"  Scaled feature range: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
        logger.info(f"  Scaler mean shape: {self.scaler.mean_.shape}")
        logger.info(f"  Scaler scale shape: {self.scaler.scale_.shape}")

        return X_train_scaled, X_val_scaled

    def initialize_model(self):
        """Initialize model with scaler.

        CRITICAL: prepare_data() must be called first!
        """
        if self.scaler is None:
            raise ValueError("Must call prepare_data() before initialize_model()!")

        self.model = V7EnhancedFNN(
            input_size=self.input_size,
            scaler=self.scaler  # ← Pass scaler to model
        )
        self.model.to(self.device)

        logger.info("✅ Model initialized with scaler")
        return self.model

    def save_checkpoint(self, path, epoch, optimizer, loss):
        """Save checkpoint with scaler included.

        CRITICAL: Scaler MUST be saved in checkpoint!
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'input_size': self.input_size,
            'scaler': self.scaler,  # ← CRITICAL: Save scaler
            'scaler_mean': self.scaler.mean_,
            'scaler_scale': self.scaler.scale_
        }

        torch.save(checkpoint, path)
        logger.info(f"✅ Checkpoint saved with scaler: {path}")

        # Verify scaler is in checkpoint
        loaded = torch.load(path, map_location='cpu')
        if 'scaler' not in loaded:
            raise ValueError("CRITICAL ERROR: Scaler not found in checkpoint!")

        logger.success("✅ Scaler verified in checkpoint")
```

---

## Training Script: `sagemaker_train.py`

```python
#!/usr/bin/env python3
"""SageMaker training script for V7 Enhanced models."""

import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from loguru import logger
import joblib


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


def train_v7_model(symbol, data_dir, output_dir, epochs=30, batch_size=64, lr=0.001):
    """Train V7 Enhanced model with proper normalization."""

    logger.info(f"Training V7 Enhanced model for {symbol}")
    logger.info("="*80)

    # Load features
    features_path = Path(data_dir) / f"features_{symbol}_1m_latest.parquet"
    df = pd.read_parquet(features_path)
    logger.info(f"Loaded {len(df)} samples")

    # Engineer features
    from apps.trainer.amazon_q_features import engineer_amazon_q_features
    df = engineer_amazon_q_features(df)

    # Feature list (72 features)
    features = [
        'atr_14', 'bb_lower_20', 'bb_lower_50', 'bb_position_20', 'bb_position_50',
        # ... (all 72 features)
    ]

    # Prepare data
    X = df[features].values
    y = df['target'].values  # Assuming 3-class target (0=DOWN, 1=NEUTRAL, 2=UP)

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

    # Initialize model (scaler passed in __init__)
    model = V7EnhancedFNN(input_size=72, scaler=scaler).to(device)

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

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save checkpoint with scaler
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss.item(),
                'accuracy': val_acc,
                'input_size': 72,
                'scaler': scaler,  # ← CRITICAL
                'scaler_mean': scaler.mean_,
                'scaler_scale': scaler.scale_
            }

            checkpoint_path = Path(output_dir) / f"lstm_{symbol}_v7_enhanced.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"✅ Saved checkpoint: {checkpoint_path}")

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

        # Check confidence distribution
        test_probs = torch.softmax(test_outputs, dim=-1)
        max_probs = test_probs.max(dim=-1)[0]
        pct_over_99 = (max_probs > 0.99).float().mean().item() * 100

    logger.success(f"\n{'='*80}")
    logger.success(f"Training Complete: {symbol}")
    logger.success(f"  Test Accuracy: {test_acc:.3f}")
    logger.success(f"  Overconfidence (>99%): {pct_over_99:.1f}%")
    logger.success(f"  Checkpoint: {checkpoint_path}")
    logger.success(f"{'='*80}\n")

    return {
        'symbol': symbol,
        'test_accuracy': test_acc,
        'overconfidence_pct': pct_over_99,
        'checkpoint_path': str(checkpoint_path)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='/opt/ml/input/data/training')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/model')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()

    result = train_v7_model(
        symbol=args.symbol,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

    print(f"\n✅ Training complete: {result}")
```

---

## SageMaker Setup Steps

### 1. Create SageMaker Execution Role

```bash
# Create IAM role for SageMaker
aws iam create-role \
  --role-name CRPBot-SageMaker-ExecutionRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Principal": {"Service": "sagemaker.amazonaws.com"},
        "Action": "sts:AssumeRole"
      }
    ]
  }'

# Attach policies
aws iam attach-role-policy \
  --role-name CRPBot-SageMaker-ExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy \
  --role-name CRPBot-SageMaker-ExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

### 2. Upload Training Data to S3

```bash
# Create S3 bucket for training data
aws s3 mb s3://crpbot-sagemaker-training

# Upload feature data
aws s3 cp data/features/ s3://crpbot-sagemaker-training/data/features/ --recursive

# Upload training script
aws s3 cp sagemaker_train.py s3://crpbot-sagemaker-training/code/
```

### 3. Create Training Job (via Boto3)

```python
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = 'arn:aws:iam::ACCOUNT_ID:role/CRPBot-SageMaker-ExecutionRole'

# Define estimator
pytorch_estimator = PyTorch(
    entry_point='sagemaker_train.py',
    source_dir='.',
    role=role,
    instance_type='ml.g5.xlarge',
    instance_count=1,
    framework_version='2.1.0',
    py_version='py310',
    hyperparameters={
        'symbol': 'BTC-USD',
        'epochs': 30,
        'batch-size': 64,
        'lr': 0.001
    },
    output_path='s3://crpbot-sagemaker-training/models/',
    base_job_name='crpbot-v7-training'
)

# Start training
pytorch_estimator.fit({
    'training': 's3://crpbot-sagemaker-training/data/features/'
})
```

### 4. Monitor Training

```bash
# Via AWS CLI
aws sagemaker describe-training-job --training-job-name crpbot-v7-training-XXXXX

# Via CloudWatch Logs
aws logs tail /aws/sagemaker/TrainingJobs --follow
```

### 5. Download Trained Models

```bash
# After training completes
aws s3 cp s3://crpbot-sagemaker-training/models/crpbot-v7-training-XXXXX/output/model.tar.gz .

# Extract models
tar -xzf model.tar.gz

# Verify scaler is present
python -c "
import torch
checkpoint = torch.load('lstm_BTC-USD_v7_enhanced.pt', map_location='cpu')
print('Scaler present:', 'scaler' in checkpoint)
print('Scaler mean shape:', checkpoint['scaler'].mean_.shape)
"
```

---

## Quality Gates Verification

After training, run diagnostic:

```bash
uv run python scripts/diagnose_v7_model.py
```

**Required Metrics**:
- ✅ Logit range: ≤20 (not ±158,000!)
- ✅ Overconfidence (>99%): <10%
- ✅ Class balance: No class >60%
- ✅ Scaler present in checkpoint
- ✅ Test accuracy: ≥68%

---

## Cost Estimate

**ml.g5.xlarge**: $1.41/hour

- BTC-USD: ~2-3 hours = $4.23
- ETH-USD: ~2-3 hours = $4.23
- SOL-USD: ~2-3 hours = $4.23

**Total**: ~$12-14 for all 3 models

---

## Critical Checklist

Before starting training, verify:

- [ ] `StandardScaler` is created and fitted on training data
- [ ] Scaler is passed to model in `__init__()`
- [ ] Scaler is saved in checkpoint
- [ ] Model's `forward()` applies scaling (or data is pre-scaled)
- [ ] Diagnostic script verifies scaler presence
- [ ] Logit ranges are checked (<20)

**If any checkbox is unchecked, DO NOT proceed with training!**

---

## Troubleshooting

### Scaler Not Found in Checkpoint

```python
# Add explicit verification after save:
checkpoint = torch.load(checkpoint_path, map_location='cpu')
assert 'scaler' in checkpoint, "CRITICAL: Scaler not in checkpoint!"
assert hasattr(checkpoint['scaler'], 'mean_'), "CRITICAL: Invalid scaler!"
```

### Extreme Logits (>100)

```python
# Check if scaling is actually applied:
with torch.no_grad():
    sample = torch.randn(1, 72).to(device)
    output = model(sample, apply_scaling=True)  # ← Must use scaled input
    print(f"Logit range: ±{output.abs().max().item():.2f}")  # Should be <20
```

### Out of Memory

```bash
# Reduce batch size
--batch-size 32  # Instead of 64
```

---

## Next Steps After Training

1. Download models from S3
2. Run diagnostic to verify quality gates
3. Promote to production if all gates pass
4. Deploy to cloud server
5. Restart runtime with V7 models
6. Monitor first signals for realistic confidence (60-85%)
