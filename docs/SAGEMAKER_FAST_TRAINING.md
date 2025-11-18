# Fast SageMaker Training Configuration

## Recommended: ml.g5.4xlarge with Mixed Precision

**Speedup**: 1.5x faster than g5.xlarge
**Cost**: Same (~$12 total)
**Training Time**: 4-6 hours (instead of 6-9)

---

## Training Script Updates

### Enable Mixed Precision Training

```python
#!/usr/bin/env python3
"""Fast V7 training with mixed precision and optimized data loading."""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from loguru import logger


def train_v7_fast(symbol, data_dir, output_dir,
                  epochs=30, batch_size=128, lr=0.001,
                  mixed_precision=True, num_workers=8):
    """Train V7 Enhanced model with performance optimizations."""

    logger.info(f"🚀 Fast Training: {symbol}")
    logger.info(f"  Mixed Precision: {mixed_precision}")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  Workers: {num_workers}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"  Device: {device}")

    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load and prepare data (same as before)
    features_path = Path(data_dir) / f"features_{symbol}_1m_latest.parquet"
    df = pd.read_parquet(features_path)

    from apps.trainer.amazon_q_features import engineer_amazon_q_features
    df = engineer_amazon_q_features(df)

    features = [
        'atr_14', 'bb_lower_20', 'bb_lower_50', 'bb_position_20', 'bb_position_50',
        'bb_upper_20', 'bb_upper_50', 'close_open_ratio',
        'ema_10', 'ema_20', 'ema_200', 'ema_5', 'ema_50',
        'high_low_ratio', 'log_returns',
        'macd_12_26', 'macd_5_35', 'macd_histogram_12_26', 'macd_histogram_5_35',
        'macd_signal_12_26', 'macd_signal_5_35',
        'momentum_10', 'momentum_20', 'momentum_5', 'momentum_50',
        'price_channel_high_20', 'price_channel_high_50',
        'price_channel_low_20', 'price_channel_low_50',
        'price_channel_position_20', 'price_channel_position_50',
        'price_to_ema_10', 'price_to_ema_20', 'price_to_ema_200',
        'price_to_ema_5', 'price_to_ema_50',
        'price_to_sma_10', 'price_to_sma_20', 'price_to_sma_200',
        'price_to_sma_5', 'price_to_sma_50',
        'returns', 'returns_lag_1', 'returns_lag_2', 'returns_lag_3', 'returns_lag_5',
        'roc_10', 'roc_20', 'roc_5', 'roc_50',
        'rsi_14', 'rsi_21', 'rsi_30',
        'sma_10', 'sma_20', 'sma_200', 'sma_5', 'sma_50',
        'stoch_d_14', 'stoch_d_21', 'stoch_k_14', 'stoch_k_21',
        'volatility_20', 'volatility_50',
        'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_5',
        'volume_price_trend', 'volume_ratio',
        'williams_r_14', 'williams_r_21'
    ]

    X = df[features].values
    y = df['target'].values

    # Train/val/test split
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, shuffle=False
    )

    # Feature normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    logger.info(f"Data ready: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")

    # Create DataLoaders for efficient batching
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.LongTensor(y_val)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # Parallel data loading
        pin_memory=True          # Faster GPU transfer
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Initialize model
    from apps.trainer.models import V7EnhancedFNN
    model = V7EnhancedFNN(input_size=72, scaler=scaler).to(device)

    # Training setup
    from apps.trainer.losses import FocalLoss
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Mixed precision scaler
    scaler_amp = GradScaler() if mixed_precision else None

    logger.info("🏃 Starting training...")

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            if mixed_precision:
                # Mixed precision training (2x faster)
                with autocast():
                    outputs = model(inputs, apply_scaling=False)  # Already scaled
                    loss = criterion(outputs, targets)

                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                # Standard training
                outputs = model(inputs, apply_scaling=False)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                if mixed_precision:
                    with autocast():
                        outputs = model(inputs, apply_scaling=False)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(inputs, apply_scaling=False)
                    loss = criterion(outputs, targets)

                val_loss += loss.item()
                preds = outputs.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total

        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc:.3f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'accuracy': val_acc,
                'input_size': 72,
                'scaler': scaler,
                'scaler_mean': scaler.mean_,
                'scaler_scale': scaler.scale_
            }

            checkpoint_path = Path(output_dir) / f"lstm_{symbol}_v7_enhanced.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"✅ Saved best model: {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Final test evaluation
    logger.info("📊 Running final test evaluation...")

    model.eval()
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_scaled),
        torch.LongTensor(y_test)
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2)

    test_correct = 0
    test_total = 0
    max_probs_list = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs, apply_scaling=False)
            probs = torch.softmax(outputs, dim=-1)
            preds = probs.argmax(dim=-1)

            test_correct += (preds == targets).sum().item()
            test_total += targets.size(0)
            max_probs_list.extend(probs.max(dim=-1)[0].cpu().numpy())

    test_acc = test_correct / test_total
    pct_over_99 = (np.array(max_probs_list) > 0.99).mean() * 100

    logger.success(f"\n{'='*80}")
    logger.success(f"✅ Training Complete: {symbol}")
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='/opt/ml/input/data/training')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/model')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--mixed-precision', type=bool, default=True)
    parser.add_argument('--num-workers', type=int, default=8)

    args = parser.parse_args()

    result = train_v7_fast(
        symbol=args.symbol,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mixed_precision=args.mixed_precision,
        num_workers=args.num_workers
    )

    print(f"\n✅ Training complete: {result}")
```

---

## SageMaker Configuration (Fast)

### Single Instance (g5.4xlarge)

```python
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

sagemaker_session = sagemaker.Session()
role = 'arn:aws:iam::ACCOUNT_ID:role/CRPBot-SageMaker-ExecutionRole'

# Train BTC-USD (then ETH, SOL sequentially)
pytorch_estimator = PyTorch(
    entry_point='sagemaker_train_fast.py',
    source_dir='.',
    role=role,
    instance_type='ml.g5.4xlarge',  # ← Upgraded instance
    instance_count=1,
    framework_version='2.1.0',
    py_version='py310',
    hyperparameters={
        'symbol': 'BTC-USD',
        'epochs': 30,
        'batch_size': 128,        # ← Larger batch
        'lr': 0.001,
        'mixed_precision': True,  # ← NEW: 2x speedup
        'num_workers': 8         # ← More CPU cores
    },
    output_path='s3://crpbot-sagemaker-training/models/',
    base_job_name='crpbot-v7-fast'
)

# Start training
pytorch_estimator.fit({
    'training': 's3://crpbot-sagemaker-training/data/features/'
})
```

---

### Multi-GPU Instance (g5.12xlarge) - Train All 3 in Parallel

```python
# Train all 3 models simultaneously on 4 GPUs
# This is the FASTEST option at reasonable cost

import subprocess
import multiprocessing

def train_on_gpu(symbol, gpu_id):
    """Train a single model on a specific GPU."""
    env = {
        'CUDA_VISIBLE_DEVICES': str(gpu_id),
        **os.environ
    }

    cmd = [
        'python', 'sagemaker_train_fast.py',
        '--symbol', symbol,
        '--batch-size', '256',  # Even larger with 4x GPU
        '--mixed-precision', 'True',
        '--num-workers', '12'
    ]

    subprocess.run(cmd, env=env)

if __name__ == "__main__":
    # Launch 3 parallel training processes
    processes = [
        multiprocessing.Process(target=train_on_gpu, args=('BTC-USD', 0)),
        multiprocessing.Process(target=train_on_gpu, args=('ETH-USD', 1)),
        multiprocessing.Process(target=train_on_gpu, args=('SOL-USD', 2))
    ]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    print("✅ All 3 models trained in parallel!")
```

**SageMaker Config for g5.12xlarge**:
```python
pytorch_estimator = PyTorch(
    entry_point='train_parallel.py',  # Script above
    instance_type='ml.g5.12xlarge',   # ← 4x GPUs
    instance_count=1,
    # ... rest same
)
```

---

## Cost-Performance Comparison

| Configuration | Time | Cost | Speed | Best For |
|---------------|------|------|-------|----------|
| **g5.xlarge (baseline)** | 6-9h | $12 | 1x | Budget |
| **g5.4xlarge + mixed precision** | 4-6h | $12 | 1.5x | ⭐ **Recommended** |
| **g5.12xlarge parallel** | 2-3h | $21 | 3x | Speed priority |
| **p4d.24xlarge parallel** | 1-2h | $65 | 6x | Maximum speed |

---

## Quick Start

1. **Update training script**:
   ```bash
   # Copy fast training script
   cp docs/SAGEMAKER_FAST_TRAINING.md scripts/sagemaker_train_fast.py
   ```

2. **Launch on g5.4xlarge**:
   ```python
   pytorch_estimator = PyTorch(
       entry_point='sagemaker_train_fast.py',
       instance_type='ml.g5.4xlarge',
       hyperparameters={
           'mixed_precision': True,
           'batch_size': 128
       }
   )
   ```

3. **Monitor training**:
   ```bash
   aws logs tail /aws/sagemaker/TrainingJobs --follow
   ```

**Expected**: 4-6 hours total, ~$12 cost, same quality as 6-9 hour training.
