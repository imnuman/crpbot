# AWS g5.xlarge GPU Training Setup for Amazon Q

## Overview
This document outlines the complete setup for training V7 models on AWS g5.xlarge with proper feature normalization and calibration.

## Instance Specifications

**Instance Type:** `g5.xlarge`
- **GPU:** 1x NVIDIA A10G (24GB VRAM)
- **vCPUs:** 4
- **RAM:** 16GB
- **Storage:** 250GB gp3 SSD (minimum)
- **Est. Cost:** ~$1.006/hour on-demand
- **Region:** us-east-1 (recommended for lowest latency)

## Step 1: Launch EC2 Instance

```bash
# Launch g5.xlarge with Deep Learning AMI
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \  # Deep Learning AMI (Ubuntu 20.04)
  --instance-type g5.xlarge \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxx \
  --subnet-id subnet-xxxxxxxx \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":250,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=crpbot-training}]'
```

## Step 2: Initial Setup on Instance

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@<instance-ip>

# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Verify NVIDIA drivers
nvidia-smi  # Should show A10G GPU

# Verify CUDA
nvcc --version  # Should show CUDA 11.8+

# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

## Step 3: Clone Repository and Setup

```bash
# Clone repository
git clone https://github.com/imnuman/crpbot.git
cd crpbot

# Set up Python environment with UV
uv venv .venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
uv pip install -e .

# Verify PyTorch GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
# Should output: CUDA available: True, GPU: NVIDIA A10G
```

## Step 4: Configure Environment

```bash
# Copy .env.example to .env
cp .env.example .env

# Edit .env with your API keys
nano .env
```

Required environment variables:
```bash
# Coinbase API
COINBASE_API_KEY_NAME="organizations/.../apiKeys/..."
COINBASE_PRIVATE_KEY="-----BEGIN EC PRIVATE KEY-----\n...\n-----END EC PRIVATE KEY-----"

# CoinGecko API
COINGECKO_API_KEY="CG-..."

# Training config
DEVICE="cuda"
BATCH_SIZE=64  # Increased for GPU
NUM_WORKERS=4
EPOCHS=30
LEARNING_RATE=0.001
```

## Step 5: Prepare Training Data

```bash
# Download training data (if not present)
./scripts/download_training_data.sh

# Or fetch fresh data from Coinbase
uv run python scripts/fetch_data.py --symbol BTC-USD --interval 1m --start 2023-11-10 --output data/raw
uv run python scripts/fetch_data.py --symbol ETH-USD --interval 1m --start 2023-11-10 --output data/raw
uv run python scripts/fetch_data.py --symbol SOL-USD --interval 1m --start 2023-11-10 --output data/raw

# Engineer features with Amazon Q's 72-feature set
uv run python scripts/engineer_features.py --symbol BTC-USD --normalize
uv run python scripts/engineer_features.py --symbol ETH-USD --normalize
uv run python scripts/engineer_features.py --symbol SOL-USD --normalize
```

## Step 6: Model Training Configuration

### V7 Model Improvements (vs V6 Enhanced)

1. **Feature Normalization**:
   ```python
   from sklearn.preprocessing import StandardScaler

   # Fit scaler on training data
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)

   # Save scaler for inference
   joblib.dump(scaler, 'models/scaler_BTC-USD.pkl')
   ```

2. **Label Smoothing**:
   ```python
   # Instead of hard labels [0, 0, 1], use soft labels
   # Down: [0.95, 0.025, 0.025]
   # Up: [0.025, 0.025, 0.95]
   label_smoothing = 0.05
   ```

3. **Temperature Scaling**:
   ```python
   class V7EnhancedFNN(nn.Module):
       def __init__(self, input_size=72, temperature=2.5):
           super().__init__()
           self.temperature = nn.Parameter(torch.tensor(temperature))
           # ... layers ...

       def forward(self, x):
           logits = self.fc4(x)
           return logits / self.temperature  # Calibrated logits
   ```

4. **Dropout Regularization**:
   ```python
   self.dropout = nn.Dropout(0.3)
   x = self.relu(self.fc1(x))
   x = self.dropout(x)  # Apply after each layer
   ```

5. **Focal Loss** (instead of CrossEntropyLoss):
   ```python
   class FocalLoss(nn.Module):
       def __init__(self, alpha=0.25, gamma=2.0):
           super().__init__()
           self.alpha = alpha
           self.gamma = gamma

       def forward(self, inputs, targets):
           ce_loss = F.cross_entropy(inputs, targets, reduction='none')
           pt = torch.exp(-ce_loss)
           focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
           return focal_loss.mean()
   ```

## Step 7: Run Training (Managed by Amazon Q)

```bash
# Amazon Q will run this command with proper configuration
uv run python apps/trainer/main.py \
  --task v7_enhanced \
  --symbol BTC-USD \
  --epochs 30 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --device cuda \
  --normalize \
  --label-smoothing 0.05 \
  --dropout 0.3 \
  --temperature 2.5 \
  --focal-loss \
  --early-stopping-patience 5 \
  --checkpoint-dir models/v7 \
  --log-dir logs/v7
```

## Step 8: Monitor Training

```bash
# Monitor GPU utilization
watch -n 1 nvidia-smi

# Monitor training logs
tail -f logs/v7/training_BTC-USD.log

# TensorBoard (optional)
tensorboard --logdir logs/v7 --port 6006
```

## Step 9: Model Evaluation

```bash
# Run model diagnostic on trained V7 model
uv run python scripts/diagnose_v7_model.py \
  --model models/v7/lstm_BTC-USD_v7_enhanced.pt \
  --features data/features/features_BTC-USD_1m_latest.parquet

# Expected output:
# - Logits in range: ±10 (not ±40,000!)
# - Confidence distribution: 50-90% (not 100%)
# - Class balance: 40-60% down/neutral/up (not 100% down)
# - Calibration error: <5%
```

## Step 10: Model Promotion

```bash
# If V7 model passes all gates, promote to production
cp models/v7/lstm_BTC-USD_v7_enhanced.pt models/promoted/
cp models/v7/scaler_BTC-USD.pkl models/promoted/
```

## Cost Estimate

**Training Time:**
- Per model (BTC/ETH/SOL): ~2-3 hours @ 30 epochs
- Total for 3 models: ~6-9 hours

**Total Cost:**
- Instance cost: $1.006/hour × 9 hours = ~$9.05
- Storage: $0.10/GB-month × 250GB × 0.3 days = ~$2.50
- **Total: ~$11.55** for complete V7 training

## Amazon Q Instructions

**Amazon Q, please execute the following:**

1. Launch g5.xlarge instance in us-east-1
2. Set up CUDA/PyTorch environment
3. Clone crpbot repository
4. Download or verify training data (2 years, 1-minute candles)
5. Engineer Amazon Q's 72 features **with StandardScaler normalization**
6. Train V7 Enhanced FNN models for BTC-USD, ETH-USD, SOL-USD with:
   - Label smoothing (0.05)
   - Dropout (0.3)
   - Temperature scaling (2.5)
   - Focal loss
   - Early stopping (patience=5)
7. Run diagnostic to verify models are properly calibrated
8. Upload trained models + scalers to S3 or download to local
9. Terminate instance when complete

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
BATCH_SIZE=32  # Instead of 64

# Or use gradient accumulation
ACCUMULATION_STEPS=2
```

### Slow Training
```bash
# Enable mixed precision
uv pip install accelerate
# Add --mixed-precision fp16 to training command
```

### Connection Lost
```bash
# Use tmux to persist session
tmux new -s training
# Training commands here
# Ctrl+B, D to detach
# tmux attach -t training to reattach
```
