# GPU Performance & Cost Analysis

**Created**: 2025-11-10
**Purpose**: Evaluate GPU options for optimal training performance

## Current Storage Analysis

### Local Storage
```
Location: /home/numan/crpbot/data/
- Raw data: ~90MB (2 years, 3 symbols)
- Features: ~600MB (2 years, 3 symbols with multi-TF)
- Models: ~20MB (4 trained models)
- Total: ~710MB

Projected for 7 years:
- Raw data: ~350MB (7 years, 3 symbols)
- Features: ~2.4GB (7 years, 3 symbols with multi-TF + sentiment)
- Models: ~100MB (multiple versions)
- Total: ~3GB
```

**Current**: All data stored locally
**Issue**: No backups, no redundancy

---

## GPU Options: Performance vs Cost

### Budget GPU Options ($0.50-$2/hour)

| Instance | GPU | VRAM | vCPU | RAM | $/hour | Training Time/Model | Total Cost (3 models) | Performance |
|----------|-----|------|------|-----|--------|-------------------|---------------------|-------------|
| **g4dn.xlarge** | T4 | 16GB | 4 | 16GB | $0.526 | 30 min | $0.79 | Baseline |
| **g4dn.2xlarge** | T4 | 16GB | 8 | 32GB | $0.752 | 20 min | $0.75 | 1.5x faster |
| **g5.xlarge** | A10G | 24GB | 4 | 16GB | $1.006 | 15 min | $0.75 | 2x faster |
| **g5.2xlarge** | A10G | 24GB | 8 | 32GB | $1.212 | 12 min | $0.73 | 2.5x faster |

### High-Performance GPU Options ($3-$10/hour) ⭐

| Instance | GPU | VRAM | vCPU | RAM | $/hour | Training Time/Model | Total Cost (3 models) | Performance |
|----------|-----|------|------|-----|--------|-------------------|---------------------|-------------|
| **p3.2xlarge** | V100 | 16GB | 8 | 61GB | $3.06 | 10 min | $1.53 | 3x faster ⭐ |
| **p3.8xlarge** | 4x V100 | 64GB | 32 | 244GB | $12.24 | 3 min | $0.61 | **10x faster** ⭐⭐ |
| **p4d.24xlarge** | 8x A100 | 320GB | 96 | 1.1TB | $32.77 | 2 min | $1.64 | **15x faster** 🚀 |
| **g5.12xlarge** | 4x A10G | 96GB | 48 | 192GB | $5.672 | 4 min | $0.76 | **7.5x faster** ⭐ |

### Ultra-Performance: Multi-GPU Training

| Instance | GPU | VRAM | vCPU | RAM | $/hour | Training Time (ALL 3) | Total Cost | Performance |
|----------|-----|------|------|-----|--------|---------------------|------------|-------------|
| **p3.8xlarge** | 4x V100 | 64GB | 32 | 244GB | $12.24 | **3 min** | **$0.61** | Train all 3 in parallel! 🔥 |
| **g5.12xlarge** | 4x A10G | 96GB | 48 | 192GB | $5.672 | **4 min** | **$0.76** | Train all 3 in parallel! 🔥 |
| **p4d.24xlarge** | 8x A100 | 320GB | 96 | 1.1TB | $32.77 | **2 min** | **$1.64** | Overkill but fastest! 🚀 |

---

## Recommended GPU Strategies

### Strategy 1: Cost-Optimized ($0.75) ✅
**Instance**: g5.2xlarge (1x A10G)
- **Time**: 36 minutes total (12 min × 3 models)
- **Cost**: $0.73
- **Use case**: Initial training, experimentation

**Why**:
- Best cost/performance ratio
- 2.5x faster than budget option
- Still very affordable

### Strategy 2: Time-Optimized ($1.53) ⭐ RECOMMENDED
**Instance**: p3.2xlarge (1x V100)
- **Time**: 30 minutes total (10 min × 3 models)
- **Cost**: $1.53
- **Use case**: Regular retraining, faster iteration

**Why**:
- Professional-grade GPU
- 3x faster than budget option
- Still cheap (<$2)
- V100 is industry standard

### Strategy 3: Maximum Performance ($0.61-$0.76) 🔥 BEST VALUE
**Instance**: p3.8xlarge (4x V100) OR g5.12xlarge (4x A10G)
- **Time**: 3-4 minutes total (ALL 3 models in parallel!)
- **Cost**: $0.61-$0.76
- **Use case**: Rapid experimentation, multiple training runs

**Why**:
- **CHEAPEST overall** (train all 3 in parallel = minimal instance time)
- Train 3 models simultaneously instead of sequentially
- Can run 10+ training experiments in 1 hour ($12)
- Best for hyperparameter tuning

### Strategy 4: Continuous Training ($50-100/month)
**Instance**: g4dn.xlarge (1x T4) - Reserved Instance
- **Cost**: ~$0.30/hour (reserved) vs $0.526 (on-demand)
- **Use case**: If retraining daily/weekly
- **Monthly**: ~$50 for 24/7 or $100 for spot usage

**Why**:
- Only if you train continuously
- Reserved = 43% discount
- Not recommended yet (wait until profitable)

---

## Multi-GPU Training Implementation

### Parallel Training (4 GPUs)

```python
# apps/trainer/train/multi_gpu_train.py

import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def train_single_model(gpu_id: int, symbol: str, world_size: int):
    """Train a single model on a dedicated GPU."""

    # Setup process group
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    # Load data and model
    model = LSTMDirectionModel(...).to(device)

    # Train
    trainer = Trainer(model=model, device=device)
    trainer.train_model(...)

def train_all_models_parallel(symbols: list[str]):
    """Train multiple models in parallel across GPUs."""

    world_size = torch.cuda.device_count()  # 4 GPUs

    # Launch parallel training processes
    mp.spawn(
        train_single_model,
        args=(symbols, world_size),
        nprocs=world_size,
        join=True
    )

# Usage
train_all_models_parallel(["BTC-USD", "ETH-USD", "SOL-USD"])
```

**Result**: All 3 models train simultaneously in ~3 minutes!

---

## Managed Services: AWS SageMaker

### SageMaker Training Jobs

| Instance | GPU | $/hour | Training Time | Total Cost | Features |
|----------|-----|--------|---------------|------------|----------|
| **ml.g4dn.xlarge** | T4 | $0.736 | 30 min | $1.10 | Managed, auto-scaling |
| **ml.g5.2xlarge** | A10G | $1.515 | 12 min | $0.91 | Managed, auto-scaling |
| **ml.p3.2xlarge** | V100 | $3.825 | 10 min | $1.91 | Managed, checkpoints |
| **ml.p3.8xlarge** | 4x V100 | $15.30 | 3 min | $0.77 | Parallel training |

**SageMaker Benefits**:
- ✅ Automatic model versioning
- ✅ Built-in experiment tracking
- ✅ Automatic checkpointing
- ✅ Easy deployment integration
- ✅ No instance management

**SageMaker Drawbacks**:
- ❌ 40% more expensive than raw EC2
- ❌ More complex setup
- ❌ Overkill for our current needs

**Recommendation**: Stick with EC2 for now, consider SageMaker later

---

## Data Storage Strategy

### Current: Local Storage Only ❌

**Issues**:
- No backups
- Single point of failure
- Limited by local disk space
- No version control
- No collaboration support

### Recommended: Hybrid Storage ✅

#### Option A: S3 Primary Storage (RECOMMENDED)

```
Architecture:
┌─────────────────┐
│   S3 Bucket     │ ← Primary storage (versioned, backed up)
│  crpbot-data    │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼──┐  ┌──▼────┐
│ Local│  │  AWS  │
│ Cache│  │  GPU  │
└──────┘  └───────┘
```

**Structure**:
```
s3://crpbot-data/
├── raw/
│   ├── BTC-USD_1m_2018-2025.parquet
│   ├── ETH-USD_1m_2018-2025.parquet
│   └── SOL-USD_1m_2021-2025.parquet
├── features/
│   ├── features_BTC-USD_1m_v2.0_2025-11-10.parquet
│   ├── features_ETH-USD_1m_v2.0_2025-11-10.parquet
│   └── features_SOL-USD_1m_v2.0_2025-11-10.parquet
├── models/
│   ├── production/
│   │   ├── lstm_BTC_USD_1m_v2.0.pt
│   │   ├── lstm_ETH_USD_1m_v2.0.pt
│   │   └── lstm_SOL_USD_1m_v2.0.pt
│   └── experiments/
│       └── (experimental models...)
└── sentiment/
    ├── reddit_sentiment_2018-2025.parquet
    ├── fear_greed_index_2018-2025.parquet
    └── google_trends_2018-2025.parquet
```

**Costs**:
```
Storage (S3 Standard):
- 3GB data × $0.023/GB/month = $0.07/month
- 100GB over time × $0.023/GB/month = $2.30/month

Requests:
- 1,000 PUT requests = $0.005
- 10,000 GET requests = $0.004

Data Transfer:
- Download 3GB to GPU instance = $0.27 (one-time per training)
- Upload models (20MB) = $0 (uploads are free)

Total Monthly: ~$2.50/month
```

#### Option B: S3 + EFS (For frequent access)

**Use case**: If training multiple times per day

```
Architecture:
┌─────────────┐       ┌──────────────┐
│  S3 Bucket  │◄─────►│   EFS Volume │
│  (Archive)  │       │  (Fast cache)│
└─────────────┘       └───────┬──────┘
                              │
                      ┌───────▼───────┐
                      │  GPU Instance │
                      │   (Training)  │
                      └───────────────┘
```

**Costs**:
- EFS: 3GB × $0.30/GB/month = $0.90/month
- S3: $0.07/month (archive)
- **Total**: ~$1/month

**When to use**: If training >5 times per day (faster than downloading from S3 each time)

#### Option C: Local + S3 Backup

**Use case**: Budget-conscious, train locally

```
Architecture:
┌──────────────┐
│ Local Storage│ ← Primary (fast, free)
└──────┬───────┘
       │
   ┌───▼────┐
   │   S3   │ ← Backup only (versioned)
   └────────┘
```

**Costs**:
- Local: $0/month (use existing disk)
- S3: $2.50/month (backup + versioning)
- **Total**: $2.50/month

**Pros**: Cheapest, fast local access
**Cons**: Still vulnerable to local disk failure until backup runs

---

## Cost Comparison: Training Strategies

### Scenario: Train once with 7 years of data

| Strategy | Instance | Time | GPU Cost | Storage | Total |
|----------|----------|------|----------|---------|-------|
| **Current (CPU)** | Local | 9 days | $0 | $0 | **$0** |
| **Budget GPU** | g4dn.xlarge | 90 min | $0.79 | $0 | **$0.79** |
| **Recommended** | p3.2xlarge | 30 min | $1.53 | $0 | **$1.53** ⭐ |
| **Multi-GPU** | p3.8xlarge | 3 min | $0.61 | $0 | **$0.61** 🔥 |
| **With S3** | p3.8xlarge | 3 min | $0.61 | $2.50/mo | **$3.11** |

### Scenario: Train weekly for 1 month (4 trainings)

| Strategy | Instance | Time/Week | Monthly GPU Cost | Storage | Total |
|----------|----------|-----------|------------------|---------|-------|
| **Budget GPU** | g4dn.xlarge | 90 min | $3.16 | $2.50 | **$5.66** |
| **Recommended** | p3.2xlarge | 30 min | $6.12 | $2.50 | **$8.62** ⭐ |
| **Multi-GPU** | p3.8xlarge | 3 min | $2.44 | $2.50 | **$4.94** 🔥 |
| **Reserved** | g4dn.xlarge | 90 min | $1.80 | $2.50 | **$4.30** |

### Scenario: Train daily for 1 month (30 trainings)

| Strategy | Instance | Time/Day | Monthly GPU Cost | Storage | Total |
|----------|----------|----------|------------------|---------|-------|
| **Budget GPU** | g4dn.xlarge | 90 min | $23.67 | $2.50 | **$26.17** |
| **Recommended** | p3.2xlarge | 30 min | $45.90 | $2.50 | **$48.40** |
| **Multi-GPU** | p3.8xlarge | 3 min | $18.36 | $2.50 | **$20.86** ⭐ |
| **Reserved T4** | g4dn.xlarge | 90 min | $13.50 | $2.50 | **$16.00** |
| **24/7 Reserved** | g4dn.xlarge | On-demand | $216 | $2.50 | **$218.50** |

---

## Recommended Setup

### Phase 1: Initial Training (Now)
**Goal**: Train with 7 years of data + sentiment features

**Setup**:
1. **Storage**: S3 for backup ($2.50/month)
2. **Training**: p3.8xlarge (4x V100) for multi-GPU parallel training
3. **Cost**: $0.61 (training) + $2.50 (storage) = **$3.11 one-time**
4. **Time**: 5 minutes total (setup + training)

**Why multi-GPU**:
- Train all 3 models in parallel (3 min vs 30 min sequential)
- Cheapest per-training cost ($0.61 vs $1.53)
- Can experiment with hyperparameters quickly
- Run 10 training experiments for ~$6

### Phase 2: Regular Retraining (Monthly)
**Goal**: Retrain monthly as new data arrives

**Setup**:
1. **Storage**: S3 + incremental backups ($2.50/month)
2. **Training**: p3.8xlarge for monthly retrain ($0.61/month)
3. **Cost**: $2.50 + $0.61 = **$3.11/month**

### Phase 3: Active Development (Daily experiments)
**Goal**: Daily hyperparameter tuning and model improvements

**Setup**:
1. **Storage**: S3 + EFS cache ($3.50/month)
2. **Training**: p3.8xlarge for rapid experiments ($18/month for 30 runs)
3. **Cost**: $3.50 + $18 = **$21.50/month**

### Phase 4: Production (Continuous)
**Goal**: Live trading with daily retraining

**Setup**:
1. **Storage**: S3 + EFS ($3.50/month)
2. **Training**: Reserved g4dn.xlarge ($16/month for daily training)
3. **Kafka Infrastructure**: t3.medium EC2 ($30/month)
4. **Cost**: $3.50 + $16 + $30 = **$49.50/month**

---

## Ultimate Recommendation 🎯

### For NOW (Initial Training):

**Use p3.8xlarge (4x V100) with S3 storage**

**Why**:
- ✅ **Fastest overall**: 3 minutes for all 3 models
- ✅ **Cheapest per training**: $0.61 (vs $1.53 sequential)
- ✅ **Best for experimentation**: Can run 10+ experiments in <$7
- ✅ **Professional setup**: S3 versioning, backups, collaboration-ready
- ✅ **Total cost**: ~$3.11 (one-time setup + first training)

**Steps**:
1. Setup S3 bucket (5 min)
2. Upload 7 years of data to S3 (10 min)
3. Launch p3.8xlarge instance (2 min)
4. Train all 3 models in parallel (3 min)
5. Download models, terminate instance (2 min)

**Total time**: 22 minutes
**Total cost**: $3.11

### For LATER (After profitable):

**Upgrade to reserved instances + managed services**
- Reserved g4dn.xlarge: $16/month for daily retraining
- SageMaker: $50/month for managed training + deployment
- Total: ~$70/month for production-grade setup

---

## GPU Setup Scripts

### Quick Launch: p3.8xlarge Multi-GPU

```bash
#!/bin/bash
# launch_multi_gpu_training.sh

# 1. Create S3 bucket
aws s3 mb s3://crpbot-data
aws s3api put-bucket-versioning \
  --bucket crpbot-data \
  --versioning-configuration Status=Enabled

# 2. Upload data to S3
aws s3 sync data/ s3://crpbot-data/ --exclude "*.log"

# 3. Launch p3.8xlarge instance
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type p3.8xlarge \
  --key-name your-key \
  --security-group-ids sg-xxx \
  --subnet-id subnet-xxx \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100}}]' \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "Launched instance: $INSTANCE_ID"

# 4. Wait for instance to be running
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# 5. Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo "Instance ready at: $PUBLIC_IP"

# 6. SSH and setup
ssh -i your-key.pem ubuntu@$PUBLIC_IP << 'EOF'
# Install dependencies
git clone https://github.com/your-repo/crpbot.git
cd crpbot
pip install -r requirements.txt

# Download data from S3
aws s3 sync s3://crpbot-data/features/ data/features/

# Train all models in parallel (4 GPUs)
python apps/trainer/train_multi_gpu.py \
  --symbols BTC-USD ETH-USD SOL-USD \
  --epochs 15 \
  --devices cuda:0 cuda:1 cuda:2

# Upload trained models
aws s3 sync models/ s3://crpbot-data/models/

# Cleanup
exit
EOF

# 7. Terminate instance
aws ec2 terminate-instances --instance-ids $INSTANCE_ID

echo "Training complete! Models saved to S3."
```

---

## Summary

### Your Questions Answered:

**Q: Can we spend more on GPU to get better and faster result?**

**A**: Yes! Here's the hierarchy:

1. **Budget** ($0.75): g5.2xlarge - 12 min per model, 36 min total
2. **Recommended** ($1.53): p3.2xlarge - 10 min per model, 30 min total ⭐
3. **Best Value** ($0.61): p3.8xlarge - 3 min for ALL 3 models 🔥
4. **Overkill** ($1.64): p4d.24xlarge (8x A100) - 2 min total

**Winner**: **p3.8xlarge at $0.61** - train all 3 models in parallel in 3 minutes!

**Q: Where are we storing data?**

**A**: Currently all local. Recommended:
- **S3** ($2.50/month) for versioned storage
- **Local** for fast access during development
- **EFS** ($0.90/month) if training frequently

**Total Recommended Setup**:
- Storage: S3 at $2.50/month
- Training: p3.8xlarge at $0.61 per training session
- **Total**: $3.11 for initial setup + training

**Want me to proceed with this setup?**
