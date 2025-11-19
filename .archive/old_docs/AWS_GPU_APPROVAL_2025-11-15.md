# 🎉 AWS GPU Approval - Training Infrastructure Ready

**Created**: 2025-11-15 18:45 EST (Toronto)
**Last Updated**: 2025-11-15 18:45 EST (Toronto)
**Author**: Builder Claude
**Status**: APPROVED - Ready to Use
**Purpose**: Document AWS GPU approval and update V5 training strategy

---

## 📧 Approval Details

**From**: Amazon Web Services
**Date**: 2025-11-14 05:38 EST
**Status**: ✅ **APPROVED**

**Approval Message**:
```
Thank you for your patience.

I am happy to confirm that your limit increase request for EC2
Instances/All G and VT instances in the US East (Northern Virginia)
region has been approved.

Your new quota is 4, and will be available in the US East (Northern
Virginia) region within the next hour.
```

---

## 📊 What Was Approved

**Instance Types**: EC2 G and VT instances
**Region**: us-east-1 (Northern Virginia)
**Quota**: 4 instances (concurrent)
**Availability**: Immediate (approved 2025-11-14)

**Specific Instance Families**:
- ✅ **G instances**: GPU compute (g4dn, g5, g6)
- ✅ **VT instances**: Video transcoding (not needed for ML)

---

## 🎯 Impact on V5 Strategy

### Before GPU Approval:
```
Training Options:
1. Google Colab (free, limited, unreliable)
2. AWS CPU (slow, 60-90 min per model)

Limitations:
- Colab session limits (90 min idle, 12 hr max)
- Data upload/download overhead (650 MB)
- No persistent environment
```

### After GPU Approval:
```
Training Options:
1. AWS GPU (fast, reliable, professional) ⭐ PRIMARY
2. Google Colab (backup option)
3. AWS CPU (not recommended)

Advantages:
- Full control, no session limits
- Data stays on AWS (S3 + EBS)
- Professional setup, scalable
- Cost-effective (~$0.50/training run)
```

---

## 💰 Cost Analysis

### Recommended: g4dn.xlarge

**Specifications**:
- GPU: NVIDIA T4 (16GB VRAM)
- vCPU: 4 cores
- RAM: 16 GB
- Storage: Up to 125 GB NVMe SSD
- Network: Up to 25 Gbps

**Pricing**:
- On-Demand: **$0.526/hour**
- Spot: **~$0.158/hour** (70% savings)

**Training Time Estimates**:
| Task | Time (GPU) | Cost (On-Demand) | Cost (Spot) |
|------|-----------|------------------|-------------|
| 1 LSTM model (15 epochs) | ~10-15 min | $0.13-0.20 | $0.04-0.06 |
| 3 LSTM models | ~30-45 min | $0.26-0.39 | $0.08-0.12 |
| 1 Transformer (15 epochs) | ~15-20 min | $0.13-0.18 | $0.04-0.05 |
| **Full V5 training** | **~1 hour** | **$0.53** | **$0.16** |

**Monthly Cost** (Phase 1 - 4 weeks):
- Training iterations: ~10-15 runs
- Total GPU time: ~10-15 hours/month
- **On-Demand**: ~$5-8/month
- **Spot**: ~$2-3/month

---

### Alternative: g5.xlarge (More Powerful)

**Specifications**:
- GPU: NVIDIA A10G (24GB VRAM)
- vCPU: 4 cores
- RAM: 16 GB
- Storage: Up to 250 GB NVMe SSD

**Pricing**:
- On-Demand: **$1.006/hour**
- Spot: **~$0.30/hour**

**When to Use**:
- Larger models (future phases)
- Faster training needed
- More VRAM required

---

## 📋 Updated V5 Phase 1 Budget

### With AWS GPU (g4dn.xlarge on-demand):

```
Tardis Historical:  $98/month    (Coinbase + Kraken, Canada-compliant)
AWS EC2 (GPU):      $5-8/month   (g4dn.xlarge, ~10-15 hours)
AWS S3 (storage):   $10/month    (~100 GB Tardis data)
AWS RDS (database): $20/month    (PostgreSQL for signals)
──────────────────────────────────────────────────
Total:              $133-136/month ✅ UNDER $150!
```

**Compared to Original V5 Budget**:
- Original: $148/month (CPU-based estimate)
- With GPU: $133-136/month
- **Savings**: $12-15/month + faster training!

---

## 🚀 AWS GPU Setup Plan

### Phase 1: Instance Setup (15 minutes)

**1. Launch GPU Instance**:
```bash
# Using AWS Deep Learning AMI (PyTorch pre-installed)
aws ec2 run-instances \
  --image-id ami-0c94855ba95c574c8 \
  --instance-type g4dn.xlarge \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxx \
  --subnet-id subnet-xxxxx \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=crpbot-gpu-training}]'
```

**2. Configure Security Group**:
- SSH (22): Your IP only
- Outbound: All traffic (for package downloads)

**3. Attach Storage**:
- Root EBS: 50 GB (for OS + dependencies)
- Data EBS: 100 GB (for Tardis data + models)

---

### Phase 2: Software Setup (20 minutes)

**1. Install Dependencies**:
```bash
# SSH to instance
ssh -i your-key.pem ubuntu@<instance-ip>

# Clone repo
git clone https://github.com/imnuman/crpbot.git
cd crpbot

# Install uv and dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install -e .
uv pip install -e ".[dev]"
```

**2. Configure AWS Access**:
```bash
# Copy credentials (or use IAM role)
aws configure

# Verify S3 access
aws s3 ls s3://crpbot-ml-data-20251110/
```

**3. Download Data from S3**:
```bash
# Download feature files
aws s3 cp s3://crpbot-ml-data-20251110/features/ data/features/ --recursive

# Or mount S3 as filesystem (optional)
# Using s3fs-fuse
```

---

### Phase 3: Training Execution (5 minutes)

**1. Verify GPU**:
```bash
# Check NVIDIA GPU
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

**Expected Output**:
```
True
Tesla T4
```

**2. Run Training**:
```bash
# Train all 3 LSTM models
uv run python apps/trainer/main.py --task lstm --coin BTC --epochs 15
uv run python apps/trainer/main.py --task lstm --coin ETH --epochs 15
uv run python apps/trainer/main.py --task lstm --coin SOL --epochs 15

# Train Transformer
uv run python apps/trainer/main.py --task transformer --epochs 15
```

**3. Save Models to S3**:
```bash
# Upload trained models
aws s3 cp models/v5/ s3://crpbot-ml-data-20251110/models/v5/ --recursive
```

**4. Terminate Instance**:
```bash
# Exit SSH
exit

# Terminate instance (or stop to save EBS)
aws ec2 terminate-instances --instance-ids i-xxxxx
```

---

## 💡 Cost Optimization Tips

### 1. Use Spot Instances (70% savings)
```bash
# Launch spot instance instead
aws ec2 run-instances \
  --instance-type g4dn.xlarge \
  --instance-market-options '{"MarketType":"spot"}' \
  ...
```

**Pros**: 70% cheaper ($0.158/hour vs $0.526/hour)
**Cons**: Can be interrupted (rare for short training runs)
**Recommendation**: Use for Phase 1 training

---

### 2. Stop (Don't Terminate) Between Runs
```bash
# Stop instance (keeps EBS, no compute charges)
aws ec2 stop-instances --instance-ids i-xxxxx

# Restart when needed
aws ec2 start-instances --instance-ids i-xxxxx
```

**Savings**: Only pay for EBS storage (~$5/month) between training runs

---

### 3. Use EBS Snapshots
```bash
# Create snapshot of configured instance
aws ec2 create-snapshot --volume-id vol-xxxxx --description "crpbot-gpu-setup"

# Launch new instance from snapshot (skip software setup)
```

---

### 4. Automate with Lambda (Future)
```python
# Lambda function to:
# 1. Launch spot instance
# 2. Run training
# 3. Upload results to S3
# 4. Terminate instance
# Total cost: <$0.20 per training run
```

---

## 📊 Comparison: AWS GPU vs Google Colab

| Aspect | AWS GPU (g4dn.xlarge) | Google Colab |
|--------|----------------------|--------------|
| **Cost** | $0.53/run (on-demand) | Free (with limits) |
| **Speed** | ~1 hour per full training | ~57 min per full training |
| **Reliability** | ✅ 100% available | ⚠️ Subject to limits |
| **Session Limits** | ✅ None | ❌ 90 min idle, 12 hr max |
| **Data Transfer** | ✅ Local (S3) | ❌ Upload/download 650 MB |
| **Persistence** | ✅ Full control | ❌ Session-based |
| **GPU** | NVIDIA T4 (16GB) | NVIDIA T4 (16GB) |
| **RAM** | 16 GB | 12.7 GB |
| **Storage** | Up to 125 GB NVMe | ~100 GB (session) |
| **Scalability** | ✅ 4 concurrent instances | ❌ 1 session |
| **Professional** | ✅ Yes | ❌ Consumer tier |

**Recommendation**:
- **Primary**: AWS GPU (g4dn.xlarge spot)
- **Backup**: Google Colab (if AWS unavailable)

---

## 🎯 Updated V5 Training Strategy

### Week 3: Model Training (2025-12-02 to 2025-12-08)

**Primary Method: AWS GPU** ⭐

**Day 1-2: Setup**:
1. Launch g4dn.xlarge spot instance
2. Install dependencies and configure
3. Download Tardis feature data from S3
4. Verify GPU and PyTorch CUDA

**Day 3-5: Training**:
1. Train 3 LSTM models (BTC, ETH, SOL)
   - 15-20 epochs each
   - ~30-45 min total on GPU
2. Train Transformer (multi-coin)
   - 15-20 epochs
   - ~15-20 min on GPU
3. Save checkpoints to S3

**Day 6-7: Iteration** (if needed):
1. Adjust hyperparameters
2. Re-train underperforming models
3. Total GPU time: ~2-3 hours

**Cost for Week 3**:
- Spot instance: ~3 hours × $0.158/hour = **$0.47**
- EBS storage: ~$5 for week
- **Total: <$6 for entire training week**

---

## 📝 Next Steps

### Immediate (Builder Claude):
1. ✅ Create AWS GPU launch script
2. ✅ Prepare training automation
3. ✅ Test GPU instance launch
4. ⏸️ Wait for Tardis subscription

### Week 1 (After Tardis Subscription):
1. Download Tardis data to S3
2. Launch GPU instance
3. Test training pipeline
4. Optimize batch sizes for T4

### Week 3 (Training Week):
1. Launch spot instance
2. Train all V5 models
3. Upload to S3
4. Terminate instance

---

## 🔧 Technical Details

### Instance Specifications (g4dn.xlarge)

**GPU**:
- Model: NVIDIA Tesla T4
- CUDA Cores: 2,560
- Tensor Cores: 320
- VRAM: 16 GB GDDR6
- Memory Bandwidth: 320 GB/s
- FP32 Performance: 8.1 TFLOPS
- TF32 Performance: 65 TFLOPS

**Ideal For**:
- LSTM training (small to medium models)
- Transformer training (moderate size)
- Batch inference
- Mixed precision training

**Not Ideal For**:
- Very large transformers (>100M params)
- Extremely large batch sizes (>128)
- Models requiring >16GB VRAM

**Our Models**:
- LSTM: ~1M params ✅ Perfect fit
- Transformer: ~5M params ✅ Perfect fit
- Batch size: 32 (LSTM), 16 (Transformer) ✅ Fits easily

---

## ✅ Summary

**Status**: AWS GPU approved and ready to use

**Impact**:
- ✅ Professional GPU training environment
- ✅ Cost-effective (~$0.50 per training run)
- ✅ Faster and more reliable than Colab
- ✅ Fully integrated with existing AWS infrastructure

**Updated V5 Budget**:
- Phase 1: $133-136/month (down from $148)
- Training cost: ~$5-8/month (minimal)
- Still well under $200/month budget ✅

**Next Action**:
- Subscribe to Tardis.dev ($98/month)
- Start V5 Week 1 (data download)
- Prepare AWS GPU for Week 3 training

---

**File**: `AWS_GPU_APPROVAL_2025-11-15.md`
**Status**: Complete and ready for V5 integration
**Created**: 2025-11-15 18:45 EST (Toronto)
**Next Update**: After first GPU training run
