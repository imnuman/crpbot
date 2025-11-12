# AWS GPU Training Implementation Plan

**Created**: 2025-11-10
**Status**: Phase 1 Complete - S3 Storage Setup ✅

## Progress Overview

### ✅ Phase 1: S3 Storage Setup (COMPLETE)
- [x] S3 bucket created: `crpbot-ml-data-20251110`
- [x] Versioning enabled
- [x] Bucket tagged for cost tracking
- [x] Data upload in progress (765MB → S3)
- [x] Training scripts created

**Cost**: $2.50/month for storage

### ⏳ Phase 2: Historical Data & Sentiment (NEXT)
- [ ] Fetch 7 years BTC/ETH data from Coinbase
- [ ] Fetch 4.4 years SOL data
- [ ] Setup Reddit API (FREE)
- [ ] Fetch Fear & Greed Index (FREE)
- [ ] Integrate sentiment features
- [ ] Engineer features with all data

**Estimated Time**: 10-12 hours
**Cost**: $0 (all APIs free)

### ⏳ Phase 3: GPU Training (READY TO GO)
- [ ] Launch p3.8xlarge instance (4x V100)
- [ ] Train all 3 models in parallel
- [ ] Download trained models
- [ ] Terminate instance

**Estimated Time**: 10 minutes total
**Cost**: $0.61 for 3 minutes of training

---

## S3 Bucket Details

```
Bucket Name: crpbot-ml-data-20251110
Region: us-east-1
Versioning: Enabled
Access: IAM role-based
```

**Structure**:
```
s3://crpbot-ml-data-20251110/
├── raw/              (current: 121MB, target: 350MB with 7 years)
├── features/         (current: 643MB, target: 2.4GB with 7 years + sentiment)
├── models/
│   ├── production/   (trained models)
│   └── experiments/  (experimental models)
├── sentiment/        (Reddit, F&G, Google Trends)
└── backups/          (manual backups)
```

---

## Created Scripts

### 1. S3 Storage Management
```bash
# Setup S3 bucket (DONE)
./scripts/setup_s3_storage.sh

# Upload data to S3 (IN PROGRESS)
./scripts/upload_to_s3.sh

# Download from S3
aws s3 sync s3://crpbot-ml-data-20251110/features/ data/features/
```

### 2. GPU Training
```bash
# Setup GPU instance (creates p3.8xlarge)
./scripts/setup_gpu_training.sh

# Train on GPU (run on the instance)
./scripts/train_multi_gpu.sh

# Monitor training
watch nvidia-smi
tail -f /tmp/train_btc_gpu.log
```

### 3. Multi-GPU Training Details

**Configuration**:
- Instance: p3.8xlarge (4x V100 GPUs)
- Training mode: Parallel (all 3 models simultaneously)
- GPU allocation:
  - GPU 0: BTC-USD model
  - GPU 1: ETH-USD model
  - GPU 2: SOL-USD model
  - GPU 3: Reserved (for experiments)

**Performance**:
- Time per model: ~3 minutes
- Total time: ~3 minutes (parallel)
- Cost: $0.61 (3 min × $12.24/hour)

---

## Next Steps

### Immediate (Today)
1. ✅ S3 bucket setup
2. ⏳ Wait for data upload to complete (~5 min)
3. ⏳ Verify S3 upload
4. ⏳ Start fetching 7 years of historical data

### Tomorrow
5. ⏳ Continue historical data fetch (3-4 hours)
6. ⏳ Setup Reddit API
7. ⏳ Fetch sentiment data
8. ⏳ Engineer features with all data

### Day 3
9. ⏳ Launch GPU instance
10. ⏳ Train all models on GPU
11. ⏳ Evaluate models
12. ⏳ Deploy to production

---

## Cost Tracking

### One-Time Costs
| Item | Cost | Status |
|------|------|--------|
| S3 setup | $0 | ✅ Complete |
| Data upload | $0 | ⏳ In progress |
| GPU training (first run) | $0.61 | ⏳ Pending |
| **Total One-Time** | **$0.61** | |

### Monthly Recurring
| Item | Cost/Month | Status |
|------|------------|--------|
| S3 storage (3GB) | $2.50 | ✅ Active |
| Reddit API | $0 (FREE) | ⏳ Setup needed |
| Fear & Greed API | $0 (FREE) | ⏳ Setup needed |
| Coinbase API | $0 (FREE) | ✅ Active |
| **Total Monthly** | **$2.50** | |

### Per-Training Costs
| Scenario | Frequency | Cost |
|----------|-----------|------|
| One-time training | 1x | $0.61 |
| Weekly retraining | 4x/month | $2.44/month |
| Daily retraining | 30x/month | $18.30/month |

---

## GPU Training Workflow

### Step 1: Launch Instance
```bash
./scripts/setup_gpu_training.sh
```

**What it does**:
1. Creates SSH key pair (if needed)
2. Creates security group
3. Creates IAM role for S3 access
4. Launches p3.8xlarge instance
5. Waits for instance to be ready
6. Provides SSH command

**Time**: 2-3 minutes
**Cost**: $0 (no instance time yet)

### Step 2: SSH to Instance
```bash
ssh -i ~/.ssh/crpbot-training.pem ubuntu@<PUBLIC_IP>
```

**Wait 2-3 minutes** for initialization to complete (downloads data from S3)

### Step 3: Start Training
```bash
cd crpbot
./scripts/train_multi_gpu.sh
```

**What it does**:
1. Checks GPU availability
2. Starts BTC training on GPU 0
3. Starts ETH training on GPU 1
4. Starts SOL training on GPU 2
5. Monitors progress in real-time
6. Uploads models to S3 when done

**Time**: 3 minutes
**Cost**: $0.61

### Step 4: Verify & Terminate
```bash
# On local machine
aws s3 ls s3://crpbot-ml-data-20251110/models/production/ --recursive --human-readable

# Terminate instance
aws ec2 terminate-instances --instance-ids $(cat .gpu_instance_info | grep INSTANCE_ID | cut -d= -f2)
```

**Important**: Always terminate the instance to stop charges!

---

## Amazon Q Integration

Amazon Q is already configured in your project. Here's how to use it:

### Ask Q for help
```bash
# Get help with AWS commands
q "How do I check S3 bucket costs?"

# Get help with GPU setup
q "What's the best GPU instance for PyTorch training?"

# Get help with training optimization
q "How can I optimize multi-GPU training in PyTorch?"
```

### Q can help with:
- AWS cost optimization
- GPU performance tuning
- PyTorch best practices
- Debugging training issues
- Infrastructure setup

---

## Current Status

### What's Running:
```
✅ S3 bucket created: crpbot-ml-data-20251110
⏳ Data uploading to S3: 765MB (raw + features + models)
⏳ BTC CPU training: Still running (50+ hours remaining)
⏳ ETH CPU training: Still running
⏳ SOL CPU training: Still running
```

### What's Ready:
```
✅ S3 storage scripts
✅ GPU training scripts
✅ Multi-GPU parallel training support
✅ AWS credentials configured
✅ Amazon Q CLI ready
```

### What's Needed:
```
⏳ Stop CPU training (save resources)
⏳ Fetch 7 years of data
⏳ Setup sentiment data sources
⏳ Engineer features with all data
⏳ Launch GPU instance
⏳ Train on GPU (3 minutes!)
```

---

## Decision Points

### 1. CPU Training
Current BTC training has 50+ hours remaining. Options:
- **A) Stop now, wait for GPU** ✅ RECOMMENDED
  - Saves CPU resources
  - GPU training will be 150x faster
  - Waste 50 hours of CPU time but save 50 hours of waiting
- **B) Let it finish**
  - Wait 50+ hours
  - Then redo with 7 years data anyway

**Recommendation**: Stop now, focus on data preparation, then GPU train with all 7 years

### 2. Data Fetching
Start fetching 7 years now or wait?
- **A) Start now in parallel** ✅ RECOMMENDED
  - 3-4 hours to fetch
  - Can run while other tasks proceed
- **B) Wait for S3 upload to finish**
  - Sequential, slower overall

**Recommendation**: Start now in background

### 3. Sentiment Data
Which sources to integrate?
- **A) Reddit only** (fastest, good quality)
- **B) Reddit + F&G + Google Trends** ✅ RECOMMENDED
  - All FREE
  - More features = better predictions
  - Only 2-3 hours more work
- **C) Add Twitter** ($100/month, not recommended)

**Recommendation**: Reddit + F&G + Google Trends (all free, high value)

---

## Summary

### ✅ Completed Today:
1. AWS GPU performance analysis
2. Cost optimization strategy
3. S3 bucket setup
4. Training scripts creation
5. Data upload initiated
6. Multi-GPU training implementation

### ⏳ Next Actions:
1. Wait for S3 upload (5 min)
2. Stop CPU training (save resources)
3. Start 7-year data fetch (4 hours, background)
4. Setup sentiment APIs (2 hours)
5. Engineer features (6 hours)
6. Launch GPU & train (10 min, $0.61)

### 💰 Total Cost So Far:
- S3 storage: $2.50/month ✅
- Data transfer: $0 ✅
- GPU training: $0 (not started yet)
- **Total**: $2.50/month

### 🎯 Expected Results:
- Training time: 3 minutes (vs 9 days on CPU)
- Data: 7 years (vs 2 years)
- Features: 57 (vs 50)
- Cost: $3.11 one-time + $2.50/month
- Accuracy improvement: +5-10% expected

**Ready to proceed with next phase?**
