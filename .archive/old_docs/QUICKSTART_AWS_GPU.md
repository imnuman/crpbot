# AWS GPU Training Quick Start

## ✅ What's Done

### Phase 1: S3 Storage Setup (COMPLETE)
```
✅ S3 bucket created: crpbot-ml-data-20251110
✅ Versioning enabled
✅ IAM tags added for cost tracking
⏳ Data uploading: 765MB → S3 (260MB/765MB done, ~5 min remaining)
```

### Scripts Created
```
✅ scripts/setup_s3_storage.sh       - S3 bucket setup
✅ scripts/upload_to_s3.sh           - Upload data to S3
✅ scripts/setup_gpu_training.sh     - Launch GPU instance
✅ scripts/train_multi_gpu.sh        - Multi-GPU training
✅ docs/TRAINING_OPTIMIZATION_STRATEGY.md
✅ docs/GPU_PERFORMANCE_ANALYSIS.md
✅ docs/AWS_GPU_IMPLEMENTATION_PLAN.md
```

---

## 🚀 Quick Start: GPU Training

### Step 1: Wait for Upload (5 min)
```bash
# Check upload status
aws s3 ls s3://crpbot-ml-data-20251110 --recursive --human-readable --summarize
```

### Step 2: Launch GPU Instance
```bash
./scripts/setup_gpu_training.sh
```

**What it does**:
- Creates SSH key: `~/.ssh/crpbot-training.pem`
- Creates security group
- Launches p3.8xlarge (4x V100 GPUs)
- Installs dependencies
- Downloads data from S3
- Cost: ~$0.61 for 3 minutes

### Step 3: SSH to Instance
```bash
# Command will be shown by setup script
ssh -i ~/.ssh/crpbot-training.pem ubuntu@<PUBLIC_IP>

# Wait 2-3 minutes for initialization
```

### Step 4: Train All Models
```bash
cd crpbot
./scripts/train_multi_gpu.sh
```

**Training**:
- BTC on GPU 0
- ETH on GPU 1
- SOL on GPU 2
- Time: 3 minutes
- Auto-uploads models to S3

### Step 5: Terminate Instance
```bash
# On local machine
aws ec2 terminate-instances --instance-ids $(cat .gpu_instance_info | grep INSTANCE_ID | cut -d= -f2)
```

**IMPORTANT**: Always terminate to stop charges!

---

## 💰 Cost Summary

### One-Time Setup
```
S3 bucket creation:     $0
Data upload (765MB):    $0
GPU training (3 min):   $0.61
──────────────────────────────
TOTAL ONE-TIME:        $0.61
```

### Monthly Recurring
```
S3 storage (3GB):      $2.50/month
Reddit API:            $0 (FREE)
Fear & Greed API:      $0 (FREE)
Coinbase API:          $0 (FREE)
──────────────────────────────
TOTAL MONTHLY:         $2.50/month
```

---

## 📊 Current Status

### S3 Bucket
```
Bucket: s3://crpbot-ml-data-20251110
Region: us-east-1
Upload: ⏳ In progress (260MB/765MB, ~34%)
```

### Data Stored
```
⏳ raw/ - 121MB (uploading)
⏳ features/ - 643MB (uploading)
⏳ models/ - 1.6MB (uploading)
```

### Training
```
⏳ BTC CPU training: Running (50+ hours remaining)
⏳ ETH CPU training: Running
⏳ SOL CPU training: Running
```

**Recommendation**: Stop CPU training once GPU is ready

---

## 🎯 Next Steps

### Today
1. ✅ S3 setup complete
2. ⏳ Wait for upload (~5 min)
3. ⏳ Verify upload
4. ⏳ Launch GPU & train (10 min, $0.61)

### Tomorrow
5. ⏳ Fetch 7 years historical data
6. ⏳ Setup Reddit sentiment API
7. ⏳ Engineer features with all data
8. ⏳ Retrain on GPU with full dataset

### This Week
9. ⏳ Evaluate models (68% accuracy gate)
10. ⏳ Deploy Kafka pipeline
11. ⏳ Start paper trading

---

## 📝 Important Commands

### Check S3 Upload Progress
```bash
aws s3 ls s3://crpbot-ml-data-20251110 --recursive --human-readable --summarize
```

### Stop CPU Training (Save Resources)
```bash
pkill -f "train.*lstm"
```

### Monitor GPU Training
```bash
# On GPU instance
watch nvidia-smi
tail -f /tmp/train_btc_gpu.log
```

### Download Trained Models
```bash
# On local machine
aws s3 sync s3://crpbot-ml-data-20251110/models/production/ models/
```

---

## 🔥 Pro Tips

1. **Use Spot Instances**: Save 50-70% on GPU costs
2. **Parallel Training**: Train all 3 models simultaneously on multi-GPU
3. **S3 Versioning**: Automatic backup of all data/models
4. **Amazon Q**: Ask Q for AWS optimization tips

---

## ❓ Need Help?

```bash
# AWS costs
q "How much am I spending on S3?"

# GPU optimization
q "How to speed up PyTorch training?"

# Check instance status
aws ec2 describe-instances --filters "Name=tag:Name,Values=crpbot-gpu-training"
```

---

## 📈 Expected Improvements

### With GPU Training
- Speed: 150x faster (3 min vs 9 days)
- Cost: $0.61 per training
- Iterations: 10+ experiments per hour

### With 7 Years Data
- Training data: 3.5x more (2 years → 7 years)
- Accuracy: +5-10% expected

### With Sentiment Features
- Features: +7 (Reddit, F&G, Google Trends)
- Signal quality: +3-5% expected

**Total Expected**:
- Accuracy: 60% → 70-75%
- Time to production: 3 days (vs 2 weeks)
- Total cost: $3.11 one-time + $2.50/month

---

**Ready to train on GPU? Wait for S3 upload to complete (~5 min), then run `./scripts/setup_gpu_training.sh`**
