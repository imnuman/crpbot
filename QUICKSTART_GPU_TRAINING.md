# GPU Training Quick Start - Execute Now

**Timeline**: Train all 3 models in 10 minutes for $0.61
**Current CPU Training**: 50+ hours remaining 😱
**GPU Training**: 3 minutes total ⚡

---

## Why GPU Now, Reddit Later?

✅ **DO NOW: GPU Training with Current 58 Features**
- Features are ready (multi-TF data already engineered)
- Get baseline models trained 150x faster
- Only costs $0.61 for all 3 models
- Can evaluate model performance immediately

✅ **DO LATER: Add Reddit Sentiment & Retrain**
- Sentiment is additive (doesn't block current training)
- Takes 30 min to setup + 1-2 hours to fetch historical data
- Only costs $0.61 to retrain with new features
- Allows incremental improvement

**Bottom Line**: Train now with 58 features → Evaluate → Add sentiment → Retrain ($0.61 each time)

---

## Quick Start: 3 Commands to Train on GPU

### Step 1: Launch GPU & Train (5 min setup + 3 min training)

```bash
# Stop slow CPU training (optional - saves resources)
pkill -f "train.*lstm"

# Launch GPU instance and start training
./scripts/setup_gpu_training.sh
```

**What happens**:
1. Creates p3.8xlarge Spot instance (4x V100 GPUs)
2. Downloads feature data from S3 (765MB, ~1-2 min)
3. Installs PyTorch + CUDA (~1 min)
4. Auto-starts training on SSH connection
5. Trains all 3 models in parallel (~3 min)
6. Uploads models to S3
7. **YOU MUST TERMINATE INSTANCE MANUALLY** (see Step 3)

### Step 2: Monitor Training (optional)

SSH to the instance (command shown by script):
```bash
ssh -i ~/.ssh/crpbot-training.pem ubuntu@<PUBLIC_IP>

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check training logs
tail -f /tmp/train_btc_gpu.log
tail -f /tmp/train_eth_gpu.log
tail -f /tmp/train_sol_gpu.log
```

Expected output:
```
GPU 0: BTC training, epoch 5/15, accuracy 62.5%
GPU 1: ETH training, epoch 5/15, accuracy 61.8%
GPU 2: SOL training, epoch 5/15, accuracy 59.2%
```

### Step 3: Terminate Instance (CRITICAL!)

**After training completes** (~3-5 minutes), terminate the instance:

```bash
# On LOCAL machine (not GPU instance):
INSTANCE_ID=$(cat .gpu_instance_info | grep INSTANCE_ID | cut -d= -f2)
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID"

# Verify termination
aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].State.Name' --output text
# Should show: "shutting-down" or "terminated"
```

**⚠️ CRITICAL**: Forgetting to terminate costs $12.24/hour! Set a phone alarm for 10 minutes.

---

## What You Get

After 10 minutes:
- ✅ 3 trained LSTM models (BTC, ETH, SOL)
- ✅ Models saved to S3 (versioned backup)
- ✅ Training logs with accuracy metrics
- ✅ Total cost: $0.61

Baseline accuracy target: 60%+
Production gate: 68%+ (may need more data/features)

---

## Amazon Q Usage

Ask Amazon Q to execute for you:

```bash
# Execute full GPU training workflow
q "Execute Task 1 from AMAZON_Q_TASK_INSTRUCTIONS.md: GPU Training for all 3 models (BTC, ETH, SOL). Follow all steps including instance termination."

# Check status
q "What's the status of GPU training? Show model accuracy."

# Get cost breakdown
q "How much did GPU training cost? Show breakdown."
```

Amazon Q will:
1. Launch GPU instance
2. Monitor training progress
3. Report accuracy metrics
4. Terminate instance automatically
5. Calculate final cost

---

## Next Steps After GPU Training

### Immediate (Today)
1. ✅ GPU training complete
2. Evaluate model accuracy (target: 60%+)
3. If accuracy < 60%, investigate:
   - Feature quality issues?
   - Need more data (7 years vs 2 years)?
   - Hyperparameter tuning needed?

### This Week
4. Setup Reddit API (30 min, Task 2)
5. Fetch historical Reddit sentiment (1-2 hours)
6. Engineer sentiment features (+4 features)

### Next Week
7. Retrain with sentiment (62 features, $0.61)
8. Compare accuracy improvement
9. If improvement > 3%, promote to production

### When Ready for Scale
10. Deploy Phase 1 infrastructure ($37/month)
11. Integrate RDS for trade tracking
12. Setup observability (Prometheus + Grafana)

---

## Cost Summary

| Activity | Duration | Cost |
|----------|----------|------|
| GPU Training (now) | 3 min | $0.61 |
| Reddit API Setup | 30 min | $0 |
| Retrain with Sentiment | 3 min | $0.61 |
| **Total** | **~1 hour total work** | **$1.22** |

Monthly costs:
- S3 storage: $2.50/month (already running)
- GPU retraining (weekly): $2.44/month
- **Total**: $4.94/month

Compare to:
- Current CPU training: 9 days per full retrain (3 models)
- GPU training: 3 minutes per full retrain
- **Speedup**: 150x faster! 🚀

---

## Troubleshooting

**Issue**: Spot instance request rejected
**Solution**: Script will retry with on-demand ($24.48/hour vs $12.24)

**Issue**: CUDA out of memory error
**Solution**: Reduce batch size in `apps/trainer/config.py` from 64 to 32

**Issue**: Can't SSH to instance
**Solution**: Security group may not allow your IP. Script handles this automatically, but if issues persist:
```bash
MY_IP=$(curl -s ifconfig.me)
aws ec2 authorize-security-group-ingress \
  --group-id $(cat .gpu_instance_info | grep SECURITY_GROUP | cut -d= -f2) \
  --protocol tcp --port 22 --cidr "$MY_IP/32"
```

**Issue**: Models not uploaded to S3
**Solution**: Check S3 permissions, manually upload:
```bash
aws s3 cp models/ s3://crpbot-ml-data-20251110/models/production/ --recursive
```

---

## Ready to Execute?

**Option 1: Manual Execution**
```bash
./scripts/setup_gpu_training.sh
# ... wait for training to complete ...
# ... terminate instance ...
```

**Option 2: Amazon Q Execution** (Recommended)
```bash
q "Execute GPU training task from AMAZON_Q_TASK_INSTRUCTIONS.md. Train all 3 models (BTC, ETH, SOL) and terminate instance when done. Report final accuracy and cost."
```

Amazon Q will handle everything automatically and report results.

---

**Let's get those models trained! 🚀**

Current CPU training: 50+ hours remaining
GPU training: 3 minutes

**The choice is clear.** Execute now!
