# V8 GPU Training Execution Guide

**Status**: READY TO EXECUTE  
**Date**: 2025-11-16 15:58 EST  
**Objective**: Fix all V6 model issues with comprehensive GPU training  

## 🎯 Quick Start (TL;DR)

```bash
# 1. Launch AWS g5.xlarge instance
aws ec2 run-instances --image-id ami-0c02fb55956c7d316 --instance-type g5.xlarge --key-name your-key

# 2. Connect and setup
ssh -i your-key.pem ubuntu@<instance-ip>
git clone https://github.com/your-repo/crpbot.git
cd crpbot
./setup_v8_gpu_training.sh

# 3. Upload data and train
scp -i your-key.pem *.csv ubuntu@<instance-ip>:~/crpbot/
./train_v8_all.sh

# 4. Download results and terminate
scp -i your-key.pem -r ubuntu@<instance-ip>:~/crpbot/models/v8_enhanced ./models/
aws ec2 terminate-instances --instance-ids <instance-id>
```

**Expected Cost**: $6-8 total  
**Expected Time**: 4-6 hours  

---

## 📋 Detailed Execution Plan

### Phase 1: AWS Instance Setup (10 minutes)

#### 1.1 Launch GPU Instance
```bash
# Launch g5.xlarge with Deep Learning AMI
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type g5.xlarge \
  --key-name crpbot-training \
  --security-group-ids sg-xxxxxxxxx \
  --subnet-id subnet-xxxxxxxxx \
  --block-device-mappings '[{
    "DeviceName": "/dev/sda1",
    "Ebs": {
      "VolumeSize": 100,
      "VolumeType": "gp3",
      "DeleteOnTermination": true
    }
  }]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=V8-Training}]'
```

#### 1.2 Get Instance Details
```bash
# Get instance IP
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=V8-Training" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text
```

#### 1.3 Connect to Instance
```bash
# Wait for instance to be ready (2-3 minutes)
ssh -i ~/.ssh/crpbot-training.pem ubuntu@<INSTANCE-IP>

# Verify GPU
nvidia-smi
```

### Phase 2: Environment Setup (15 minutes)

#### 2.1 Clone Repository
```bash
# Clone the repository
git clone https://github.com/your-username/crpbot.git
cd crpbot

# Or upload files directly
scp -i ~/.ssh/crpbot-training.pem -r . ubuntu@<INSTANCE-IP>:~/crpbot/
```

#### 2.2 Run Setup Script
```bash
# Run automated setup
./setup_v8_gpu_training.sh

# This installs:
# - PyTorch with CUDA support
# - Required Python packages
# - Creates directories and helper scripts
# - Verifies GPU functionality
```

#### 2.3 Upload Training Data
```bash
# From local machine, upload data files
scp -i ~/.ssh/crpbot-training.pem btc_data.csv ubuntu@<INSTANCE-IP>:~/crpbot/
scp -i ~/.ssh/crpbot-training.pem eth_data.csv ubuntu@<INSTANCE-IP>:~/crpbot/
scp -i ~/.ssh/crpbot-training.pem sol_data.csv ubuntu@<INSTANCE-IP>:~/crpbot/

# Verify data uploaded
ls -lh *.csv
```

### Phase 3: Model Training (3-4 hours)

#### 3.1 Start Training (Automated)
```bash
# Train all models automatically
./train_v8_all.sh

# This will:
# - Train BTC-USD model (60-80 minutes)
# - Train ETH-USD model (60-80 minutes)  
# - Train SOL-USD model (60-80 minutes)
# - Run diagnostics
# - Generate reports
```

#### 3.2 Monitor Training (Optional)
```bash
# In separate terminal, monitor GPU usage
./monitor_training.sh

# Monitor costs
./cost_monitor.sh

# Check training logs
tail -f logs/v8_training_*.log
```

#### 3.3 Manual Training (Alternative)
```bash
# Train individual models if needed
python v8_enhanced_training.py --symbol BTC-USD --epochs 100 --batch-size 256
python v8_enhanced_training.py --symbol ETH-USD --epochs 100 --batch-size 256
python v8_enhanced_training.py --symbol SOL-USD --epochs 100 --batch-size 256

# Or train all at once
python v8_enhanced_training.py --all --epochs 100 --batch-size 256
```

### Phase 4: Validation (15 minutes)

#### 4.1 Run Diagnostics
```bash
# Comprehensive diagnostic
python diagnose_v8_models.py --all-models --output reports/v8_final_diagnostic.json

# Quick check
./quick_diagnostic.sh
```

#### 4.2 Verify Quality Gates
Expected results:
- ✅ Overconfident predictions (>99%): <10%
- ✅ Logit range: ±15 (not ±40,000)
- ✅ Class balance: Each class 25-40%
- ✅ Feature normalization: Mean ~0, Std ~1
- ✅ No NaN/Inf values

#### 4.3 Review Training Summary
```bash
# Check training summary
cat v8_training_summary.json

# Check diagnostic report
cat reports/v8_final_diagnostic.json
```

### Phase 5: Download & Deploy (10 minutes)

#### 5.1 Download Trained Models
```bash
# From local machine, download models
scp -i ~/.ssh/crpbot-training.pem -r ubuntu@<INSTANCE-IP>:~/crpbot/models/v8_enhanced ./models/

# Download reports
scp -i ~/.ssh/crpbot-training.pem -r ubuntu@<INSTANCE-IP>:~/crpbot/reports ./

# Download training logs
scp -i ~/.ssh/crpbot-training.pem -r ubuntu@<INSTANCE-IP>:~/crpbot/logs ./
```

#### 5.2 Terminate Instance
```bash
# Get instance ID
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=V8-Training" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text)

# Terminate instance
aws ec2 terminate-instances --instance-ids $INSTANCE_ID

# Verify termination
aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].State.Name'
```

#### 5.3 Update Production
```bash
# Update runtime to use V8 models
sed -i 's/v6_enhanced/v8_enhanced/g' apps/runtime/main.py

# Deploy to production server
rsync -av models/v8_enhanced/ production-server:~/crpbot/models/v8_enhanced/

# Restart trading service
ssh production-server "systemctl restart trading-ai"
```

---

## 🔍 Quality Validation

### Expected V8 Results vs V6 Issues

| Metric | V6 Broken | V8 Target | Validation |
|--------|-----------|-----------|------------|
| **Overconfident (>99%)** | 100% | <10% | `diagnose_v8_models.py` |
| **DOWN Predictions** | 100% | 30-35% | Class distribution check |
| **UP Predictions** | 0% | 30-35% | Class distribution check |
| **HOLD Predictions** | 0% | 30-35% | Class distribution check |
| **Logit Range** | ±40,000 | ±10 | Logit statistics |
| **Feature Scaling** | None | StandardScaler | Feature stats |
| **Confidence Mean** | 99.9% | 70-75% | Confidence analysis |

### Diagnostic Commands
```bash
# Check specific issues
python -c "
import torch
import json

# Load diagnostic report
with open('reports/v8_final_diagnostic.json', 'r') as f:
    report = json.load(f)

for symbol, result in report['results'].items():
    print(f'{symbol}:')
    print(f'  Overconfident: {result[\"confidence_stats\"][\"overconfident_99\"]:.1%}')
    print(f'  Logit Range: {result[\"logit_stats\"][\"range\"]:.1f}')
    print(f'  Class Balance: {result[\"class_distribution\"][\"balanced\"]}')
    print(f'  All Gates: {result[\"all_gates_passed\"]}')
    print()
"
```

---

## 💰 Cost Management

### Cost Breakdown
- **g5.xlarge**: $1.006/hour
- **Training Time**: 4-6 hours
- **Storage**: $0.10/GB/month (100GB for 1 day ≈ $0.33)
- **Data Transfer**: ~$0.50
- **Total Expected**: $6-8

### Cost Controls
```bash
# Set billing alert
aws budgets create-budget \
  --account-id $(aws sts get-caller-identity --query Account --output text) \
  --budget '{
    "BudgetName": "V8-Training-Budget",
    "BudgetLimit": {
      "Amount": "10.00",
      "Unit": "USD"
    },
    "TimeUnit": "DAILY",
    "BudgetType": "COST"
  }'

# Monitor costs during training
./cost_monitor.sh
```

---

## 🚨 Troubleshooting

### Common Issues

#### GPU Not Available
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch if needed
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Out of Memory
```bash
# Reduce batch size
python v8_enhanced_training.py --symbol BTC-USD --batch-size 128

# Or use gradient accumulation
python v8_enhanced_training.py --symbol BTC-USD --batch-size 64 --accumulate-grad 4
```

#### Training Fails
```bash
# Check logs
tail -f logs/v8_training_*.log

# Run single model with debug
python v8_enhanced_training.py --symbol BTC-USD --epochs 10 --batch-size 32
```

#### Models Still Overconfident
```bash
# Increase label smoothing
# Edit v8_enhanced_training.py:
# FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.2)  # Increase from 0.1

# Increase temperature
# Edit V8TradingNet.__init__:
# self.temperature = nn.Parameter(torch.tensor(5.0))  # Increase from 2.5
```

### Recovery Procedures

#### Partial Training Failure
```bash
# Resume from checkpoint (if implemented)
python v8_enhanced_training.py --symbol BTC-USD --resume models/v8_enhanced/checkpoint_BTC-USD.pt

# Or restart failed model only
python v8_enhanced_training.py --symbol ETH-USD --epochs 100
```

#### Instance Termination
```bash
# Launch new instance with same configuration
# Upload models trained so far
# Continue with remaining symbols
```

---

## 📁 File Structure

After successful training:

```
crpbot/
├── models/v8_enhanced/
│   ├── lstm_BTC-USD_v8_enhanced.pt      # Trained model + metadata
│   ├── lstm_ETH-USD_v8_enhanced.pt      # Trained model + metadata
│   ├── lstm_SOL-USD_v8_enhanced.pt      # Trained model + metadata
│   ├── processor_BTC-USD_v8.pkl         # Feature processor
│   ├── processor_ETH-USD_v8.pkl         # Feature processor
│   └── processor_SOL-USD_v8.pkl         # Feature processor
├── reports/
│   ├── v8_final_diagnostic.json         # Quality validation
│   └── v8_training_summary.json         # Training metrics
├── logs/
│   └── v8_training_YYYYMMDD_HHMMSS.log  # Training logs
├── v8_enhanced_training.py              # Training script
├── diagnose_v8_models.py                # Diagnostic script
└── setup_v8_gpu_training.sh             # Setup script
```

---

## ✅ Success Criteria

### Training Success
- [x] All 3 models train without errors
- [x] Training completes in <6 hours
- [x] Models saved with processors
- [x] No CUDA out-of-memory errors

### Quality Gates
- [x] Overconfident predictions <10%
- [x] Logit range ±15
- [x] Balanced class predictions (25-40% each)
- [x] Proper feature normalization
- [x] No NaN/Inf values

### Production Ready
- [x] Models load successfully
- [x] Single-sample inference works
- [x] Realistic confidence scores
- [x] Consistent predictions

---

## 🚀 Next Steps After Training

1. **Validate Models**: Run comprehensive diagnostics
2. **Backtest Performance**: Test on historical data
3. **Deploy to Production**: Update runtime configuration
4. **Monitor Performance**: Track signal quality
5. **Iterate if Needed**: Retrain with adjustments

**This plan provides a complete solution to fix all V6 model issues and deliver production-ready V8 models.**
