# AWS Training Scripts

This directory contains automation scripts for AWS GPU training with automatic termination to prevent cost overruns.

## 🚨 Problem

**Without automation**: Forgotten GPU instances can run for days/weeks, accumulating hundreds of dollars in unexpected AWS charges.

**Example**: A g5.xlarge instance left running for 3 days = ~$72 in charges ($1.006/hour × 72 hours)

## ✅ Solution

Three scripts work together to prevent forgotten instances:

### 1. `auto_terminate_after_training.sh`

**Purpose**: Runs ON the GPU instance to monitor training and self-terminate when complete.

**Usage**:
```bash
# On GPU instance
./auto_terminate_after_training.sh "training_command" [max_hours]

# Example:
./auto_terminate_after_training.sh "\
  uv run python apps/trainer/main.py --task lstm --coin BTC --epochs 15 && \
  uv run python apps/trainer/main.py --task lstm --coin ETH --epochs 15 && \
  uv run python apps/trainer/main.py --task lstm --coin SOL --epochs 15 && \
  aws s3 sync models/ s3://crpbot-ml-data/models/" \
  4
```

**Features**:
- Monitors training process PID
- Timeout protection (default: 4 hours)
- Automatic instance termination when training completes
- Logging to `/tmp/auto_terminate.log`

**How it works**:
1. Starts training command in background
2. Monitors process every 10 seconds
3. When process exits OR timeout reached:
   - Waits 5 seconds for log flush
   - Self-terminates instance via `shutdown -h now`
   - Fallback: Uses AWS CLI if shutdown fails

---

### 2. `launch_training_with_auto_terminate.sh`

**Purpose**: Launch EC2 instance with user data script that includes auto-termination.

**Usage**:
```bash
# Launch spot instance (70% cheaper)
./launch_training_with_auto_terminate.sh --spot --instance-type g5.xlarge --max-hours 4

# Launch on-demand instance
./launch_training_with_auto_terminate.sh --instance-type g5.xlarge --max-hours 4
```

**Parameters**:
- `--spot`: Use spot pricing (default: on-demand)
- `--instance-type TYPE`: Instance type (default: g5.xlarge)
- `--max-hours N`: Maximum training time (default: 4)

**Features**:
- Creates user data script with auto-termination built-in
- Supports both spot and on-demand instances
- Tags instances with `AutoTerminate=true`
- Automatically clones repo, installs dependencies, downloads data from S3
- Runs training with auto-termination wrapper

**Prerequisites** (UPDATE THESE IN THE SCRIPT):
- AMI_ID: Ubuntu 22.04 LTS with GPU drivers
- KEY_NAME: Your SSH key name
- SECURITY_GROUP: Security group ID with SSH access
- SUBNET_ID: Subnet ID (optional)

---

### 3. `check_running_instances.sh`

**Purpose**: Monitor ALL running EC2 instances and alert if any are running too long.

**Usage**:
```bash
# Manual check
./check_running_instances.sh

# Automated monitoring (add to crontab)
0 * * * * cd /root/crpbot && ./scripts/check_running_instances.sh | mail -s "AWS Instance Check" your@email.com
```

**Output Example**:
```
==========================================
AWS EC2 INSTANCES MONITOR
==========================================
Region: us-east-1
Warning threshold: 2h
Critical threshold: 6h
==========================================

RUNNING INSTANCES:

⚠️  WARNING | i-0123456789abcdef | g5.xlarge (SPOT)
       Name: crpbot-training
       Running: 3h 25m
       Est. Cost: $1.02

🔴 CRITICAL | i-fedcba9876543210 | g5.xlarge (ON-DEMAND)
       Name: crpbot-training-old
       Running: 72h 14m
       Est. Cost: $72.43

==========================================
⚠️  2 instance(s) need attention!

To terminate an instance:
  aws ec2 terminate-instances --instance-ids <instance-id>
==========================================
```

**Features**:
- Calculates runtime and estimated costs
- Distinguishes spot vs on-demand pricing
- Thresholds: 2h warning, 6h critical
- Exit code = number of instances needing attention (for automation)
- Provides termination commands

**Automated Monitoring**:
```bash
# Add to crontab for hourly checks
crontab -e

# Add line:
0 * * * * cd /root/crpbot && ./scripts/check_running_instances.sh | mail -s "AWS Instance Check" your@email.com
```

---

## 🏃 Quick Start: Training with Auto-Termination

### Method A: Launch with built-in termination (RECOMMENDED)

```bash
# 1. Update configuration in launch_training_with_auto_terminate.sh
#    - KEY_NAME: your SSH key
#    - SECURITY_GROUP: your security group ID
#    - SUBNET_ID: your subnet ID (optional)

# 2. Launch instance
./scripts/launch_training_with_auto_terminate.sh --spot --instance-type g5.xlarge --max-hours 4

# 3. Instance will automatically:
#    - Clone repo
#    - Install dependencies
#    - Download data from S3
#    - Train all models
#    - Upload models to S3
#    - Self-terminate when complete
```

### Method B: Manual training with auto-termination wrapper

```bash
# 1. Launch instance manually
aws ec2 run-instances ...

# 2. SSH to instance
ssh -i ~/.ssh/crpbot-gpu.pem ubuntu@<INSTANCE_IP>

# 3. Setup
git clone https://github.com/imnuman/crpbot.git
cd crpbot
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv pip install -e .

# 4. Download auto-termination script
curl -O https://raw.githubusercontent.com/imnuman/crpbot/main/scripts/auto_terminate_after_training.sh
chmod +x auto_terminate_after_training.sh

# 5. Run training with auto-termination (4 hour max)
nohup ./auto_terminate_after_training.sh "\
  uv run python apps/trainer/main.py --task lstm --coin BTC --epochs 15 && \
  uv run python apps/trainer/main.py --task lstm --coin ETH --epochs 15 && \
  uv run python apps/trainer/main.py --task lstm --coin SOL --epochs 15 && \
  aws s3 sync models/ s3://crpbot-ml-data/models/" \
  4 &

# 6. Monitor (optional)
tail -f /tmp/auto_terminate.log

# 7. Instance will self-terminate when training completes
```

---

## 💰 Cost Comparison

### Without Auto-Termination (Forgotten Instance)
```
g5.xlarge on-demand: $1.006/hour
Forgotten for 3 days: $1.006 × 72 hours = $72.43

Annual risk: ~$2,000+ if multiple instances forgotten
```

### With Auto-Termination
```
g5.xlarge spot:      $0.30/hour
Training time:       1 hour
Cost per training:   $0.30

Monthly (10 retrains): $3.00
Annual:               $36.00

Savings:              $1,964/year
```

---

## 🔒 Safety Features

### Multiple Layers of Protection

1. **Auto-termination wrapper**: Kills instance when training completes
2. **Timeout protection**: Max runtime limit (default: 4 hours)
3. **Monitoring script**: Alerts for long-running instances
4. **Cost estimation**: Shows estimated costs for all running instances
5. **Exit codes**: Non-zero exit = instances need attention

### Failure Modes

**What if auto-termination fails?**
- Timeout protection kicks in after max hours
- Monitoring script alerts after 2 hours
- Manual termination via AWS CLI or Console

**What if instance loses network?**
- Timeout protection still works (local timer)
- Monitoring script will detect long runtime

**What if training crashes?**
- Auto-termination wrapper detects process exit
- Instance terminates immediately
- Failed training doesn't accumulate costs

---

## 📋 Checklist: Setup Auto-Termination

### One-Time Setup

- [ ] Update `launch_training_with_auto_terminate.sh` with your AWS config:
  - [ ] KEY_NAME: Your SSH key name
  - [ ] SECURITY_GROUP: Your security group ID
  - [ ] SUBNET_ID: Your subnet ID (optional)
- [ ] Verify IAM role `EC2-S3-Access` exists with S3 permissions
- [ ] Test with dry-run: `./check_running_instances.sh`
- [ ] (Optional) Set up cron for automated monitoring

### Before Each Training

- [ ] Run `./check_running_instances.sh` to verify no forgotten instances
- [ ] Choose training method (A or B above)
- [ ] Launch with auto-termination enabled
- [ ] Monitor initial startup (first 5 minutes)

### After Training

- [ ] Verify instance terminated automatically
- [ ] Run `./check_running_instances.sh` to confirm
- [ ] Download models from S3 to local
- [ ] Test models in dry-run mode

---

## 🔧 Troubleshooting

### Issue: Instance didn't terminate

**Check logs**:
```bash
# On instance (if still accessible)
tail -100 /tmp/auto_terminate.log
```

**Manual termination**:
```bash
# Get instance ID
./check_running_instances.sh

# Terminate
aws ec2 terminate-instances --instance-ids i-xxxxx
```

### Issue: Training failed but instance still running

**Cause**: Auto-termination should have caught this, but timeout protection is backup

**Fix**: Wait for timeout or manually terminate

### Issue: Can't find running instances

**Cause**: Wrong region

**Fix**: Set correct region
```bash
export AWS_REGION=us-east-1
./check_running_instances.sh
```

---

## 📚 Related Documentation

- `MASTER_TRAINING_WORKFLOW.md` - Complete training workflow
- `CLAUDE.md` - Project architecture and guidelines
- `V7_CLOUD_DEPLOYMENT.md` - V7 runtime deployment

---

## 🎯 Summary

**Problem**: Forgotten GPU instances cost ~$72 per 3 days

**Solution**:
1. Use `launch_training_with_auto_terminate.sh` to launch instances with auto-termination
2. OR use `auto_terminate_after_training.sh` wrapper on manual instances
3. Monitor with `check_running_instances.sh` daily

**Result**: Training costs drop from potential $2,000+/year to $36/year

---

**Last Updated**: 2025-11-19
**Maintainer**: Builder Claude
