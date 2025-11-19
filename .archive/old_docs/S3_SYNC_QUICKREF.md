# S3 Sync Quick Reference

Fast commands for syncing models and data between local/cloud and S3.

---

## 📦 Models Sync

### Upload Models to S3 (After Training)
```bash
# Upload all models
aws s3 sync models/ s3://crpbot-market-data-dev/models/

# Upload only PyTorch models
aws s3 sync models/ s3://crpbot-market-data-dev/models/ --exclude "*" --include "*.pt"

# Upload promoted models only
aws s3 sync models/promoted/ s3://crpbot-market-data-dev/models/promoted/
```

### Download Models from S3
```bash
# Download all models
aws s3 sync s3://crpbot-market-data-dev/models/ models/

# Download only promoted models
aws s3 sync s3://crpbot-market-data-dev/models/promoted/ models/promoted/

# Download specific model
aws s3 cp s3://crpbot-market-data-dev/models/lstm_BTC_USD_1m_*.pt models/
```

---

## 📊 Data Sync

### Upload Data to S3 (After Fetching/Engineering)
```bash
# Upload raw OHLCV data
aws s3 sync data/raw/ s3://crpbot-market-data-dev/data/raw/

# Upload engineered features
aws s3 sync data/features/ s3://crpbot-market-data-dev/data/features/

# Upload specific symbol
aws s3 sync data/raw/ s3://crpbot-market-data-dev/data/raw/ --exclude "*" --include "BTC-USD*"
```

### Download Data from S3
```bash
# Download all raw data
aws s3 sync s3://crpbot-market-data-dev/data/raw/ data/raw/

# Download specific symbol (faster for QC)
aws s3 sync s3://crpbot-market-data-dev/data/raw/ data/raw/ --exclude "*" --include "BTC-USD*"

# Download features
aws s3 sync s3://crpbot-market-data-dev/data/features/ data/features/
```

---

## 🔍 List S3 Contents

### Check What's in S3
```bash
# List all models
aws s3 ls s3://crpbot-market-data-dev/models/ --recursive --human-readable

# List promoted models only
aws s3 ls s3://crpbot-market-data-dev/models/promoted/ --human-readable

# List raw data
aws s3 ls s3://crpbot-market-data-dev/data/raw/ --recursive --human-readable

# List features
aws s3 ls s3://crpbot-market-data-dev/data/features/ --recursive --human-readable

# Get total size
aws s3 ls s3://crpbot-market-data-dev/ --recursive --summarize
```

---

## 🔄 Typical Workflows

### Cloud Claude: After Training
```bash
cd /root/crpbot

# 1. Train models
uv run python apps/trainer/main.py --task lstm --coin BTC --epochs 15

# 2. Upload to S3
aws s3 sync models/ s3://crpbot-market-data-dev/models/

# 3. Commit code changes
git add . && git commit -m "feat: train BTC LSTM model" && git push

# 4. Report to Local Claude for QC
```

### Local Claude: QC Review
```bash
cd /home/numan/crpbot

# 1. Pull latest code
git pull origin main

# 2. Download models from S3
aws s3 sync s3://crpbot-market-data-dev/models/ models/

# 3. Evaluate models
uv run python scripts/evaluate_model.py --model models/lstm_BTC_USD_1m_*.pt

# 4. Approve/reject model promotion
```

### Cloud Claude: After Data Fetch
```bash
cd /root/crpbot

# 1. Fetch data
uv run python scripts/fetch_data.py --symbol BTC-USD --interval 1m --start 2023-11-10

# 2. Engineer features
uv run python scripts/engineer_features.py --input data/raw/BTC-USD_*.parquet

# 3. Upload to S3
aws s3 sync data/raw/ s3://crpbot-market-data-dev/data/raw/
aws s3 sync data/features/ s3://crpbot-market-data-dev/data/features/

# 4. Commit and push
git add . && git commit -m "data: fetch and engineer BTC features" && git push
```

### Local Claude: Data QC
```bash
cd /home/numan/crpbot

# 1. Download specific data for validation
aws s3 sync s3://crpbot-market-data-dev/data/raw/ data/raw/ --exclude "*" --include "BTC-USD*"

# 2. Run data quality checks
uv run python scripts/validate_data_quality.py --symbol BTC-USD

# 3. Approve data quality
```

---

## ⚡ Performance Tips

### Faster Sync with Filters
```bash
# Only sync new/modified files (default behavior)
aws s3 sync local/ s3://bucket/

# Delete remote files not in local (careful!)
aws s3 sync local/ s3://bucket/ --delete

# Dry-run to see what would change
aws s3 sync local/ s3://bucket/ --dryrun
```

### Parallel Transfers
```bash
# Increase max concurrent requests (default: 10)
aws configure set default.s3.max_concurrent_requests 20

# Increase multipart chunk size (default: 8MB)
aws configure set default.s3.multipart_chunksize 16MB
```

### Exclude/Include Patterns
```bash
# Exclude logs, include only models
aws s3 sync models/ s3://bucket/models/ --exclude "*.log" --include "*.pt"

# Multiple patterns
aws s3 sync data/ s3://bucket/data/ \
  --exclude "*" \
  --include "BTC-USD*" \
  --include "ETH-USD*" \
  --include "SOL-USD*"
```

---

## 🚨 Safety Checks

### Before Deleting from S3
```bash
# ALWAYS dry-run first
aws s3 rm s3://bucket/path/ --recursive --dryrun

# Then execute if safe
aws s3 rm s3://bucket/path/ --recursive
```

### Before Sync with --delete
```bash
# ALWAYS dry-run first
aws s3 sync local/ s3://bucket/ --delete --dryrun

# Review what would be deleted, then execute
aws s3 sync local/ s3://bucket/ --delete
```

---

## 📋 Daily Checklist

### Morning Sync (Local)
```bash
git pull origin main
aws s3 sync s3://crpbot-market-data-dev/models/ models/
aws s3 sync s3://crpbot-market-data-dev/data/features/ data/features/
```

### Evening Upload (Cloud)
```bash
aws s3 sync models/ s3://crpbot-market-data-dev/models/
aws s3 sync data/raw/ s3://crpbot-market-data-dev/data/raw/
git add . && git commit -m "..." && git push
```

---

**S3 Bucket**: `s3://crpbot-market-data-dev/`
**Region**: `us-east-1`
**Access**: Both local and cloud have read/write access
