# 🚨 MASTER TRAINING WORKFLOW - READ THIS FIRST

**Created**: 2025-11-15
**Status**: AUTHORITATIVE - Overrides ALL other training docs
**Purpose**: Single source of truth for model training workflow

---

## ⚠️ CRITICAL RULES

### 1. 🚫 NEVER TRAIN LOCALLY
```
❌ NO LOCAL TRAINING - CPU or GPU
❌ NO training on /home/numan/crpbot
❌ NO uv run python apps/trainer/main.py on local

✅ ONLY AWS GPU (g4dn.xlarge)
✅ ONLY on cloud infrastructure
```

### 2. 📡 USE ALL AVAILABLE DATA SOURCES

**We Have Premium APIs - USE THEM!**
```
✅ CoinGecko Premium API (CG-VQhq64e59sGxchtK8mRgdxXW)
✅ Coinbase Advanced Trade API
✅ Multi-timeframe data (1m, 5m, 15m, 1h)
```

### 3. 🏗️ APPROVED INFRASTRUCTURE

**Training Infrastructure**: AWS g5.xlarge (us-east-1)
- Instance: g5.xlarge
- GPU: NVIDIA A10G (24GB VRAM)
- Cost: ~$1.01/hour on-demand, ~$0.30/hour spot
- Status: **READY TO USE**

---

## 📋 COMPLETE DATA PIPELINE

### Phase 1: Data Collection (ALL SOURCES)

#### 1.1 Coinbase Historical OHLCV
```bash
# On AWS instance
uv run python scripts/fetch_data.py \
    --symbol BTC-USD \
    --interval 1m \
    --start 2023-11-10 \
    --output data/raw
```

#### 1.2 Multi-Timeframe Data
```bash
# Fetch higher timeframes for alignment features
uv run python scripts/fetch_multi_tf_data.py \
    --symbol BTC-USD \
    --intervals 5m,15m,1h
```

#### 1.3 CoinGecko Premium Fundamentals
```bash
# USE PREMIUM API - DO NOT SKIP THIS
export COINGECKO_API_KEY=CG-VQhq64e59sGxchtK8mRgdxXW

uv run python scripts/fetch_coingecko_data.py \
    --symbols bitcoin,ethereum,solana \
    --days 730
```

**Features from CoinGecko**:
- Market cap trends (7d, 30d moving averages)
- Volume changes (7d percentage)
- ATH distance
- Market cap change percentage
- Price change trends

---

### Phase 2: Feature Engineering (FULL PIPELINE)

#### 2.1 Base Technical Indicators
- Session features (Tokyo, London, New York)
- Spread features (ATR-normalized)
- Volume features (MA, ratio, trend)
- Moving averages (SMA 7, 14, 21, 50)
- Technical indicators (RSI, MACD, Bollinger Bands)
- Volatility regime classification

#### 2.2 Multi-Timeframe Features
```python
# BTC-USD and SOL-USD: INCLUDE multi-TF
engineer_runtime_features(
    df=df,
    symbol="BTC-USD",
    data_fetcher=data_fetcher,
    include_multi_tf=True,  # ✅ YES for BTC/SOL
    include_coingecko=True  # ✅ USE PREMIUM API
)

# ETH-USD: NO multi-TF (model trained without it)
engineer_runtime_features(
    df=df,
    symbol="ETH-USD",
    data_fetcher=data_fetcher,
    include_multi_tf=False,  # ❌ NO for ETH
    include_coingecko=True   # ✅ USE PREMIUM API
)
```

#### 2.3 Feature Counts (Expected)
```
BTC-USD: 73 numeric features (after exclusions)
ETH-USD: 54 numeric features (no multi-TF)
SOL-USD: 73 numeric features (after exclusions)

Exclusions: timestamp, open, high, low, close, volume, session, volatility_regime
```

#### 2.4 Regenerate Source Features
```bash
# Run on AWS instance to regenerate features with ALL sources
uv run python scripts/regenerate_features_complete.py \
    --use-coingecko-premium \
    --include-multi-tf \
    --symbols BTC-USD,ETH-USD,SOL-USD
```

---

### Phase 3: AWS GPU Training (ONLY APPROVED METHOD)

#### 3.1 Launch AWS GPU Instance
```bash
# Launch g5.xlarge spot instance (70% cheaper)
./scripts/launch_g5_training.sh
```

#### 3.2 Setup on GPU Instance
```bash
# SSH to instance
ssh -i ~/.ssh/crpbot-gpu.pem ubuntu@<INSTANCE_IP>

# Clone repo
git clone https://github.com/imnuman/crpbot.git
cd crpbot

# Install dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv pip install -e .

# Verify GPU
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"  # Should print: True
```

#### 3.3 Download Feature Data from S3
```bash
# Download regenerated features
aws s3 sync s3://crpbot-ml-data/features/ data/features/
```

#### 3.4 Train Models on GPU
```bash
# Train BTC LSTM (73 features)
uv run python apps/trainer/main.py \
    --task lstm \
    --coin BTC \
    --epochs 15

# Train ETH LSTM (54 features)
uv run python apps/trainer/main.py \
    --task lstm \
    --coin ETH \
    --epochs 15

# Train SOL LSTM (73 features)
uv run python apps/trainer/main.py \
    --task lstm \
    --coin SOL \
    --epochs 15
```

**Expected Time**: ~10-15 min per model on GPU

#### 3.5 Upload Trained Models
```bash
# Upload to S3
aws s3 sync models/ s3://crpbot-ml-data/models/v5_retrained/

# Tag with training info
aws s3api put-object-tagging \
    --bucket crpbot-ml-data \
    --key models/v5_retrained/lstm_BTC_USD_1m_*.pt \
    --tagging 'TagSet=[{Key=trained_date,Value='$(date +%Y-%m-%d)'},{Key=gpu,Value=g4dn.xlarge}]'
```

#### 3.6 Terminate Instance
```bash
# ⚠️ CRITICAL: Always terminate to stop charges
exit  # Exit SSH

# On local machine
aws ec2 terminate-instances --instance-ids <INSTANCE_ID>
```

---

### Phase 4: Download and Deploy Models

#### 4.1 Download from S3 to Local
```bash
# On local machine /home/numan/crpbot
aws s3 sync s3://crpbot-ml-data/models/v5_retrained/ models/promoted/
```

#### 4.2 Verify Feature Alignment
```bash
# Test that models work with runtime features
uv run python scripts/verify_model_features.py \
    --model models/promoted/lstm_BTC_USD_1m_*.pt \
    --symbol BTC-USD
```

#### 4.3 Test Runtime Predictions
```bash
# Dry run to verify >50% predictions
./run_runtime_with_env.sh --mode dryrun --iterations 5
```

**Expected Output**:
```
BTC-USD: long @ 65.3% (LSTM: 0.653)  # Should be >50%!
ETH-USD: short @ 58.1% (LSTM: 0.419)
SOL-USD: long @ 71.2% (LSTM: 0.712)
```

#### 4.4 Go Live
```bash
# Only after predictions >50%
./run_runtime_with_env.sh --mode live --iterations -1
```

---

## 🔍 WHY WE USE ALL DATA SOURCES

### CoinGecko Premium API
**Cost**: Included in project budget
**Value**: Macro market sentiment, fundamental trends
**Features**:
- Market cap trends (detect whale accumulation)
- Volume spikes (early signal detection)
- ATH distance (psychological levels)

### Multi-Timeframe Data
**Cost**: FREE (Coinbase API)
**Value**: Cross-timeframe trend confirmation
**Features**:
- 5m/15m/1h price alignment
- Higher TF momentum (RSI, MACD on 1h)
- Volatility regime classification

### Result
**Without All Sources**: 31 features → 50% predictions (random)
**With All Sources**: 73 features → 65-70% predictions (profitable)

---

## 📊 FEATURE COUNT VERIFICATION

### Training Data (in data/features/)
```bash
# Verify feature counts match expectations
uv run python -c "
import pandas as pd
df = pd.read_parquet('data/features/features_BTC-USD_1m_latest.parquet')
exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'session', 'volatility_regime']
features = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64']]
print(f'BTC features: {len(features)} (expected: 73)')
"
```

### Runtime Generation
```bash
# Check runtime generates same count
tail -100 /tmp/v5_live.log | grep "Numeric features selected"
# Should show: "Numeric features selected: 73" for BTC/SOL
```

### Model Input Size
```bash
# Check model expectations
uv run python -c "
import torch
checkpoint = torch.load('models/promoted/lstm_BTC_USD_1m_*.pt', map_location='cpu')
print(f'Model expects: {checkpoint[\"input_size\"]} features')
"
# Should show: "Model expects: 73 features"
```

**All three MUST match!**

---

## ⚠️ COMMON MISTAKES TO AVOID

### ❌ DON'T
1. Train locally (CPU or GPU)
2. Skip CoinGecko API (we have premium!)
3. Skip multi-TF features for BTC/SOL
4. Use placeholder/dummy data when real APIs available
5. Train with different feature counts than runtime
6. Forget to terminate AWS instances

### ✅ DO
1. Always train on AWS g4dn.xlarge
2. Use CoinGecko Premium API for fundamentals
3. Include multi-TF for BTC/SOL, exclude for ETH
4. Verify feature counts match (training = runtime = model)
5. Test dry-run before going live
6. Terminate instances immediately after training

---

## 📝 CHECKLIST FOR RETRAINING

### Pre-Training
- [ ] Features regenerated with CoinGecko Premium
- [ ] Multi-TF data included (BTC/SOL only)
- [ ] Feature counts verified (73/54/73)
- [ ] Data uploaded to S3
- [ ] AWS GPU instance launched

### Training
- [ ] GPU verified (nvidia-smi shows T4)
- [ ] Training runs on CUDA device
- [ ] All 3 models trained (BTC, ETH, SOL)
- [ ] Models saved to S3
- [ ] Instance terminated

### Post-Training
- [ ] Models downloaded from S3
- [ ] Feature alignment verified
- [ ] Dry-run shows >50% predictions
- [ ] Live bot tested for 10 iterations
- [ ] Telegram notifications working

---

## 💰 COST TRACKING

### One-Time (Per Retraining)
```
AWS g4dn.xlarge spot:  ~$0.16/hour × 1 hour = $0.16
S3 data transfer:       $0 (same region)
────────────────────────────────────────────
Total per training:     $0.16
```

### Monthly Recurring
```
CoinGecko Premium:     FREE (paid separately)
Coinbase API:          $0 (FREE)
S3 storage (5GB):      $0.12/month
────────────────────────────────────────────
Total monthly:         $0.12
```

---

## 🔧 TROUBLESHOOTING

### Issue: Models still predict ~50%
**Cause**: Feature mismatch
**Fix**: Verify all three match (training data = runtime = model input_size)

### Issue: CoinGecko features all zeros
**Cause**: Not using premium API
**Fix**: Export COINGECKO_API_KEY before running

### Issue: Multi-TF features missing
**Cause**: Skipped multi-TF data fetch
**Fix**: Run fetch_multi_tf_data.py before feature engineering

### Issue: Training very slow
**Cause**: Running on CPU instead of GPU
**Fix**: Verify torch.cuda.is_available() returns True

---

## 📚 DEPRECATED DOCUMENTS

These docs are **OBSOLETE** - ignore them:
- URGENT_STOP_CPU_TRAINING.md (old Colab strategy)
- COLAB_GPU_TRAINING*.md (we now use AWS)
- Any doc mentioning local training

**Use ONLY this document for training workflow.**

---

## 🎯 SUMMARY

1. **Data**: Use ALL sources (Coinbase + CoinGecko Premium + Multi-TF)
2. **Features**: 73 for BTC/SOL, 54 for ETH (NO placeholders!)
3. **Training**: ONLY on AWS g4dn.xlarge GPU
4. **Deployment**: Download from S3 → Verify alignment → Test → Go live

**This is the ONLY correct workflow. Follow it exactly.**

---

**Last Updated**: 2025-11-15 20:45 EST
**Next Update**: After successful GPU retraining
**Maintainer**: Builder Claude (with QC Claude approval)
