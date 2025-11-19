# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 🚨 CRITICAL: Training Infrastructure Rules

**BEFORE ANY TRAINING OR FEATURE ENGINEERING**:
```bash
cat MASTER_TRAINING_WORKFLOW.md
```

### Key Rules
1. **NEVER train locally** - ONLY on AWS g4dn.xlarge GPU instance
2. **USE all premium APIs** - CoinGecko Premium (CG-VQhq64e59sGxchtK8mRgdxXW) is paid for
3. **ALWAYS verify feature alignment** - Training features = Runtime features = Model input_size
4. **ALWAYS terminate AWS instances after training** - Prevent unexpected charges

---

## 🤝 Dual-Environment Setup: QC Claude vs Builder Claude

This project operates across **two environments** with distinct roles:

### Identify Your Environment

```bash
# Check your working directory
pwd

# If output is: /home/numan/crpbot
#   → You are QC Claude (Local Machine)

# If output is: /root/crpbot
#   → You are Builder Claude (Cloud Server)
```

### QC Claude (Local Machine: `/home/numan/crpbot`)

**Primary Responsibilities**:
- ✅ **Quality Control**: Review Builder Claude's work before deployment
- ✅ **Documentation**: Maintain CLAUDE.md, PROJECT_MEMORY.md, and project docs
- ✅ **Testing**: Verify changes locally before cloud deployment
- ✅ **AWS Operations**: Run GPU training jobs on AWS (has AWS credentials)
- ✅ **Coordination**: Create handoff docs for Builder Claude

**Key Capabilities**:
- Has AWS credentials (`~/.aws/credentials`) for EC2 and S3 operations
- Can launch training jobs, upload/download from S3
- Local development environment for testing

**Common Tasks**:
```bash
# Review Builder Claude's changes
git pull origin main
git diff HEAD~5  # Review recent changes

# Run AWS training
aws s3 sync data/features/ s3://crpbot-ml-data/features/
# Then launch GPU instance for training

# Create handoff document
cat > HANDOFF_TO_BUILDER_CLAUDE.md <<EOF
# Changes Ready for Deployment

## What Changed
- [List changes made]

## Testing Done
- [List tests run]

## Deployment Steps
1. Pull latest from main
2. [Specific deployment commands]
EOF
git add HANDOFF_TO_BUILDER_CLAUDE.md
git commit -m "docs: handoff for deployment"
git push
```

### Builder Claude (Cloud Server: `root@178.156.136.185:~/crpbot`)

**Primary Responsibilities**:
- ✅ **Development**: Primary code development and implementation
- ✅ **Debugging**: Fix bugs and issues in production
- ✅ **Production Runtime**: Deploy and monitor live trading bot
- ✅ **Cloud Execution**: Run runtime, dashboard, and production services

**Key Capabilities**:
- Direct access to production environment
- Runs 24/7 live trading bot
- Has production database and logs

**Common Tasks**:
```bash
# Deploy changes
git pull origin main
# Restart runtime if needed

# Monitor production
tail -f /tmp/v6_65pct.log
cd apps/dashboard && uv run python app.py

# Check production status
sqlite3 tradingai.db "SELECT COUNT(*) FROM signals"
ps aux | grep python
```

### Coordination Protocol

**When QC Claude makes changes**:
1. Test locally first
2. Update documentation (CLAUDE.md, PROJECT_MEMORY.md)
3. Create handoff document with deployment steps
4. Push to GitHub
5. Notify Builder Claude (via handoff doc in repo)

**When Builder Claude makes changes**:
1. Develop and test on cloud
2. Commit with clear messages
3. Push to GitHub
4. Update QC Claude if review needed

**Cross-Environment Sync**:
```bash
# QC Claude → Builder Claude (code changes)
git push origin main
# Builder Claude pulls with: git pull origin main

# QC Claude → Builder Claude (models from AWS training)
aws s3 sync models/ s3://crpbot-ml-data/models/v6_retrained/
# Builder Claude downloads with: aws s3 sync s3://crpbot-ml-data/models/v6_retrained/ models/promoted/

# Builder Claude → QC Claude (production data/logs for analysis)
# Use git for small files, S3 for large data
```

### Decision Matrix: Who Does What?

| Task | QC Claude | Builder Claude |
|------|-----------|----------------|
| AWS GPU Training | ✅ Primary | ✅ Can do (has creds) |
| Code Development | ✅ Review/Test | ✅ Primary |
| Documentation Updates | ✅ Primary | ✅ Can update |
| Production Deployment | ❌ No access | ✅ Primary |
| Bug Fixes | ✅ Can fix locally | ✅ Primary for prod |
| Model Evaluation | ✅ Primary | ✅ Can help |
| Live Runtime Monitoring | ❌ No access | ✅ Primary |
| Feature Engineering | ✅ Primary | ✅ Can help |

**Note**: Both environments have AWS credentials configured. QC Claude is designated as primary for training operations, but Builder Claude can handle AWS operations when needed.

---

## 📋 V7 Ultimate Implementation Guide

**CRITICAL**: V7 STEP 4 COMPLETE - Ready for cloud deployment on `feature/v7-ultimate` branch.

### What is V7 Ultimate?

V7 Ultimate is a manual trading system (signal generation only) based on Renaissance Technologies methodology:
- **7 Mathematical Theories**: Shannon Entropy, Hurst Exponent, Kolmogorov Complexity, Market Regime, Risk Metrics, Fractal Dimension, **Market Context (CoinGecko)** ⭐ **NEW**
- **Enhanced ML**: 4-layer FNN with BatchNorm, Dropout, Temperature Scaling
- **Premium Data**: CoinGecko Analyst API ($129/month) ✅ **INTEGRATED**
- **LLM Synthesis**: DeepSeek API integration ($5/day budget) ✅ **COMPLETE**
- **Expected Performance**: 58-65% initially → 70-75% with learning

### Implementation Status

**Current Step**: STEP 4 COMPLETE → Ready for STEP 5 (Dashboard/Telegram)

✅ **Completed Components**:
1. ✅ V7 Runtime Orchestrator (551 lines) - `apps/runtime/v7_runtime.py`
2. ✅ **7 Mathematical Theories** - All implemented in `libs/theories/` (including CoinGecko Market Context)
3. ✅ **CoinGecko Integration** - `libs/data/coingecko_client.py` + `libs/theories/market_context.py`
4. ✅ DeepSeek LLM Integration - Complete in `libs/llm/` ($5/day budget)
5. ✅ Bayesian Learning Framework - `libs/bayesian/bayesian_learner.py`
6. ✅ Rate Limiting (30 signals/hour)
7. ✅ Cost Controls ($5/day, $150/month budgets)
8. ✅ FTMO Rules Integration
9. ✅ Documentation Cleanup (172 → 7 essential files)

📋 **Next Steps**:
- Deploy to cloud server (178.156.136.185)
- Add DeepSeek API key to `.env`
- Run continuous V7 runtime
- STEP 5: Dashboard/Telegram integration

### V7 Key Principles

1. **No Auto-Execution**: Manual trading only (human confirms each signal)
2. **Mathematical Foundation**: Every signal backed by mathematical evidence
3. **Continuous Learning**: Bayesian updates from trade outcomes
4. **Risk Management**: Monte Carlo simulation for every trade
5. **Quality over Quantity**: 2-5 high-quality signals/day (not 20-30 low-quality)

### V7 Files Created (13 New Files)

**Runtime**:
- `apps/runtime/v7_runtime.py` (551 lines) - Main V7 orchestrator

**LLM Integration** (`libs/llm/`):
- `deepseek_client.py` - DeepSeek API client
- `signal_synthesizer.py` - Theory → LLM prompt converter
- `signal_parser.py` - LLM response → structured signal parser
- `signal_generator.py` - Complete signal generation orchestrator

**Mathematical Theories** (`libs/theories/`):
- `shannon_entropy.py` - Market predictability analysis
- `hurst_exponent.py` - Trend persistence detection
- `kolmogorov_complexity.py` - Pattern complexity measurement
- `market_regime.py` - Bull/bear/sideways classification
- `risk_metrics.py` - VaR, Sharpe ratio, volatility
- `fractal_dimension.py` - Market structure analysis
- `market_context.py` - **CoinGecko macro analysis (7th theory)** ⭐ **NEW**

**CoinGecko Integration** (`libs/data/`):
- `coingecko_client.py` - CoinGecko Analyst API client ($129/month)

**Bayesian Learning** (`libs/bayesian/`):
- `bayesian_learner.py` - Beta distribution learning from outcomes

**Documentation**:
- `V7_CLOUD_DEPLOYMENT.md` - Complete deployment guide ⭐

### Deploying V7 to Cloud

**Prerequisites**:
1. DeepSeek API key (get from https://platform.deepseek.com/)
2. Add to `.env`: `DEEPSEEK_API_KEY=sk-...`

**Deployment Commands** (see `V7_CLOUD_DEPLOYMENT.md` for full guide):
```bash
# On cloud server (178.156.136.185)
cd ~/crpbot
git pull origin feature/v7-ultimate

# Add DeepSeek API key to .env
nano .env

# Test with 1 iteration
.venv/bin/python3 apps/runtime/v7_runtime.py --iterations 1 --sleep-seconds 10

# Run continuous (background)
nohup .venv/bin/python3 apps/runtime/v7_runtime.py \
  --iterations -1 \
  --sleep-seconds 120 \
  > /tmp/v7_runtime.log 2>&1 &

# Monitor
tail -f /tmp/v7_runtime.log
```

---

## 🔄 GitHub Sync Protocol (CRITICAL)

**ALWAYS sync with GitHub before and after work** to keep both environments in sync.

### Pre-Work Sync (ALWAYS RUN FIRST)

```bash
# Pull latest changes before starting any work
git pull origin main

# If you have local changes that conflict:
git stash                    # Save your work temporarily
git pull origin main         # Pull latest
git stash pop                # Restore your work
# Resolve conflicts if needed
```

### Post-Work Sync (ALWAYS RUN AFTER CHANGES)

```bash
# After making changes, commit and push
git add .
git status                   # Review what you're committing
git commit -m "descriptive message"
git push origin main
```

### Sync Frequency

**QC Claude (Local)**:
- ✅ Pull before starting any documentation updates
- ✅ Pull before AWS training operations
- ✅ Push after updating CLAUDE.md or PROJECT_MEMORY.md
- ✅ Push after creating handoff documents

**Builder Claude (Cloud)**:
- ✅ Pull at start of each session
- ✅ Pull before deploying changes
- ✅ Push after bug fixes or feature development
- ✅ Push production status updates

### Conflict Resolution

If `git pull` shows conflicts:

```bash
# 1. Identify conflicted files
git status

# 2. Open conflicted files and resolve manually
# Look for <<<<<<< HEAD markers

# 3. After resolving:
git add <resolved-files>
git commit -m "fix: resolve merge conflicts"
git push origin main
```

### Sync Verification

```bash
# Check if you're in sync
git status
# Should show: "Your branch is up to date with 'origin/main'"

# See recent commits from other Claude
git log --oneline -10

# See what changed recently
git diff HEAD~5
```

### Emergency: Force Sync (Use Carefully)

```bash
# If local changes should be discarded (DESTRUCTIVE):
git fetch origin
git reset --hard origin/main

# If you want to keep a backup first:
git branch backup-$(date +%Y%m%d-%H%M%S)
git fetch origin
git reset --hard origin/main
```

---

## 📋 Quick Commands

### Development
```bash
# Setup (first run)
make setup              # Install deps + pre-commit hooks

# Code quality
make fmt                # Format with ruff
make lint               # Lint with ruff + mypy
make test               # Run all tests
make unit               # Run unit tests only
make smoke              # Run 5-min smoke backtest
```

### Runtime
```bash
# Dryrun mode (testing, no real trades)
./run_runtime_with_env.sh --mode dryrun --iterations 5

# Live mode (production)
./run_runtime_with_env.sh --mode live --iterations -1

# Dashboard (monitor signals)
cd apps/dashboard && uv run python app.py
# Access: http://localhost:8050
```

### Training (AWS GPU ONLY)
```bash
# NEVER run locally! See MASTER_TRAINING_WORKFLOW.md for full workflow

# On AWS GPU instance only:
uv run python apps/trainer/main.py --task lstm --coin BTC --epochs 15
uv run python apps/trainer/main.py --task lstm --coin ETH --epochs 15
uv run python apps/trainer/main.py --task lstm --coin SOL --epochs 15
```

### Testing
```bash
# All tests
uv run pytest tests/

# Specific test
uv run pytest tests/unit/test_ftmo_rules.py -v

# With coverage
uv run pytest tests/ --cov=apps --cov=libs --cov-report=html
```

### Debugging
```bash
# Check feature alignment (CRITICAL)
python -c "import pandas as pd; df = pd.read_parquet('data/features/features_BTC-USD_1m_latest.parquet'); print(f'Features: {len([c for c in df.columns if c not in [\"timestamp\", \"open\", \"high\", \"low\", \"close\", \"volume\", \"session\", \"volatility_regime\"] and df[c].dtype in [\"float64\", \"int64\"]])}')"

# Check model input size
python -c "import torch; c = torch.load('models/promoted/lstm_BTC-USD_v6_enhanced.pt', map_location='cpu'); print(f'Model expects: {c[\"input_size\"]} features')"

# Check runtime logs
tail -100 /tmp/v5_live.log | grep "Numeric features selected"

# Test Coinbase connection
uv run python test_kraken_connection.py
```

---

## 🏗️ Architecture Overview

### Project Structure
```
crpbot/
├── apps/
│   ├── trainer/          # Model training (LSTM, Transformer)
│   ├── runtime/          # Production signal generation
│   └── dashboard/        # Flask monitoring dashboard
├── libs/
│   ├── features/         # Feature engineering pipelines
│   ├── config/           # Pydantic settings
│   ├── db/              # SQLAlchemy models
│   └── utils/           # Timezone, helpers
├── scripts/             # Data fetching, feature engineering, monitoring
├── models/              # Trained model weights (.pt files)
├── data/                # Training data (parquet files)
└── tests/               # Unit, integration, smoke tests
```

### Data Flow
```
Raw API Data → Feature Engineering → Model Training → S3 Storage → Runtime Inference
     ↓              ↓                      ↓              ↓              ↓
Coinbase/      73/54/72 features      AWS GPU       models/        Live signals
Kraken         + CoinGecko            (g4dn.xlarge)  promoted/      + Telegram
```

### Model System

**Current Development**: V7 Ultimate (Enhanced 4-layer FNN + Mathematical Framework)
- **BTC-USD**: 71.4% RF accuracy, 71.1% NN accuracy, 72 features
- **ETH-USD**: 68.9% RF accuracy, 69.8% NN accuracy, 72 features
- **SOL-USD**: 70.9% RF accuracy, 69.7% NN accuracy, 72 features
- **Architecture**: 72→256→128→64→3 with BatchNorm + Dropout (0.3)
- **Status**: In development (feature/v7-ultimate branch)

**V7 Enhancements**:
- Temperature scaling (T=2.5) for calibrated confidence
- Batch normalization layers
- Dropout regularization (0.3)
- 6 mathematical theories integration (Shannon, Hurst, Markov, Kalman, Bayesian, Monte Carlo)
- CoinGecko Analyst API data enrichment
- DeepSeek LLM synthesis (planned)

**Model Evolution**:
- V5 FIXED: 3-layer LSTM (73/54 features, multi-TF for BTC/SOL only)
- V6 Real: 2-layer LSTM (31 features, uniform across symbols)
- V6 Enhanced: 4-layer FNN (72 features, Amazon Q engineered)
- V7 Ultimate: Enhanced FNN + Mathematical Framework ⭐ **IN DEVELOPMENT**

**Ensemble System** (`apps/runtime/ensemble.py`):
- Loads best available model per symbol
- Confidence threshold: 60-65% (configurable via CONFIDENCE_THRESHOLD env var)
- Manual signal generation (no auto-execution)

---

## 🔑 Critical Concepts

### Feature Count Alignment (MOST IMPORTANT)

**The Problem**: If training uses 73 features but runtime generates 54, predictions are random (~50%).

**The Solution**: Ensure exact match across all three:
1. **Training data** (parquet files in `data/features/`)
2. **Runtime generation** (`apps/runtime/runtime_features.py`)
3. **Model expectations** (saved in checkpoint['input_size'])

**Verification**:
```bash
# 1. Training data
python -c "import pandas as pd; df = pd.read_parquet('data/features/features_BTC-USD_1m_latest.parquet'); print(len([c for c in df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'session', 'volatility_regime'] and df[c].dtype in ['float64', 'int64']]))"

# 2. Runtime logs
tail -100 /tmp/v5_live.log | grep "Numeric features selected"

# 3. Model checkpoint
python -c "import torch; c = torch.load('models/promoted/lstm_BTC-USD_v6_enhanced.pt', map_location='cpu'); print(c['input_size'])"
```

**All three MUST match!**

### Feature Engineering Pipeline

**V6 Enhanced Features** (72 total):
- Base technical: SMA, EMA (5/10/20/50/200)
- Momentum: ROC, momentum over multiple windows
- Oscillators: RSI (14/21/30), Stochastic, Williams %R
- Trend: MACD (12/26 and 5/35 variants), Bollinger Bands (20/50)
- Price channels, volatility, ATR
- Lagged features: returns and volume (lag 1/2/3/5)
- Price ratios: high/low, close/open, price to SMA/EMA ratios

**Feature Exclusions** (always exclude from model input):
- `timestamp`, `open`, `high`, `low`, `close`, `volume` (raw OHLCV)
- `session`, `volatility_regime` (categorical - convert to numeric if needed)

**Multi-Timeframe Features** (V5/V6 Real only):
- 5m/15m/1h OHLCV data
- TF alignment scores
- Higher timeframe technical indicators

**CoinGecko Premium Features** (if using):
- Market cap trends, ATH distance, volume changes
- Requires `COINGECKO_API_KEY=CG-VQhq64e59sGxchtK8mRgdxXW`

### FTMO Risk Management

**Hard Limits** (`apps/runtime/ftmo_rules.py`):
- Daily loss: 4.5% max (enforced strictly)
- Total loss: 9% max (account termination)
- Position sizing: Risk-based (1-2% per trade)

**Rate Limiting** (`apps/runtime/rate_limiter.py`):
- High tier: 5 signals/hour max
- Medium tier: 10 signals/hour max
- Low tier: Unlimited (logged only)

**Kill Switch**:
- Set `KILL_SWITCH=true` in `.env` to halt all trading
- Instant stop via Telegram command (if configured)

---

## 🧪 Testing Strategy

### Test Types

**Unit Tests** (`tests/unit/`):
- FTMO rules enforcement
- Confidence scoring
- Rate limiting
- Dataset utilities

**Integration Tests** (`tests/integration/`):
- Runtime guardrails (FTMO + rate limiter + kill switch)

**Smoke Tests** (`tests/smoke/`):
- 5-minute backtest to verify end-to-end flow

### Running Tests

```bash
# All tests
make test  # or: uv run pytest tests/

# Specific category
make unit   # Unit tests only
make smoke  # Smoke tests only

# Single test file
uv run pytest tests/unit/test_ftmo_rules.py -v

# With verbose output and logging
uv run pytest tests/ -v -s

# With coverage report
uv run pytest tests/ --cov=apps --cov=libs --cov-report=html
```

---

## ☁️ AWS Infrastructure

### GPU Training (g4dn.xlarge)
- **GPU**: NVIDIA T4 (16GB VRAM)
- **Cost**: $0.526/hour on-demand, $0.158/hour spot
- **Region**: us-east-1 (N. Virginia)
- **Training time**: ~10-15 min per model, ~1 hour for all 3

### S3 Storage
- **Bucket**: `crpbot-ml-data` or `crpbot-ml-data-20251110`
- **Structure**:
  - `raw/` - Historical OHLCV data
  - `features/` - Engineered features (parquet)
  - `models/v6_retrained/` - Trained model weights

### Training Workflow (AWS GPU)
```bash
# See MASTER_TRAINING_WORKFLOW.md for complete workflow

# Quick summary:
1. Fetch data locally → Engineer features → Upload to S3
2. Launch g4dn.xlarge spot instance
3. SSH to instance → Clone repo → Download features from S3
4. Train models on GPU (15 epochs each, ~10-15 min per model)
5. Upload models to S3
6. Terminate instance (CRITICAL - stop billing)
7. Download models locally → Verify → Deploy
```

**Cost Tracking**:
- One training run: ~$0.16 (spot) or ~$0.53 (on-demand)
- Monthly (10-15 runs): ~$5-8
- S3 storage (5GB): ~$0.12/month

---

## 📝 Configuration

### Environment Variables (`.env`)

**Data Provider**:
```bash
DATA_PROVIDER=coinbase  # Options: coinbase, kraken, cryptocompare

# Coinbase Advanced Trade API (JWT auth)
COINBASE_API_KEY_NAME=organizations/.../apiKeys/...
COINBASE_API_PRIVATE_KEY=-----BEGIN EC PRIVATE KEY-----...

# CoinGecko Premium
COINGECKO_API_KEY=CG-VQhq64e59sGxchtK8mRgdxXW
```

**Runtime**:
```bash
CONFIDENCE_THRESHOLD=0.65  # Min confidence to emit signal
KILL_SWITCH=false          # Emergency stop
RUNTIME_MODE=dryrun        # or "live"
```

**Database**:
```bash
DB_URL=sqlite:///tradingai.db
# Or PostgreSQL: postgresql+psycopg://user:pass@localhost:5432/tradingai
```

**Telegram** (optional):
```bash
TELEGRAM_TOKEN=...
TELEGRAM_CHAT_ID=...
```

### Model Configuration
- **Model path**: `models/promoted/` (production models)
- **Model version**: Set via `MODEL_VERSION` env var
- **Loading priority**: V6 Enhanced → V6 Real → V5 FIXED

---

## 🚨 Common Pitfalls

### 1. Training Locally on CPU
**Symptom**: Training takes 60-90 min per model
**Fix**: ALWAYS use AWS GPU (see `MASTER_TRAINING_WORKFLOW.md`)

### 2. Feature Count Mismatch
**Symptom**: Models predict ~50% (random)
**Fix**: Verify alignment (see "Feature Count Alignment" above)

### 3. Multi-TF Features Missing
**Symptom**: Feature count too low (31 instead of 73)
**Fix**: Set `include_multi_tf=True` in runtime feature engineering

### 4. CoinGecko Features All Zeros
**Symptom**: CoinGecko features show 0.0 values
**Fix**: Export `COINGECKO_API_KEY=CG-VQhq64e59sGxchtK8mRgdxXW` before running

### 5. AWS Instance Not Terminated
**Symptom**: Unexpected AWS charges
**Fix**: ALWAYS run `aws ec2 terminate-instances --instance-ids <id>` after training

### 6. Predictions Too Low (<5%)
**Symptom**: Confidence consistently below threshold
**Expected**: Models are cautious during ranging/choppy markets (this is good!)

---

## 📚 Key Documentation

**V7 Documentation** (CURRENT):
- `V7_CLOUD_DEPLOYMENT.md` - **STEP 4 Complete - Cloud deployment guide** ⭐
- `apps/runtime/v7_runtime.py` - Main V7 runtime orchestrator (551 lines)
- `V7_PROJECT_STATUS_AND_ROADMAP.md` - Original implementation roadmap
- `PROJECT_MEMORY.md` - Session continuity and dual-environment setup
- `v7_training_summary.json` - Latest training metrics
- `V6_DIAGNOSTIC_AND_V7_PLAN.md` - Why V7 was needed

**Training & Infrastructure**:
- `MASTER_TRAINING_WORKFLOW.md` - AUTHORITATIVE training guide
- `CLAUDE.md` - This file (project architecture)
- `README.md` - Project overview

**Architecture** (Code):
- `apps/trainer/models/lstm.py` - Model architectures
- `apps/runtime/ensemble.py` - Model loading and inference
- `apps/runtime/runtime_features.py` - Runtime feature pipeline
- `apps/runtime/signal_formatter.py` - V7 signal formatting
- `libs/config/config.py` - Configuration system

**Deprecated** (ignore these):
- Any docs mentioning Colab training
- Any docs mentioning local CPU training
- Any docs predating 2025-11-15
- V6-specific deployment docs (superseded by V7)

---

## 🎯 Development Workflow

1. **Create feature branch**: `git checkout -b feat/feature-name`
2. **Make changes**: Edit code, write tests
3. **Pre-commit hooks run automatically**: Format, lint, type-check
4. **Run tests**: `make test` or `uv run pytest tests/`
5. **Verify feature alignment** (if touching feature engineering)
6. **Push and create PR**: CI checks must pass
7. **Merge to main**: Deploy to production

---

## 🔮 Current Status (Nov 2025)

**Active Branch**: `feature/v7-ultimate`
**Current Phase**: V7 Ultimate - Manual Signal System (Implementation)

**V7 Status**:
- **STEP 4 COMPLETE**: Signal generation pipeline fully implemented (13 new files)
- **Ready for cloud deployment**: Just needs DeepSeek API key added to `.env`
- Training metrics: 70.2% avg accuracy (RF), 60.2% avg confidence
- Architecture: Enhanced 4-layer FNN with BatchNorm + Dropout
- Runtime: 551-line orchestrator with 6 theories + DeepSeek LLM synthesis
- Method: Manual trading (signal generation only, no auto-execution)
- Cost: ~$0.0003 per signal (~$1.75/month at 6 signals/hour)

**V7 Framework** (All Implemented ✅):
- 6 mathematical theories (Shannon, Hurst, Kolmogorov, Market Regime, Risk Metrics, Fractal)
- DeepSeek LLM synthesis ($0.27/M input, $1.10/M output tokens)
- CoinGecko Analyst API integration ($129/month)
- Rate limiting: 6 signals/hour
- Cost controls: $3/day, $100/month budgets
- Bayesian learning from trade outcomes
- Expected: 58-65% win rate initially → 70-75% with learning

**Symbols Tracked**: BTC-USD, ETH-USD, SOL-USD
**Data Sources**: Coinbase Advanced Trade API + CoinGecko Premium
**Runtime Mode**: Ready for deployment (V7 STEP 4 complete, awaiting cloud deployment)

**Key Files**:
- `V7_CLOUD_DEPLOYMENT.md` - **STEP 4 deployment guide** ⭐
- `apps/runtime/v7_runtime.py` - Main V7 runtime (551 lines)
- `libs/llm/*` - DeepSeek integration (4 files)
- `libs/theories/*` - 6 mathematical theories (6 files)
- `libs/bayesian/*` - Bayesian learning (1 file)
- `PROJECT_MEMORY.md` - Session continuity and dual-environment setup
- `v7_training_summary.json` - Latest training metrics

**Monitoring**:
```bash
# Dashboard
cd apps/dashboard && uv run python app.py

# Logs (check latest log file)
ls -lt /tmp/*.log | head -3

# Database
sqlite3 tradingai.db "SELECT * FROM signals ORDER BY timestamp DESC LIMIT 10"
```

---

**Last Updated**: 2025-11-18
**Next Milestone**: Deploy V7 to cloud server → STEP 5 (Dashboard/Telegram integration)
