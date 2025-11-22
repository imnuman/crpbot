# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 🚨 CRITICAL: Dual-Environment Setup

This project operates across **two environments** - identify yours first:

```bash
pwd
# /home/numan/crpbot  → QC Claude (Local - Quality Control & Training)
# /root/crpbot        → Builder Claude (Cloud - Production & Deployment)
```

### QC Claude (Local: `/home/numan/crpbot`)
- **Quality Control**: Review before deployment
- **AWS Training**: GPU training on g4dn.xlarge (NEVER local)
- **Documentation**: Maintain CLAUDE.md, handoff docs
- **Testing**: Local verification

### Builder Claude (Cloud: `root@178.156.136.185`)
- **Production Runtime**: V7 running 24/7
- **Deployment**: Primary development and bug fixes
- **Monitoring**: Live system, database, logs
- **Database**: SQLite (`/root/crpbot/tradingai.db`)

### Sync Protocol
```bash
# ALWAYS sync before work
git pull origin main

# Push changes after work
git add . && git commit -m "message" && git push origin main
```

---

## 🎯 V7 Ultimate - Current Production System

**STATUS**: ✅ **OPERATIONAL** - Monitoring phase (2025-11-22 to 2025-11-25)

### Architecture

**V7 Ultimate = 11 Mathematical Theories + DeepSeek LLM**

**Theories** (Production):
1. Shannon Entropy - Market predictability
2. Hurst Exponent - Trend persistence
3. Markov Regime (6-state) - Market state detection
4. Kalman Filter - Price denoising
5. Bayesian Inference - Win rate learning
6. Monte Carlo - Risk simulation (10k scenarios)
7. Random Forest - Pattern validation
8. Autocorrelation - Time series dependencies
9. Stationarity - Mean reversion testing
10. Variance Analysis - Volatility regimes
11. Market Context (CoinGecko) - Macro analysis

**Signal Flow**:
```
Market Data (Coinbase)
  → 11 Theory Analysis
  → DeepSeek LLM Synthesis
  → Signal Parsing
  → Paper Trading
  → Performance Tracking
```

**Key Files**:
- `apps/runtime/v7_runtime.py` - Main runtime (33KB)
- `libs/llm/signal_generator.py` - Orchestrates theory → LLM → signal
- `libs/analysis/*` - Core 6 theories (Shannon, Hurst, Markov, Kalman, Bayesian, Monte Carlo)
- `libs/theories/*` - Statistical 4 theories (RF, Autocorr, Stationarity, Variance)
- `libs/theories/market_context.py` - CoinGecko integration

### Production Status (2025-11-22)
- **Runtime**: PID 2620770, 6 hours uptime
- **Symbols**: 10 (BTC, ETH, SOL, XRP, DOGE, ADA, AVAX, LINK, POL, LTC)
- **Paper Trades**: 13 (target: 20+ for statistical analysis)
- **Win Rate**: 53.8%, P&L: +5.48%
- **Dashboard**: http://178.156.136.185:3000
- **Database**: SQLite (local, 4,075 signals)

### Current Phase: Data Collection
**Mission**: Collect 20+ paper trades before optimization
**Review Date**: 2025-11-25 (Monday)
**Decision Criteria**: Sharpe ratio calculation
- Sharpe < 1.0 → Implement Phase 1 enhancements
- Sharpe 1.0-1.5 → Monitor 1 more week
- Sharpe > 1.5 → Continue as-is

---

## 📋 Quick Commands

### Development
```bash
make setup              # Initial setup (deps + pre-commit hooks)
make fmt                # Format with ruff
make lint               # Lint with ruff + mypy
make test               # All tests
make unit               # Unit tests only
make smoke              # 5-min smoke backtest
```

### V7 Runtime (Production)
```bash
# Check V7 status (Builder Claude)
ps aux | grep v7_runtime | grep -v grep

# Monitor logs
tail -f /tmp/v7_runtime_*.log

# Restart V7 (if needed)
pkill -f v7_runtime.py
nohup .venv/bin/python3 apps/runtime/v7_runtime.py \
  --iterations -1 \
  --sleep-seconds 300 \
  --max-signals-per-hour 3 \
  > /tmp/v7_runtime_$(date +%Y%m%d_%H%M).log 2>&1 &

# Check database
sqlite3 tradingai.db "SELECT COUNT(*) FROM signals;"
sqlite3 tradingai.db "SELECT COUNT(*) FROM signal_results;"
```

### AWS Training (QC Claude ONLY)
```bash
# CRITICAL: Read workflow first
cat MASTER_TRAINING_WORKFLOW.md

# NEVER train locally - ONLY on AWS g4dn.xlarge GPU
# Cost: $0.16/run (spot), ~10-15 min per model
# ALWAYS terminate instance after training

# Quick workflow:
# 1. Engineer features locally
# 2. Upload to S3
# 3. Launch g4dn.xlarge spot instance
# 4. Train on GPU
# 5. Upload models to S3
# 6. TERMINATE INSTANCE (critical!)
# 7. Download and deploy
```

### Database Operations
```bash
# Recent signals (last 24h)
sqlite3 tradingai.db "
SELECT timestamp, symbol, direction, confidence
FROM signals
WHERE timestamp > datetime('now', '-24 hours')
ORDER BY timestamp DESC LIMIT 20;"

# Paper trading performance
sqlite3 tradingai.db "
SELECT
  COUNT(*) as total,
  SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
  ROUND(AVG(pnl_percent), 2) as avg_pnl
FROM signal_results;"

# Signal distribution
sqlite3 tradingai.db "
SELECT direction, COUNT(*), AVG(confidence)
FROM signals
WHERE timestamp > datetime('now', '-7 days')
GROUP BY direction;"
```

---

## 🏗️ Architecture

### Project Structure
```
crpbot/
├── apps/
│   ├── runtime/              # V7 runtime + legacy V6
│   │   ├── v7_runtime.py     # ⭐ Main V7 orchestrator (production)
│   │   ├── v7_telegram_bot_runner.py
│   │   └── runtime_features.py (legacy V6)
│   ├── trainer/              # Model training (AWS GPU only)
│   ├── dashboard/            # Flask dashboard (deprecated)
│   └── dashboard_reflex/     # ⭐ Reflex dashboard (production)
├── libs/
│   ├── llm/                  # ⭐ V7 LLM integration (4 files)
│   │   ├── signal_generator.py    # Main orchestrator
│   │   ├── deepseek_client.py     # API client
│   │   ├── signal_synthesizer.py  # Theory → Prompt
│   │   └── signal_parser.py       # LLM → Signal
│   ├── analysis/             # ⭐ Core 6 theories
│   │   ├── shannon_entropy.py
│   │   ├── hurst_exponent.py
│   │   ├── markov_chain.py
│   │   ├── kalman_filter.py
│   │   ├── bayesian_inference.py
│   │   └── monte_carlo.py
│   ├── theories/             # ⭐ Statistical 4 theories + context
│   │   ├── random_forest_validator.py
│   │   ├── autocorrelation_analyzer.py
│   │   ├── stationarity_test.py
│   │   ├── variance_tests.py
│   │   └── market_context.py
│   ├── tracking/             # Performance tracking
│   │   ├── performance_tracker.py
│   │   └── paper_trader.py
│   ├── data/                 # Data clients
│   │   ├── coinbase_client.py
│   │   └── coingecko_client.py
│   ├── db/                   # SQLAlchemy models
│   ├── config/               # Pydantic settings
│   └── utils/                # Helpers
├── models/                   # Model weights (.pt files)
├── data/                     # Training data (parquet files)
└── tests/                    # Tests
```

### Data Flow (V7 Ultimate)
```
Coinbase API → OHLCV Data (200+ candles)
                     ↓
            11 Theories Analysis
                     ↓
         Theory Results → LLM Prompt
                     ↓
              DeepSeek API
                     ↓
         Parse LLM Response → Signal
                     ↓
              FTMO Validation
                     ↓
         Rate Limit Check (3/hour)
                     ↓
            Store in SQLite DB
                     ↓
    Paper Trading + Performance Tracking
                     ↓
          Telegram + Dashboard
```

---

## 🔑 Critical Concepts

### Database Architecture

**Production**: SQLite (local file)
- Location: `/root/crpbot/tradingai.db` (cloud server)
- Tables: `signals`, `signal_results`, `theory_performance`
- **NOT using RDS** (RDS stopped 2025-11-22, saves $49/month)

**Configuration** (`.env`):
```bash
DB_URL=sqlite:///tradingai.db  # ✅ Active
# PostgreSQL option commented out (not used)
```

### V7 Signal Generation

**SignalGenerator Flow** (`libs/llm/signal_generator.py`):
1. **Initialize** - Load all 11 theory analyzers
2. **Analyze** - Run each theory on market data
3. **Synthesize** - Convert theory results to LLM prompt
4. **Generate** - Call DeepSeek API
5. **Parse** - Extract structured signal from LLM response
6. **Validate** - Check FTMO rules, rate limits
7. **Return** - SignalGenerationResult with full metadata

**Theory Integration**:
- Each theory returns analysis dict with numeric scores
- SignalSynthesizer formats into natural language prompt
- DeepSeek LLM receives 6-theory summary + market context
- Parser extracts: direction, confidence, entry/SL/TP, reasoning

### A/B Testing

Two variants running concurrently:
- **v7_deepseek_only**: Pure LLM signals (69.2% avg confidence)
- **v7_full_math**: Math-heavy signals (47.2% avg confidence)

Distribution: Random 50/50 split per signal
Tracking: `signal_variant` column in database

### Cost Controls

**Budget Enforcement** (`apps/runtime/v7_runtime.py`):
- Daily: $5 max (DeepSeek API)
- Monthly: $150 max
- Per signal: ~$0.0003-0.0005
- Current usage: $0.19/$150 (0.13%)

**Rate Limiting**:
- Max: 3 signals/hour (conservative mode)
- Sliding window: Last 60 minutes
- Prevents DeepSeek API overuse

### FTMO Risk Management

**Hard Limits** (`apps/runtime/ftmo_rules.py`):
- Daily loss: 4.5% max (enforced before each signal)
- Total loss: 9% max (account termination)
- Position sizing: Risk-based (1-2% per trade)

**Kill Switch**:
```bash
# Emergency stop
export KILL_SWITCH=true  # In .env
```

### Paper Trading

**Automated Tracking** (`libs/tracking/paper_trader.py`):
- Detects entry when price hits signal entry ±0.5%
- Monitors for SL/TP hit
- Records outcome: win/loss
- Updates Bayesian win rate
- Stores in `signal_results` table

**Current Status**:
- 13 paper trades completed
- 53.8% win rate
- +5.48% total P&L
- Need 20+ for statistical significance

---

## 🧪 Testing

### Test Structure
```bash
tests/
├── unit/                     # Unit tests
│   ├── test_ftmo_rules.py
│   ├── test_confidence_scoring.py
│   └── test_rate_limiter.py
├── integration/              # Integration tests
│   └── test_runtime_guardrails.py
└── smoke/                    # 5-min backtest
    └── test_end_to_end.py
```

### Running Tests
```bash
make test         # All tests
make unit         # Unit only
make smoke        # Smoke only
pytest tests/unit/test_ftmo_rules.py -v  # Specific test
pytest tests/ --cov=apps --cov=libs      # With coverage
```

---

## ☁️ AWS Infrastructure

### Current State (2025-11-22)

**Active**:
- S3 buckets (~$1-5/month):
  - `crpbot-ml-data-20251110` - Training data
  - `sagemaker-us-east-1-*` - SageMaker artifacts

**Stopped/Deleted** (Cost cleanup):
- ✅ RDS `crpbot-rds-postgres-db` - STOPPED (saves $35/month)
- ✅ RDS `crpbot-dev` - STOPPED (saves $14/month)
- ✅ Redis `crpbot-redis-dev` - DELETED (saves $12/month)
- ✅ Redis `crp-re-wymqmkzvh0gm` - DELETED (saves $12/month)
- **Total savings**: $61/month (bill: $140 → $79/month)

**GPU Training** (On-demand only):
- Instance: g4dn.xlarge (NVIDIA T4)
- Cost: $0.16/run (spot), $0.53 (on-demand)
- Duration: 10-15 min per model
- **CRITICAL**: Always terminate after training

### Training Workflow

**NEVER train locally** - Follow `MASTER_TRAINING_WORKFLOW.md`:

1. **Feature Engineering** (local):
```bash
# Engineer features for all symbols
python scripts/engineer_features.py
```

2. **Upload to S3**:
```bash
aws s3 sync data/features/ s3://crpbot-ml-data-20251110/features/
```

3. **Launch GPU Instance**:
```bash
aws ec2 run-instances --instance-type g4dn.xlarge --spot-instance
```

4. **Train on GPU**:
```bash
# On GPU instance
uv run python apps/trainer/main.py --task lstm --coin BTC --epochs 15
uv run python apps/trainer/main.py --task lstm --coin ETH --epochs 15
uv run python apps/trainer/main.py --task lstm --coin SOL --epochs 15
```

5. **Upload Models**:
```bash
aws s3 sync models/ s3://crpbot-ml-data-20251110/models/
```

6. **TERMINATE INSTANCE** (CRITICAL):
```bash
aws ec2 terminate-instances --instance-ids <INSTANCE_ID>
```

7. **Download and Deploy**:
```bash
# QC Claude (local)
aws s3 sync s3://crpbot-ml-data-20251110/models/ models/

# Builder Claude (cloud)
aws s3 sync s3://crpbot-ml-data-20251110/models/ models/promoted/
```

---

## 📝 Configuration

### Environment Variables (`.env`)

**Required**:
```bash
# Data Provider
DATA_PROVIDER=coinbase
COINBASE_API_KEY_NAME=organizations/.../apiKeys/...
COINBASE_API_PRIVATE_KEY=-----BEGIN EC PRIVATE KEY-----...

# Premium APIs
COINGECKO_API_KEY=CG-VQhq64e59sGxchtK8mRgdxXW
DEEPSEEK_API_KEY=sk-...

# Database (SQLite)
DB_URL=sqlite:///tradingai.db

# Safety
KILL_SWITCH=false
CONFIDENCE_THRESHOLD=0.65
MAX_SIGNALS_PER_HOUR=3
```

**Optional**:
```bash
# Telegram (notifications)
TELEGRAM_TOKEN=...
TELEGRAM_CHAT_ID=...

# FTMO (if trading live)
FTMO_LOGIN=...
FTMO_PASS=...
FTMO_SERVER=...
```

---

## 🚨 Common Pitfalls

### 1. Training Locally
**Symptom**: Training takes 60-90 min
**Fix**: Use AWS g4dn.xlarge GPU only

### 2. AWS Instance Not Terminated
**Symptom**: Unexpected $40+ charges
**Fix**: ALWAYS terminate: `aws ec2 terminate-instances --instance-ids <id>`

### 3. RDS Connection Errors
**Issue**: V7 does NOT use RDS (uses local SQLite)
**If you see**: PostgreSQL errors → Check `.env` for `DB_URL=sqlite:///tradingai.db`

### 4. Feature Count Mismatch
**Symptom**: Models predict ~50% (random)
**Fix**: Verify training features = runtime features = model input_size

### 5. V7 Stopped Running
**Check**:
```bash
ps aux | grep v7_runtime | grep -v grep  # Should show 1 process
tail -100 /tmp/v7_runtime_*.log          # Check for errors
```

### 6. Low Confidence Signals
**Expected**: V7 is conservative (high HOLD rate = intentional)
**Not a bug**: 76% HOLD signals is normal during ranging markets

---

## 📚 Key Documentation

### Production (V7)
- `CURRENT_STATUS_AND_NEXT_ACTIONS.md` - **Current phase & tasks** ⭐
- `apps/runtime/v7_runtime.py` - Main runtime source
- `QUANT_FINANCE_10_HOUR_PLAN.md` - Phase 1 enhancements (pending)
- `QUANT_FINANCE_PHASE_2_PLAN.md` - Phase 2 advanced features (future)

### Infrastructure
- `MASTER_TRAINING_WORKFLOW.md` - **GPU training workflow** ⭐
- `DATABASE_VERIFICATION_2025-11-22.md` - Database setup verification
- `AWS_COST_CLEANUP_2025-11-22.md` - AWS cost optimization
- `CLAUDE.md` - This file

### Reference
- `README.md` - Project overview
- `Makefile` - Available commands
- `.env` - Configuration (DO NOT commit)

---

## 🔮 Current Status (November 2025)

**Branch**: `feature/v7-ultimate`
**Phase**: Monitoring & Data Collection (2025-11-22 to 2025-11-25)

**V7 Ultimate**:
- ✅ All 11 theories operational (100%)
- ✅ DeepSeek LLM integration working
- ✅ Paper trading active (13 trades, 53.8% win rate)
- ✅ A/B testing running (2 variants)
- ✅ Dashboard live (http://178.156.136.185:3000)
- ⏳ Collecting data (need 20+ trades for Sharpe ratio)

**Next Actions**:
- **Builder Claude**: Daily monitoring (5-10 min/day)
- **QC Claude**: Await Monday 2025-11-25 review
- **Decision**: Based on Sharpe ratio, decide on Phase 1 enhancements

**Production Environment**:
- Runtime: Cloud server (178.156.136.185)
- Database: SQLite (local file, 4,075 signals)
- Cost: $0.19/$150 DeepSeek budget (0.13% used)
- AWS: $79/month (down from $140 after cleanup)

---

**Last Updated**: 2025-11-22
**Next Review**: 2025-11-25 (Monday - Sharpe ratio decision)
