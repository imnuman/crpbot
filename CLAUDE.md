# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🚨 CRITICAL: Session Start Protocol

**ALWAYS DO FIRST** when starting a new session:
1. Read `PROJECT_MEMORY.md` for session continuity and current context
2. Check git status: `git fetch && git log -5 --oneline`
3. Identify your environment (local vs cloud) and role

## 🔴 CRITICAL UPDATE: V4 → V5 PIVOT (November 15, 2025)

**MAJOR STRATEGIC DECISION**:
- ❌ **V4 is OBSOLETE**: Models stuck at 50% accuracy due to noisy free Coinbase data
- ✅ **V5 Strategy**: Upgrade to Tardis.dev professional market data
- ✅ **This is an UPGRADE (10% change)**, not a rebuild - 90% of code stays

**What's Changing**:
1. **Data Source**: Free Coinbase OHLCV → Tardis.dev tick data + order book
2. **Features**: 31-50 features → 53 features (adding 20 microstructure features)
3. **Budget**: $197/month Phase 1, $549/month Phase 2 (approved)

**What's NOT Changing** (90% reuse):
- ✅ Model architecture (LSTM, Transformer, RL)
- ✅ Training pipeline
- ✅ Runtime system
- ✅ FTMO compliance rules
- ✅ Ensemble logic
- ✅ Confidence calibration

**Current Status**:
- 🔴 **BLOCKED**: Waiting for Tardis.dev subscription ($147/month)
- 📋 **Next**: 4-week validation timeline (Week 1: Data, Week 2: Features, Week 3: Training, Week 4: Validation)
- 🎯 **Target**: 65-75% accuracy (vs V4's 50% ceiling)

**All V4 work (Colab notebooks, 50-feature models) is now OBSOLETE.**

## 🎯 Dual Environment Setup

We operate in a **dual-environment setup**:
- **Local Machine** (`/home/numan/crpbot`): Local Claude for QC review & testing
- **Cloud Server** (`/root/crpbot`): Cloud Claude for development & training

Both environments sync via Git (GitHub: `imnuman/crpbot`). See `PROJECT_MEMORY.md` for detailed role definitions.

---

## Project Overview

**CRPBot** is a cryptocurrency trading AI system designed for FTMO challenge compliance. It uses an ensemble of machine learning models (LSTM + Transformer + RL) to generate high-confidence trading signals with strict risk management for BTC-USD, ETH-USD, and SOL-USD.

### Directory Structure

```
crpbot/
├── apps/
│   ├── trainer/          # Model training (LSTM, Transformer, RL)
│   │   ├── models/       # Model architecture definitions
│   │   ├── train/        # Training logic per model type
│   │   ├── eval/         # Evaluation and promotion gates
│   │   ├── features.py   # Feature engineering functions
│   │   └── main.py       # Training entry point
│   ├── runtime/          # Production runtime loop
│   │   ├── main.py       # Runtime entry point
│   │   ├── confidence.py # Confidence calibration
│   │   ├── ftmo_rules.py # FTMO compliance checks
│   │   └── rate_limiter.py # Signal rate limiting
│   ├── mt5_bridge/       # MT5/FTMO broker integration
│   └── kafka/            # Data streaming (optional)
├── libs/
│   ├── config/           # Configuration and settings
│   ├── constants/        # FTMO limits, feature lists
│   ├── data/             # Data loading utilities
│   ├── db/               # Database models and queries
│   ├── confidence/       # Calibration algorithms
│   ├── rl_env/           # RL environment (PPO)
│   └── aws/              # AWS S3 integration
├── scripts/
│   ├── fetch_data.py     # Coinbase API data fetching
│   ├── engineer_features.py # Feature engineering
│   ├── evaluate_model.py # Model evaluation & promotion
│   └── infrastructure/   # Deployment scripts
├── data/
│   ├── raw/              # Raw OHLCV parquet files
│   └── features/         # Engineered features parquet
├── models/
│   ├── promoted/         # Production-ready models
│   ├── gpu_trained/      # Models from GPU training
│   └── new/              # Latest training outputs
├── tests/
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── smoke/            # 5-min backtest smoke tests
└── infra/
    ├── aws/              # AWS infrastructure (Terraform/CDK)
    ├── docker/           # Docker configurations
    └── systemd/          # VPS service units
```

## Core Architecture

### Three-Layer Model Ensemble

1. **Per-Coin LSTM Models** (35% weight)
   - Binary direction prediction (up/down) for 15-minute horizon
   - 60-minute lookback window, 2-layer bidirectional LSTM
   - Separate models trained for each symbol
   - Model files: `lstm_{SYMBOL}_1m_*.pt`

2. **Multi-Coin Transformer** (40% weight)
   - Trend strength prediction (0-1 continuous)
   - 100-minute lookback, 4-layer transformer encoder with 8-head attention
   - Learns cross-asset patterns across all symbols
   - Model files: `transformer_multi_*.pt`

3. **RL Agent (PPO)** (25% weight)
   - Execution optimization considering spreads/slippage
   - **Status**: Stub implementation, needs completion

### Data Pipeline Flow

```
Coinbase API → Raw OHLCV (1m candles, parquet)
  ↓
Feature Engineering (50+ indicators: ATR, RSI, MACD, sessions, volatility regime)
  ↓
Walk-Forward Splits (70% train, 15% val, 15% test)
  ↓
Model Training (LSTM per-coin + Transformer multi-coin)
  ↓
Evaluation & Promotion (68% accuracy, 5% calibration error gates)
  ↓
Runtime Signal Generation (ensemble → confidence calibration → FTMO rules → rate limiting)
```

### Key Safety Mechanisms

- **FTMO Compliance**: Daily 5% / Total 10% loss limits enforced in real-time
- **Rate Limiting**: Max 10 signals/hour (5 high-confidence/hour)
- **Kill Switch**: `KILL_SWITCH=true` environment variable for emergency stops
- **Position Sizing**: 1% risk per trade with validation
- **Audit Trail**: All signals logged to database with full context

## Common Commands

### Setup & Dependencies
```bash
make setup              # First-time setup: install deps + pre-commit hooks
make sync               # Sync dependencies with uv
uv pip install -e .     # Install package in editable mode
```

### Code Quality
```bash
make fmt                # Auto-format with ruff
make lint               # Run ruff + mypy type checking
make test               # Run all tests (unit + integration + smoke)
make unit               # Run unit tests only
make smoke              # Run 5-minute backtest smoke test
```

### Data Pipeline
```bash
# Fetch historical data (2 years recommended for training)
uv run python scripts/fetch_data.py \
    --symbol BTC-USD \
    --interval 1m \
    --start 2023-11-10 \
    --output data/raw

# Engineer features (39 columns: 5 OHLCV + 31 features + 3 categorical)
uv run python scripts/engineer_features.py \
    --input data/raw/BTC-USD_1m_2023-11-10_2025-11-10.parquet \
    --symbol BTC-USD \
    --interval 1m

# Batch feature engineering for all symbols
./batch_engineer_features.sh

# Validate data quality
uv run python scripts/validate_data_quality.py --symbol BTC-USD
```

### Model Training

```bash
# Train LSTM for specific coin (15-20 epochs recommended)
make train COIN=BTC EPOCHS=15
# Or directly:
uv run python apps/trainer/main.py --task lstm --coin BTC --epochs 15

# Train Transformer (multi-coin, runs after all LSTM models)
uv run python apps/trainer/main.py --task transformer --epochs=15

# Train RL Agent (stub - needs implementation)
make rl STEPS=8000000

# Batch training for all symbols
./batch_engineer_features.sh  # Feature engineering
./batch_engineer_50_features.sh  # Extended features (if using 50-feature set)
```

**Training Approaches**:
- **Local/Cloud CPU**: Use `apps/trainer/main.py` directly (slower)
- **Google Colab GPU**: Upload data/scripts to Colab for faster training (~57 min for 3 models)
  - See `prepare_colab_files.sh` to prepare files for upload
  - Notebooks: `colab_*.ipynb` for various training/evaluation tasks

### Model Evaluation & Promotion
```bash
# Evaluate model against promotion gates
uv run python scripts/evaluate_model.py \
    --model models/lstm_BTC_USD_1m_*.pt \
    --symbol BTC-USD \
    --model-type lstm \
    --min-accuracy 0.68 \
    --max-calibration-error 0.05

# Promoted models → models/promoted/ directory
```

### Runtime Execution
```bash
# Dry-run mode (recommended for testing)
make run-dry
uv run python apps/runtime/main.py --mode dryrun --iterations -1 --sleep-seconds 120

# Live production mode
make run-bot
uv run python apps/runtime/main.py --mode live --iterations -1
```

### Single Test Execution
```bash
# Run specific test file
pytest tests/unit/test_ensemble.py -v

# Run specific test by name
pytest -k "test_ensemble_prediction" -v

# Run with coverage
pytest --cov=apps --cov=libs tests/
```

## Critical Configuration

### Environment Variables (.env)

**Required for Data Fetching:**
```bash
DATA_PROVIDER=coinbase
COINBASE_API_KEY=organizations/*/apiKeys/*
COINBASE_PRIVATE_KEY=-----BEGIN EC PRIVATE KEY-----\n...\n-----END EC PRIVATE KEY-----
```

**Required for Trading:**
```bash
CONFIDENCE_THRESHOLD=0.75       # Minimum signal confidence
ENSEMBLE_WEIGHTS=0.35,0.40,0.25 # LSTM, Transformer, RL weights
KILL_SWITCH=false               # Emergency stop switch
MAX_SIGNALS_PER_HOUR=10
MAX_SIGNALS_PER_HOUR_HIGH=5
RUNTIME_MODE=dryrun             # Use 'dryrun' for testing, 'live' for production
```

**Database:**
```bash
DB_URL=sqlite:///tradingai.db
# Or PostgreSQL for production:
# DB_URL=postgresql+psycopg://user:pass@localhost:5432/tradingai
```

### Model File Naming Convention

- LSTM models: `lstm_{SYMBOL}_1m_{hash}.pt` (e.g., `lstm_BTC_USD_1m_a7aff5c4.pt`)
- Transformer: `transformer_multi_v{timestamp}.pt`
- Promoted models: `{model_type}_{symbol}_promoted.pt` in `models/promoted/`

## Architecture-Specific Notes

### Feature Engineering (31 Features)

- **Session Features** (5): Tokyo/London/NY sessions, day_of_week, is_weekend
- **Spread Features** (4): spread, spread_pct, ATR, spread_atr_ratio
- **Volume Features** (3): volume_ma, volume_ratio, volume_trend
- **Moving Averages** (8): SMA 7/14/21/50 + price ratios
- **Technical Indicators** (8): RSI, MACD×3, Bollinger Bands×4
- **Volatility Regime** (3): low/medium/high classification (one-hot)

Features are stored in parquet format: `data/features/features_{SYMBOL}_1m_latest.parquet`

### Multi-Timeframe Features (Phase 3.5 / V2)

**Module**: `apps/trainer/multi_tf_features.py`

**Capabilities**:
- Load data from multiple timeframes (1m, 5m, 15m, 1h)
- Resample higher TF data to align with base TF (1m)
- Cross-TF alignment scoring (measures trend agreement across timeframes)
- Volatility regime classification using ATR percentiles

**Usage**:
```bash
# Fetch multi-TF data
./scripts/fetch_multi_tf_data.sh

# Engineer multi-TF features
uv run python scripts/engineer_multi_tf_features.py --symbol BTC-USD

# Test multi-TF functionality
uv run python scripts/test_multi_tf.py
```

**New Features Added**:
- Cross-TF alignment score (0-1, higher = stronger trend agreement)
- Cross-TF alignment direction (-1, 0, 1)
- Cross-TF alignment strength (0-1)
- Multi-TF OHLCV features (5m_, 15m_, 1h_ prefixes)

### Walk-Forward Splitting

Time-aware splits to prevent data leakage:
- **Train**: 70% (earliest data)
- **Validation**: 15% (middle period)
- **Test**: 15% (most recent data)

Split points logged during training, e.g.:
```
Train until: 2025-03-24 22:38:42+00:00
Val until:   2025-07-10 06:55:21+00:00
Test:        2025-07-10 06:55:21+00:00 onwards
```

### LSTM Model Architecture Details

```python
# Per-coin LSTM structure (example for BTC-USD)
Input: (batch_size, 60, 31)  # 60 timesteps, 31 features
LSTM Layer 1: bidirectional, hidden_size=64 → output (batch, 60, 128)
LSTM Layer 2: bidirectional, hidden_size=64 → output (batch, 60, 128)
Dropout: 0.2
FC Layers: 128 → 64 → 1
Sigmoid: Output probability [0, 1]

# Training params
- Batch size: 32
- Optimizer: Adam
- Loss: Binary Cross Entropy
- Early stopping: patience=5 epochs
- Total params: ~62,337
```

### Transformer Model Architecture Details

```python
# Multi-coin Transformer structure
Input: (batch_size, 100, 31)  # 100 timesteps, 31 features
Input Projection: 31 → 128 (d_model)
Positional Encoding: Sinusoidal
Transformer Encoder:
  - 4 layers
  - 8 attention heads (d_k = d_model // num_heads = 16)
  - FFN dim: 512
  - Dropout: 0.1
Output: 128 → 512 → 1
Sigmoid: Trend strength [0, 1]

# Training params
- Batch size: 16
- Optimizer: AdamW with warm-up + cosine decay
- Loss: MSE
- Gradient clipping: max_norm=1.0
```

### Runtime Signal Generation Logic

The runtime system (`apps/runtime/main.py`) generates trading signals through this flow:

```python
# Ensemble prediction flow
1. Load promoted models (LSTM per-coin + Transformer)
   - LSTM: models/promoted/lstm_{SYMBOL}_promoted.pt
   - Transformer: models/promoted/transformer_multi_promoted.pt
   - RL: Stub implementation (25% weight currently assigned to neutral)

2. Fetch latest market data and generate features
   - Uses same feature engineering as training
   - Maintains feature history (60+ timesteps)

3. Get predictions from each model:
   - lstm_pred = lstm_model(features[-60:])  # Last 60 minutes
   - trans_pred = transformer_model(features[-100:])  # Last 100 minutes
   - rl_pred = 0.5  # Stub: neutral prediction

4. Combine with ensemble weights:
   ensemble = lstm*0.35 + trans*0.40 + rl*0.25
   (Configurable via ENSEMBLE_WEIGHTS env var)

5. Apply confidence calibration (Platt scaling)
   - Calibrates raw ensemble output to true probability
   - Learned from validation set during training

6. Classify confidence tier:
   - High: ≥75% confidence (max 5/hour)
   - Medium: ≥65% confidence (max 10/hour total)
   - Low: ≥55% confidence (max 10/hour total)

7. Check FTMO compliance rules:
   - Daily loss limit: 5%
   - Total loss limit: 10%
   - Position size: 1% risk per trade

8. Check rate limits:
   - Hour-based rolling window
   - Separate caps for high-confidence signals

9. Log to database + emit signal
   - Database: Full context (features, predictions, confidence)
   - Telegram: Signal notification (planned)
```

**Key Configuration** (`.env`):
```bash
CONFIDENCE_THRESHOLD=0.75       # Minimum confidence for signals
ENSEMBLE_WEIGHTS=0.35,0.40,0.25 # LSTM, Transformer, RL
MAX_SIGNALS_PER_HOUR=10         # Total signals/hour
MAX_SIGNALS_PER_HOUR_HIGH=5     # High-confidence signals/hour
```

### Promotion Gates (Phase 3)

Models must pass these thresholds to be promoted to production:

- **Win Rate**: ≥68% on test set
- **Calibration Error**: ≤5% (Brier score or ECE)
- **Backtest Sharpe**: >1.0 (optional gate)
- **Max Drawdown**: <15% (optional gate)

Promotion command automatically checks gates and copies to `models/promoted/`.

## Important Development Patterns

### Understanding the Data Flow

```
1. Data Fetching (scripts/fetch_data.py)
   ↓ Writes to: data/raw/{SYMBOL}_1m_{start}_{end}.parquet

2. Feature Engineering (scripts/engineer_features.py)
   ↓ Reads from: data/raw/*.parquet
   ↓ Writes to: data/features/features_{SYMBOL}_1m_latest.parquet

3. Training (apps/trainer/main.py)
   ↓ Reads from: data/features/*.parquet
   ↓ Creates train/val/test splits (70/15/15)
   ↓ Writes to: models/{model_type}_{SYMBOL}_{hash}.pt

4. Evaluation (scripts/evaluate_model.py)
   ↓ Reads from: models/*.pt
   ↓ Checks promotion gates (68% accuracy, 5% calibration)
   ↓ Writes to: models/promoted/*.pt

5. Runtime (apps/runtime/main.py)
   ↓ Loads from: models/promoted/*.pt
   ↓ Generates signals with ensemble
   ↓ Logs to: database (tradingai.db)
```

### Adding New Features

1. Update `apps/trainer/features.py` with new indicator calculation
2. Modify `FEATURE_COLUMNS` list in `libs/constants/` to include new feature
3. Update feature engineering script to compute new indicator
4. Re-run feature engineering: `./batch_engineer_features.sh`
5. Re-train models with new features
6. Update documentation in `FEATURE_ENGINEERING_WORKFLOW.md`

**CRITICAL**: Feature count must match between training and runtime!

### Adding New Model Type

1. Create model class in `apps/trainer/models/` (inherit from `nn.Module`)
2. Add training logic in `apps/trainer/train/train_*.py`
3. Update `apps/trainer/main.py` to support new `--task` option
4. Add evaluation logic in `scripts/evaluate_model.py`
5. Update ensemble weights in `apps/runtime/ensemble.py`
6. Update `ENSEMBLE_WEIGHTS` configuration

### Debugging Training Issues

```bash
# Check data quality
uv run python scripts/validate_data_quality.py --symbol BTC-USD

# Inspect feature distributions
uv run python -c "import pandas as pd; df = pd.read_parquet('data/features/features_BTC-USD_1m_latest.parquet'); print(df.describe())"

# Monitor training with verbose logging
LOG_LEVEL=DEBUG uv run python apps/trainer/main.py --task lstm --coin BTC --epochs 5

# Check model output shapes
pytest tests/unit/test_models.py -v -s
```

### Modifying FTMO Rules

FTMO rules are centralized in `libs/constants/ftmo.py`:

```python
FTMO_DAILY_LOSS_LIMIT = 0.05  # 5% daily loss
FTMO_TOTAL_LOSS_LIMIT = 0.10  # 10% total loss
FTMO_MIN_TRADING_DAYS = 4     # Minimum trading days
FTMO_PROFIT_TARGET = 0.10     # 10% profit target
```

Changes require updating tests in `tests/unit/test_ftmo_rules.py`.

## Common Issues & Troubleshooting

### Feature Mismatch Errors

**Error**: "Expected X features but got Y"

**Cause**: Model trained with different feature set than runtime

**Solution**:
1. Check feature count: `python -c "import pandas as pd; df = pd.read_parquet('data/features/features_BTC-USD_1m_latest.parquet'); print(f'Features: {len(df.columns)}')"`
2. Verify model expects same count
3. Either retrain model or regenerate features to match

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'apps'`

**Solution**: Install in editable mode: `uv pip install -e .`

### Data Loading Errors

**Error**: "File not found" when loading parquet

**Solution**:
1. Check file exists: `ls -lh data/raw/` or `ls -lh data/features/`
2. Verify file naming: `{SYMBOL}_1m_{start}_{end}.parquet` or `features_{SYMBOL}_1m_latest.parquet`
3. Check symlinks are correct: `ls -l data/features/features_*_latest.parquet`

### Model Training Hangs

**Issue**: Training appears stuck

**Debug**:
1. Add verbose logging: `LOG_LEVEL=DEBUG uv run python apps/trainer/main.py ...`
2. Check data loading: Verify parquet files not corrupted
3. Reduce batch size if OOM: Edit training script batch_size parameter
4. For GPU training: Use Google Colab instead of CPU

### Git Sync Issues

**Issue**: Conflicts or out of sync with GitHub

**Solution**:
1. Check status: `git status`
2. Fetch latest: `git fetch origin`
3. See divergence: `git log origin/main..HEAD` and `git log HEAD..origin/main`
4. If safe, pull: `git pull origin main`
5. If conflicts, see `GIT_SYNC_PROTOCOL.md`

## Testing Strategy

- **Unit Tests** (`tests/unit/`): Test individual components (models, features, FTMO rules)
- **Integration Tests** (`tests/integration/`): Test data pipeline, ensemble, database
- **Smoke Tests** (`tests/smoke/`): 5-minute backtest to validate end-to-end flow

Run all: `make test` (should complete in <2 minutes)

## Current Project Status (V5 - November 15, 2025)

**🚨 MAJOR PIVOT**: V4 → V5 Data Upgrade Strategy

**Check these files for latest status**:
- `PROJECT_MEMORY.md` - Current context and V5 pivot details
- `V5_PHASE1_PLAN.md` - V5 Phase 1 roadmap (to be created)
- `git log -5 --oneline` - Recent changes

**V5 Component Status**:
- **V4 Models**: ❌ OBSOLETE (50% accuracy ceiling - free data too noisy)
- **V5 Data Source**: 🔴 BLOCKED (waiting for Tardis.dev subscription - $147/month)
- **Architecture (90% reuse)**: ✅ Ready (LSTM, Transformer, RL, Ensemble, Runtime, FTMO)
- **V5 Features (53 total)**: 🟡 Pending (33 existing + 20 microstructure to be engineered)
- **V5 Pipeline**: 🔴 Not Started (Week 1: Data download, Week 2: Features, Week 3: Training, Week 4: Validation)

**V5 Budget Approved**:
- **Phase 1 (Validation)**: $197/month for 4 weeks
  - Tardis Historical: $147/month ✅
  - Coinbase real-time: Free (existing) ✅
  - AWS: ~$50/month ✅
- **Phase 2 (Live Trading)**: $549/month (only if Phase 1 succeeds)
  - Tardis Premium: $499/month
  - AWS: ~$50/month

**V5 Timeline (4 weeks)**:
- Week 1: Download Tardis data (tick + order book, 2+ years)
- Week 2: Engineer 53 features (33 existing + 20 microstructure)
- Week 3: Train models with professional data
- Week 4: Validate (target: 65-75% accuracy)

**Immediate Next Action**:
🚀 Subscribe to Tardis.dev Historical at https://tardis.dev/pricing

**Known Issues from V4** (now OBSOLETE):
- V4 models stuck at 50% accuracy - ROOT CAUSE: Free Coinbase data too noisy
- All V4 Colab work is obsolete (Nov 13-14 notebooks, 50-feature models)
- RL agent is stub implementation only (will continue in V5)

## Key File Locations

**Configuration**:
- `.env` - Environment variables (credentials, thresholds)
- `libs/config/settings.py` - Configuration loader
- `libs/constants/ftmo.py` - FTMO limits and constants

**Data**:
- `data/raw/*.parquet` - Raw OHLCV data from Coinbase
- `data/features/features_{SYMBOL}_1m_latest.parquet` - Engineered features

**Models**:
- `models/promoted/` - Production-ready models
- `models/gpu_trained/` - Models from Colab/GPU training
- `models/new/` - Latest training outputs

**Core Applications**:
- `apps/trainer/main.py` - Training entry point
- `apps/trainer/features.py` - Feature engineering logic
- `apps/trainer/models/` - Model architectures (LSTM, Transformer)
- `apps/trainer/train/` - Training loops per model type
- `apps/trainer/eval/` - Evaluation and promotion gates
- `apps/runtime/main.py` - Production runtime loop

**Scripts**:
- `scripts/fetch_data.py` - Fetch OHLCV from Coinbase
- `scripts/engineer_features.py` - Feature engineering
- `scripts/evaluate_model.py` - Model evaluation & promotion
- `batch_engineer_features.sh` - Batch feature engineering
- `prepare_colab_files.sh` - Prepare files for Colab upload

**Documentation & Status**:
- `PROJECT_MEMORY.md` - Session continuity and context
- `PHASE6_5_RESTART_PLAN.md` - Current phase tracking
- `reports/phase6_5/` - Latest reports and findings
- `COLAB_*.md` - Colab integration guides

**Tests**:
- `tests/unit/` - Unit tests
- `tests/integration/` - Integration tests
- `tests/smoke/` - 5-min backtest smoke tests

## Package Management with UV

This project uses **uv** (ultra-fast Python package manager):

```bash
# Add new dependency
uv add package-name

# Add dev dependency
uv add --dev package-name

# Remove dependency
uv remove package-name

# Update all dependencies
uv lock --upgrade
make sync
```

Dependencies are defined in `pyproject.toml` and locked in `uv.lock`.
