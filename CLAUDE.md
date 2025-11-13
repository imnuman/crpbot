# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🚨 IMPORTANT: Dual Environment Setup

**READ FIRST**: We operate in a **dual-environment setup**:
- **Local Machine** (`/home/numan/crpbot`): Local Claude for QC review & testing
- **Cloud Server** (`~/crpbot`): Cloud Claude for development & training

**Before proceeding, read**: `PROJECT_MEMORY.md` for session continuity and role definitions.

Both environments sync via Git (GitHub: `imnuman/crpbot`). Always check `git log` and `PROJECT_MEMORY.md` when starting a new chat to understand the current context.

---

## Project Overview

**CRPBot** is a cryptocurrency trading AI system designed for FTMO challenge compliance. It uses an ensemble of machine learning models (LSTM + Transformer + RL) to generate high-confidence trading signals with strict risk management for BTC-USD, ETH-USD, and SOL-USD.

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
```

**Current Training Status** (as of Phase 6.5 Restart):
- ✅ **LSTM Models**: All 3 models trained (BTC, ETH, SOL)
  - Model files: `models/lstm_{SYMBOL}_USD_1m_a7aff5c4.pt`
  - Training completed: 2025-11-10
  - Ready for evaluation against promotion gates
- ⏹️ **Transformer Model**: Queued for training (~40 min estimated)
- 🔄 **Multi-TF Features**: Module created, data fetching in progress

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

```python
# Ensemble prediction flow
1. Load promoted models (LSTM per-coin + Transformer)
2. Generate features from latest market data
3. Get predictions from each model:
   - lstm_pred = lstm_model(features[-60:])  # Last 60 minutes
   - trans_pred = transformer_model(features[-100:])  # Last 100 minutes
   - rl_pred = rl_agent(features, portfolio_state)  # Stub
4. Combine with weights:
   ensemble = lstm*0.35 + trans*0.40 + rl*0.25
5. Apply confidence calibration (Platt scaling)
6. Classify tier: High (≥75%), Medium (≥65%), Low (≥55%)
7. Check FTMO rules: daily loss, total loss, position size
8. Check rate limits: hour-based signal caps
9. Log to database + emit signal (Telegram planned)
```

### Promotion Gates (Phase 3)

Models must pass these thresholds to be promoted to production:

- **Win Rate**: ≥68% on test set
- **Calibration Error**: ≤5% (Brier score or ECE)
- **Backtest Sharpe**: >1.0 (optional gate)
- **Max Drawdown**: <15% (optional gate)

Promotion command automatically checks gates and copies to `models/promoted/`.

## Important Development Patterns

### Adding New Features

1. Update `apps/trainer/features.py` with new indicator calculation
2. Modify `FEATURE_COLUMNS` list to include new feature
3. Update feature engineering script to compute new indicator
4. Re-run feature engineering: `./batch_engineer_features.sh`
5. Re-train models with new features
6. Update documentation in `FEATURE_ENGINEERING_WORKFLOW.md`

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

## Testing Strategy

- **Unit Tests** (`tests/unit/`): Test individual components (models, features, FTMO rules)
- **Integration Tests** (`tests/integration/`): Test data pipeline, ensemble, database
- **Smoke Tests** (`tests/smoke/`): 5-minute backtest to validate end-to-end flow

Run all: `make test` (should complete in <2 minutes)

## Current Project Status (Phase 6.5)

- **Data Pipeline**: ✅ Complete (fetch → features → validation)
- **Model Architecture**: ✅ Complete (LSTM + Transformer)
- **Training Pipeline**: ✅ Complete (automated batch training)
- **Runtime Engine**: ✅ Complete (dry-run functional)
- **Model Evaluation**: ✅ Complete (promotion gates implemented)
- **Production Deployment**: 🚧 In Progress (AWS infrastructure 20% complete)

### Training Progress (Phase 6.5 Restart)

**Status**: 🔄 **In Execution** - Steps 1-3 Complete, Step 4 In Progress

**Completed**:
- ✅ **Data Infrastructure**: Coinbase API configured with JWT authentication
- ✅ **Dataset Generation**: 2 years of 1-minute OHLCV data (2023-11-10 to 2025-11-10)
  - BTC-USD: 1,030,512 rows (35 MB)
  - ETH-USD: 1,030,512 rows (32 MB)
  - SOL-USD: 1,030,513 rows (23 MB)
- ✅ **Feature Engineering**: 39 columns (5 OHLCV + 31 features + 3 categorical)
  - All 3 symbols: features files generated and validated
- ✅ **LSTM Training**: All 3 per-coin models trained (15 epochs each)
  - `lstm_BTC_USD_1m_a7aff5c4.pt` (249 KB)
  - `lstm_ETH_USD_1m_a7aff5c4.pt` (249 KB)
  - `lstm_SOL_USD_1m_a7aff5c4.pt` (249 KB)
  - Validation accuracy: ~50% (baseline, ready for evaluation gates)

**In Progress**:
- ⏹️ **Transformer Training**: Multi-coin transformer queued (~40 min estimated)
- 🔄 **Phase 3.5 (Multi-TF)**: Multi-timeframe feature engineering module created
  - Module: `apps/trainer/multi_tf_features.py`
  - Scripts: `scripts/fetch_multi_tf_data.sh`, `scripts/test_multi_tf.py`
  - Supports: 1m, 5m, 15m, 1h timeframes
  - Features: Cross-TF alignment, volatility regime classification

**Next Steps**:
1. Complete Transformer training
2. Evaluate all models against promotion gates (68% accuracy, 5% calibration error)
3. Promote qualifying models to `models/promoted/`
4. Integrate promoted models into runtime
5. Restart Phase 6.5 observation period (3-5 days)
6. Phase 7: Micro-lot testing on FTMO account

**Reference**: See `PHASE6_5_RESTART_PLAN.md` for detailed checklist and progress tracking.

## File Locations Reference

```
Configuration:     .env, libs/config/settings.py
Data:              data/raw/, data/features/
Models:            models/, models/promoted/
Training:          apps/trainer/main.py
  Multi-TF:        apps/trainer/multi_tf_features.py
Runtime:           apps/runtime/main.py
Scripts:           scripts/fetch_data.py, scripts/engineer_features.py
  Multi-TF:        scripts/fetch_multi_tf_data.sh, scripts/test_multi_tf.py
Tests:             tests/unit/, tests/integration/, tests/smoke/
Infrastructure:    infra/aws/, infra/docker/
Documentation:     docs/, PHASE6_5_RESTART_PLAN.md
Reports:           reports/phase6_5/
```

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
