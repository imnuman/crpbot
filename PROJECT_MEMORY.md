# CRPBot Project Memory

## Project Overview
**CRPBot** is an AI-powered cryptocurrency trading bot designed for FTMO-compliant trading with ensemble machine learning models.

### Key Details
- **Location**: `/root/crpbot`
- **Language**: Python 3.10+
- **Current Phase**: Phase 6.5 - Training Pipeline Restart
- **Status**: 🔴 **BLOCKED** - Feature mismatch discovered (50 vs 31)
- **Critical Issue**: Colab models incompatible with local evaluation

## Architecture

### Core Components
1. **Data Layer**: Coinbase API, SQLite/PostgreSQL, S3 storage
2. **AI/ML Layer**: LSTM (35%), Transformer (40%), RL (25%) ensemble
3. **Trading Runtime**: Signal generation, FTMO rules, rate limiting
4. **Notification/Execution**: Database logging, Telegram alerts, MT5 bridge

### Directory Structure
```
crpbot/
├── apps/
│   ├── trainer/          # LSTM/Transformer/RL training
│   ├── runtime/          # VPS runtime: scanning + signals  
│   ├── kafka/            # Kafka streaming (new)
│   └── mt5_bridge/       # FTMO/MT5 connectors
├── libs/
│   ├── data/             # Data providers (Coinbase, synthetic)
│   ├── config/           # Pydantic configuration
│   ├── db/               # Database models & operations
│   ├── confidence/       # Confidence calibration
│   ├── constants/        # Trading constants
│   ├── rl_env/           # PPO Gym environment
│   └── aws/              # S3 client & secrets
├── data/
│   ├── raw/              # Raw OHLCV data (2 years, 3 coins)
│   └── features/         # Engineered features (39 columns)
├── models/               # Trained model weights
├── scripts/              # Utilities & automation
├── tests/                # Unit, smoke, integration tests
└── docs/                 # Comprehensive documentation
```

## Current Status (Phase 6.5)

### Completed ✅
1. **Data Infrastructure**: Coinbase API with JWT authentication
2. **Dataset Generation**: BTC/ETH/SOL 2-year 1m candles (1M+ rows each)
3. **Feature Engineering**: 39 features (technical, session, volume, volatility)
4. **Model Training**: 3/3 LSTM models trained (BTC/ETH/SOL)

### In Progress 🔄
- **Transformer Training**: Global multi-coin model (queued)
- **Multi-TF Pipeline**: Parallel development for multiple timeframes

### Queued ⏹️
- Model evaluation & promotion
- Runtime smoke testing
- Phase 6.5 observation restart
- Phase 7 go/no-go decision

## Key Technologies
- **ML**: PyTorch, scikit-learn, gymnasium (RL)
- **Data**: pandas, pyarrow (parquet), ta (technical analysis)
- **API**: Coinbase Advanced Trade (JWT), python-telegram-bot
- **DB**: SQLAlchemy, PostgreSQL/SQLite
- **Infrastructure**: AWS S3, Docker, systemd
- **Dev**: pytest, ruff, mypy, pre-commit

## Configuration
- **Environment**: `.env` file with API keys, DB URL, safety settings
- **Build**: `pyproject.toml` with dependencies and tool configs
- **Automation**: `Makefile` with setup, training, testing commands

## Safety Features
- **Kill Switch**: Instant halt capability
- **Rate Limiting**: Max 10 signals/hour, 5 high-confidence/hour
- **FTMO Compliance**: 5% daily, 10% total loss limits
- **Confidence Tiers**: High (75%+), Medium, Low classification

## Development Workflow
1. Feature branches from main
2. Pre-commit hooks (format, lint, type-check)
3. All tests must pass
4. PR review required
5. CI/CD via GitHub Actions

## ⚠️ CRITICAL BLOCKER (2025-11-13)

**Issue**: Feature dimension mismatch between Colab training and local evaluation

**Impact**: Cannot evaluate or use Colab-trained models
- Colab models: 50 input features
- Local pipeline: 31 input features
- Error: `RuntimeError: size mismatch [512, 50] vs [512, 31]`

**Root Cause**: Colab environment used 19 additional features (likely multi-TF) not in local feature files

**Solution**: Retrain on Colab with correct 31-feature parquet files
- Files ready: `data/features/*.parquet` (210MB, 200MB, 184MB)
- Estimated time: ~57 minutes GPU training
- See: `COLAB_RETRAINING_INSTRUCTIONS.md` for step-by-step guide

**Documentation**:
- Problem report: `reports/phase6_5/CRITICAL_FEATURE_MISMATCH_REPORT.md`
- Retraining guide: `COLAB_RETRAINING_INSTRUCTIONS.md`
- Commits: e25b970, befdeb2

## Next Steps
1. 🔴 **URGENT**: Upload feature files to Google Drive & retrain on Colab (manual)
2. Download and evaluate retrained models
3. Complete Transformer training
4. Evaluate all models against Phase 3 gates (68% accuracy, 5% calibration)
5. Promote best models to production
6. Restart Phase 6.5 observation with meaningful signals
7. Prepare for Phase 7 deployment
