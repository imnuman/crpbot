# CRPBot Project Memory

## Project Overview
**CRPBot** is an AI-powered cryptocurrency trading bot designed for FTMO-compliant trading with ensemble machine learning models.

### Key Details
- **Location**: `/home/numan/crpbot`
- **Language**: Python 3.10+
- **Current Phase**: Phase 6.5 - Training Pipeline Restart
- **Status**: Step 4 in progress (3/4 models trained)

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

## Next Steps
1. Complete Transformer training
2. Evaluate all models against Phase 3 gates
3. Promote best models to production
4. Restart observation with meaningful signals
5. Prepare for Phase 7 deployment
