# Trading AI - FTMO Crypto Signal Generator

FTMO-focused crypto trading AI system with LSTM + Transformer models, confidence calibration, Telegram runtime, and auto-learning capabilities.

## 🚀 Quick Start

```bash
# Initial setup
make setup        # Install deps & pre-commit hooks

# Development
make fmt          # Format code
make lint         # Run linting
make test         # Run all tests
make smoke        # Run 5-min smoke backtest

# Training
make train COIN=BTC EPOCHS=10    # Train LSTM for BTC
make rl STEPS=1000               # Train RL model

# Runtime
make run-bot      # Start runtime loop
```

## 📋 Project Structure

```
crpbot/
├── apps/
│   ├── trainer/          # LSTM/Transformer/RL training
│   ├── runtime/          # VPS runtime: scanning + signals
│   └── mt5_bridge/       # FTMO/MT5 connectors
├── libs/
│   ├── features/         # OHLCV, ATR, spread, session features
│   ├── rl_env/           # PPO Gym env with execution model
│   └── synth/            # GAN data utilities
├── infra/
│   ├── docker/           # Dockerfiles
│   ├── devcontainer/     # VS Code/Cursor devcontainer
│   ├── systemd/          # Service units for VPS
│   └── scripts/          # Deployment & maintenance scripts
├── data/                 # Data (DVC tracked)
├── models/               # Model weights (DVC tracked)
└── tests/                # Unit + smoke + e2e tests
```

## 🔧 Configuration

1. Copy `.env.example` to `.env`
2. Fill in your API keys and credentials
3. Configure database URL (PostgreSQL or SQLite for dev)

See `.env.example` for all available options.

## 📊 Model Training

```bash
# LSTM per coin
python apps/trainer/main.py --task lstm --coin BTC --epochs 10

# Transformer
python apps/trainer/main.py --task transformer --epochs 8

# RL PPO
python apps/trainer/main.py --task ppo --steps 8_000_000 --exec ftmo
```

## 🧪 Testing

- **Unit tests**: `make test`
- **Smoke tests**: `make smoke` (5-minute backtest)
- **CI**: All tests run on push/PR via GitHub Actions

## 📝 Development Workflow

1. Create feature branch: `git checkout -b feat/feature-name`
2. Make changes
3. Pre-commit hooks run automatically (format, lint, type-check)
4. Run tests: `make test`
5. Push and create PR
6. PR must pass CI checks before merge

## 🚢 Deployment

See `WORK_PLAN.md` for detailed deployment instructions and timeline.

## 📚 Documentation

- `WORK_PLAN.md` - Complete development plan and timeline
- `docs/WORKFLOW_SYNC_SETUP.md` - **Workflow sync setup (Cursor IDE, Claude AI, GitHub)**
- `docs/GITHUB_TOKEN_SETUP.md` - Guide for setting up GitHub tokens and secrets
- `docs/PHASE1_TESTING.md` - Phase 1 testing checklist

## ⚠️ Safety Features

- **Kill-switch**: Instant halt via env var or Telegram command
- **Rate limiting**: Max signals per hour per tier
- **FTMO guardrails**: Daily/total loss limits enforced
- **Model rollback**: Quick rollback to previous version if issues

## 📄 License

Private - Trading system for FTMO challenges

