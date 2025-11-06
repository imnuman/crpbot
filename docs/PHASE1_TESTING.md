# Phase 1 Testing Guide

## Quick Test Checklist

### ✅ 1. Environment Setup

```bash
# Ensure you're in the project directory
cd /home/numan/crpbot

# Add uv to PATH (if not already done)
export PATH="$HOME/.local/bin:$PATH"

# Activate virtual environment
source .venv/bin/activate
```

### ✅ 2. Run All Tests

```bash
# Run all tests (should pass)
pytest tests/ -v

# Expected output: 6 tests passed
```

### ✅ 3. Test Linting

```bash
# Check code quality
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

### ✅ 4. Test Imports

```bash
# Test config system
python -c "from libs.config.config import Settings; print('✅ Config works')"

# Test MT5 bridge
python -c "from apps.mt5_bridge.interface import MockMT5Bridge; print('✅ MT5 bridge works')"
```

### ✅ 5. Test Runtime Stub

```bash
# Run runtime (should complete without errors)
python apps/runtime/main.py

# Expected: Logs showing runtime starting, 3 iterations, exiting
```

### ✅ 6. Test Trainer Stub

```bash
# Test trainer CLI
python apps/trainer/main.py --task lstm --coin BTC --epochs 1

# Expected: Log message about training LSTM
```

### ✅ 7. Test Makefile Commands

```bash
# View available commands
make help

# Run tests
make test

# Run smoke tests
make smoke

# Format code
make fmt

# Lint code
make lint
```

## What Should Work

- ✅ All 6 tests pass
- ✅ No linting errors
- ✅ Imports work correctly
- ✅ Runtime stub runs without errors
- ✅ Trainer stub runs without errors
- ✅ Makefile commands work
- ✅ Config system validates correctly
- ✅ Ensemble weights normalize correctly

## Known Limitations (Expected)

- ⚠️ Runtime uses stub/mock implementations (to be implemented in Phase 4)
- ⚠️ Trainer uses stub implementations (to be implemented in Phase 3)
- ⚠️ MT5 bridge uses mock (to be implemented in Phase 2)
- ⚠️ Smoke tests are placeholders (to be implemented in Phase 3)

These are expected - Phase 1 is just the foundation!

## Next Steps

Once Phase 1 testing passes:
1. ✅ Commit any fixes
2. ✅ Push to GitHub
3. ✅ Verify GitHub Actions CI runs successfully
4. 🚀 Proceed to Phase 2: Data Pipeline & FTMO Execution Model

