# Pre-Commit Checklist for Phase 3

## ✅ Verification Results

### Phase 3 Files Status
- ✅ All 12 Phase 3 core files exist locally
- ✅ All imports work correctly
- ⚠️  Phase 3 files are currently **untracked** (need to be added)

### Files to Commit

#### Modified Files (need staging):
- `.env.example` - Updated with new environment variables
- `WORK_PLAN.md` - Updated with Phase 3 details
- `apps/trainer/main.py` - Updated with Transformer training
- `libs/config/config.py` - Updated with new config
- `libs/data/provider.py` - Updated provider factory
- `pyproject.toml` - Updated dependencies

#### New Phase 3 Files (need to be added):
- `apps/trainer/models/lstm.py`
- `apps/trainer/models/transformer.py`
- `apps/trainer/train/train_lstm.py`
- `apps/trainer/train/train_transformer.py`
- `apps/trainer/train/dataset.py`
- `apps/trainer/train/trainer.py`
- `apps/trainer/eval/backtest.py`
- `apps/trainer/eval/evaluator.py`
- `apps/trainer/eval/tracking.py`
- `apps/trainer/eval/versioning.py`
- `scripts/evaluate_model.py`
- `docs/PHASE3_COMPLETE.md`

#### Phase 2 Files (also need to be added):
- `apps/trainer/data_pipeline.py`
- `apps/trainer/features.py`
- `libs/data/coinbase.py`
- `libs/data/quality.py`
- `libs/rl_env/execution_model.py`
- `libs/rl_env/execution_metrics.py`
- `scripts/*.py` (various scripts)

#### Documentation Files:
- `docs/COINBASE_*.md` (various Coinbase docs)
- `docs/DATA_*.md` (data pipeline docs)
- `docs/EXECUTION_MODEL.md`
- `docs/FEATURE_ENGINEERING.md`
- `docs/PHASE3_*.md`
- `docs/WORK_PLAN_REVIEW.md`

### Quick Commit Commands

```bash
# Add all Phase 3 files
git add apps/trainer/models/
git add apps/trainer/train/
git add apps/trainer/eval/
git add scripts/evaluate_model.py
git add docs/PHASE3_COMPLETE.md

# Add Phase 2 files (if not already added)
git add apps/trainer/data_pipeline.py
git add apps/trainer/features.py
git add libs/data/coinbase.py
git add libs/data/quality.py
git add libs/rl_env/
git add scripts/

# Add documentation
git add docs/

# Add modified files
git add .env.example WORK_PLAN.md apps/trainer/main.py libs/config/config.py libs/data/provider.py pyproject.toml

# Commit
git commit -m "feat: Phase 3 - LSTM/Transformer models with evaluation framework

- Add LSTM and Transformer model architectures
- Add training scripts for both models
- Add comprehensive evaluation framework with backtest engine
- Add experiment tracking and model versioning
- Add enhanced metrics (Brier score, drawdown, session metrics)
- Add promotion gates (68% accuracy, 5% calibration error)
- Integrate with data pipeline, feature engineering, and execution model"

# Push to main
git push origin main
```

## ✅ Verification

Before committing, verify:
- [x] All Phase 3 files exist and import correctly
- [ ] Run tests (if any)
- [ ] Check linting: `ruff check .`
- [ ] Check formatting: `ruff format .`
- [ ] Verify imports work: `python -c "from apps.trainer.models.lstm import LSTMDirectionModel"`

## 📊 Current Status

- **Branch**: main
- **Status**: Up to date with origin/main
- **Uncommitted changes**: Many files (modified + untracked)
- **Phase 3 files**: All present locally, but untracked

