# Phase 6.5 Restart – Training Pipeline Plan

**Date**: 2025-11-10  
**Status**: PLANNED (no execution yet)  
**Owner Branch**: `claude/phase-6.5-checklist-011CUyHzB8ZnJdnENDyCMDUQ`

---

## 🎯 Goal

Observation in Phase 6.5 produced zero signals because we never trained or promoted models.  
This document is a coordinated plan to generate data, train models, and restart the silent observation with meaningful output.

---

## 🧭 Checklist Overview

| Step | Description | Owner | Status |
|------|-------------|-------|--------|
| 1 | Data provider infrastructure (synthetic/yfinance/ccxt) | Claude → Cursor | ☐ |
| 2 | Generate raw datasets (synthetic or external) | Cursor | ☐ |
| 3 | Engineer features (`scripts/engineer_features.py`) | Cursor | ☐ |
| 4 | Train models (per-coin LSTM + global transformer) | Cursor | ☐ |
| 5 | Evaluate & promote models (Phase 3 gates) | Claude | ☐ |
| 6 | Point runtime at promoted models, smoke-test | Cursor | ☐ |
| 7 | Restart Phase 6.5 observation (3–5 days) | Cursor + Claude | ☐ |
| 8 | Observation summary & go/no-go for Phase 7 | Claude + Amazon Q | ☐ |

> Update the status column as each step is completed.  
> Nothing in this list has been performed yet.

---

## 1. Data Infrastructure

- [ ] Choose provider via `.env` (`DATA_PROVIDER=synthetic` recommended until network access is confirmed).
- [ ] Optional dependencies (install only if using those providers):
  ```bash
  uv pip install yfinance
  uv pip install ccxt
  ```
- [ ] Verify synthetic fallback works:
  ```bash
  uv run python - <<'PY'
  from libs.data.provider import create_data_provider
  provider = create_data_provider("synthetic")
  assert provider.test_connection()
  PY
  ```

---

## 2. Dataset Generation

- [ ] Fetch per-symbol data (repeat for BTC-USD, ETH-USD, etc.):
  ```bash
  uv run python scripts/fetch_data.py \
      --provider synthetic \
      --symbol BTC-USD \
      --interval 1m \
      --days 365
  ```
- [ ] Store outputs in `data/raw/` (parquet files).
- [ ] Document datasets (rows, date range) in `reports/data_inventory.md`.

---

## 3. Feature Engineering

- [ ] Generate engineered features:
  ```bash
  uv run python scripts/engineer_features.py --symbol BTC-USD --interval 1m
  ```
- [ ] Validate results:
  ```bash
  uv run python scripts/validate_data_quality.py --symbol BTC-USD
  ```
- [ ] Confirm artifacts in `data/features/`.

---

## 4. Model Training

| Model | Command | Notes |
|-------|---------|-------|
| LSTM (per coin) | `make train COIN=BTC EPOCHS=10` (repeat for ETH, BNB) | GPU: 2–4h, CPU: 6–8h |
| Transformer | `uv run python apps/trainer/main.py --task transformer --epochs 10` | GPU: 4–8h, CPU: 12–16h |

Record metrics/plots under `reports/training/`.

---

## 5. Evaluation & Promotion

- [ ] Evaluate each checkpoint against promotion gates:
  ```bash
  uv run python scripts/evaluate_model.py \
      --model models/checkpoints/lstm_BTC-USD_best.pt \
      --symbol BTC-USD \
      --model-type lstm \
      --min-accuracy 0.68 \
      --max-calibration-error 0.05
  ```
- [ ] If a model passes:
  - Symlink/copy to `models/promoted/`
  - Update `models/registry.json`
  - Log results in `reports/models/phase6_5_evaluation.md`

---

## 6. Runtime Integration

- [ ] Update runtime settings (env or config) to reference promoted models.
- [ ] Smoke test locally:
  ```bash
  uv run python apps/runtime/main.py --mode dryrun --iterations 20 --sleep-seconds 1
  ```
- [ ] Confirm signals are persisted with non-zero confidence.

---

## 7. Phase 6.5 Observation Restart

- [ ] Launch observation loop: `make run-dry`
- [ ] Daily automation:
  ```bash
  make phase6_5-daily DAY=day0
  ```
- [ ] Capture metrics & notes in `reports/phase6_5/dayX.md` and `automation_log.md`.
- [ ] Claude reviews daily snapshots; Amazon Q checks CloudWatch on Day 1 & Day 3.

**Exit Criteria**

- ≥72h runtime without crash or Sev‑1 alarm  
- FTMO guardrail simulation blocked successfully  
- Telegram notifications delivered for high-tier signals  
- Win-rate & tier distributions within ±5% of backtest baseline  
- Observation summary approved (`reports/phase6_5/summary.md`)

---

## 8. Phase 7 Preparation (in parallel)

- [ ] Confirm FTMO demo/challenge credentials (`docs/CREDENTIALS_CHECKLIST.md`).
- [ ] Review execution-model calibration plan (`docs/EXECUTION_MODEL.md`).
- [ ] Re-test kill-switch, rate limiter, rollback procedures.

---

## Reference Commands

```bash
# Fetch data
uv run python scripts/fetch_data.py --provider synthetic --symbol BTC-USD --interval 1m --days 365

# Engineer features
uv run python scripts/engineer_features.py --symbol BTC-USD --interval 1m

# Train models
make train COIN=BTC EPOCHS=10
uv run python apps/trainer/main.py --task transformer --epochs 10

# Evaluate
uv run python scripts/evaluate_model.py --model models/checkpoints/lstm_BTC-USD_best.pt --symbol BTC-USD --model-type lstm

# Observation automation
make run-dry
make phase6_5-daily DAY=day1
```

---

## Notes

- Synthetic provider gives offline coverage; switch to `yfinance` or `ccxt` when network access & dependencies are available.
- Optional dependencies (`yfinance`, `ccxt`) are not pinned in `uv.lock`; install only if required.
- Treat this file as the single source of truth for the restart effort and update the checklist as progress is made.
