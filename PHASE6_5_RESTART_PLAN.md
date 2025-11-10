# Phase 6.5 Restart – Training Pipeline Plan

**Date Started**: 2025-11-10
**Last Updated**: 2025-11-10 (Training in Progress)
**Status**: ✅ **IN EXECUTION** - Steps 1-3 Complete, Step 4 In Progress
**Owner Branch**: `main` (executing on production data)

---

## 🎯 Goal

Observation in Phase 6.5 produced zero signals because we never trained or promoted models.  
This document is a coordinated plan to generate data, train models, and restart the silent observation with meaningful output.

---

## 🧭 Checklist Overview

| Step | Description | Owner | Status |
|------|-------------|-------|--------|
| 1 | Data provider infrastructure (synthetic/yfinance/ccxt) | Claude → Cursor | ✅ Complete (2025-11-10) |
| 2 | Generate raw datasets (synthetic or external) | Cursor | ✅ Complete (2025-11-10) |
| 3 | Engineer features (`scripts/engineer_features.py`) | Cursor | ✅ Complete (2025-11-10) |
| 4 | Train models (per-coin LSTM + global transformer) | Cursor | 🔄 In Progress (1/4 complete) |
| 5 | Evaluate & promote models (Phase 3 gates) | Claude | ⏹️ Queued |
| 6 | Point runtime at promoted models, smoke-test | Cursor | ⏹️ Queued |
| 7 | Restart Phase 6.5 observation (3–5 days) | Cursor + Claude | ⏹️ Queued |
| 8 | Observation summary & go/no-go for Phase 7 | Claude + Amazon Q | ⏹️ Queued |

> Update the status column as each step is completed.  
> **Steps 1-3 completed, Step 4 in progress (BTC-USD model trained, ETH-USD training)**

---

## 1. Data Infrastructure ✅ COMPLETE

**Chosen Provider**: Coinbase Advanced Trade API (JWT authentication)
**Status**: Fully functional, production-ready

- [x] **Coinbase API configured** via `.env` with JWT authentication
  - API Key: `organizations/*/apiKeys/*` format
  - Private Key: PEM format (EC PRIVATE KEY)
  - Authentication method: JWT token generation (2-minute expiry)
- [x] **Connection verified**: Successfully fetching real market data
- [x] **Fallback available**: Synthetic provider tested and working for offline scenarios

**Implementation Details**:
- Provider: `libs/data/coinbase.py` (full JWT implementation)
- Test connection: ✅ Passed
- Rate limiting: Handled via API
- Data quality: Real 1-minute OHLCV candles from Coinbase Pro

---

## 2. Dataset Generation ✅ COMPLETE

**Date Range**: 2023-11-10 to 2025-11-10 (2 years)
**Interval**: 1-minute candles
**Provider**: Coinbase Advanced Trade API

### Generated Datasets

| Symbol | File | Size | Rows | Status |
|--------|------|------|------|--------|
| BTC-USD | `data/raw/BTC-USD_1m_2023-11-10_2025-11-10.parquet` | 35 MB | 1,030,512 | ✅ |
| ETH-USD | `data/raw/ETH-USD_1m_2023-11-10_2025-11-10.parquet` | 32 MB | 1,030,512 | ✅ |
| SOL-USD | `data/raw/SOL-USD_1m_2023-11-10_2025-11-10.parquet` | 23 MB | 1,030,513 | ✅ |

**Notes**:
- Originally planned BNB-USD, but Coinbase returned no data
- Replaced with SOL-USD (Solana) as third cryptocurrency
- All data validated: no gaps, proper timestamp ordering
- Format: Parquet (compressed columnar storage)

---

## 3. Feature Engineering ✅ COMPLETE

**Feature Count**: 39 columns total (5 OHLCV + 31 numeric features + 3 categorical)
**Documentation**: See `FEATURE_ENGINEERING_WORKFLOW.md` for detailed breakdown

### Generated Feature Files

| Symbol | File | Rows | Columns | Status |
|--------|------|------|---------|--------|
| BTC-USD | `data/features/features_BTC-USD_1m_latest.parquet` | 1,030,512 | 39 | ✅ |
| ETH-USD | `data/features/features_ETH-USD_1m_latest.parquet` | 1,030,512 | 39 | ✅ |
| SOL-USD | `data/features/features_SOL-USD_1m_latest.parquet` | 1,030,513 | 39 | ✅ |

### Feature Categories

1. **Session Features** (5): Tokyo/London/NY sessions, day_of_week, is_weekend
2. **Spread & Execution** (4): spread, spread_pct, ATR, spread_atr_ratio
3. **Volume** (3): volume_ma, volume_ratio, volume_trend
4. **Moving Averages** (8): SMA 7/14/21/50 + price ratios
5. **Technical Indicators** (8): RSI, MACD×3, Bollinger Bands×4
6. **Volatility Regime** (3): low/medium/high classification

**Validation**: All features validated for NaN values, data leakage, and quality checks ✅

---

## 4. Model Training 🔄 IN PROGRESS

**Configuration**: 15 epochs per model (conservative first run)
**Hardware**: CPU training (~60 min per LSTM, ~40 min for Transformer)
**Started**: 2025-11-10 03:21 UTC

### Training Status

| Model | Command | Status | Completion | Model File |
|-------|---------|--------|------------|------------|
| BTC-USD LSTM | `make train COIN=BTC EPOCHS=15` | ✅ Complete | 2025-11-10 04:16 | `lstm_BTC_USD_1m_a7aff5c4.pt` (249 KB) |
| ETH-USD LSTM | `make train COIN=ETH EPOCHS=15` | 🔄 **Training** | ~35 min remaining | In progress (Epoch 6/15) |
| SOL-USD LSTM | `make train COIN=SOL EPOCHS=15` | ⏹️ Queued | ~60 min | Not started |
| Multi-Coin Transformer | `make task transformer EPOCHS=15` | ⏹️ Queued | ~40 min | Not started |

### Training Configuration

**LSTM (Per-Coin)**:
- Architecture: 2-layer bidirectional LSTM (hidden_size=64)
- Parameters: ~62,337
- Batch size: 32
- Sequence length: 60 minutes
- Prediction horizon: 15 minutes
- Loss: Binary Cross Entropy
- Optimizer: Adam
- Device: CPU
- Training split: 70% train, 15% val, 15% test
- Training sequences: 721,358 (BTC-USD example)
- Validation sequences: 154,503

**Transformer (Multi-Coin)**:
- Architecture: 4-layer encoder, 8 attention heads
- Batch size: 16
- Sequence length: 100 minutes
- Loss: MSE
- Optimizer: AdamW with warm-up

**Estimated Total Training Time**: ~2-2.5 hours for all models

### Training Logs

- BTC-USD: `/tmp/train_btc_lstm.log` ✅
- ETH-USD: `/tmp/train_eth_lstm.log` 🔄
- SOL-USD: TBD
- Transformer: TBD

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
