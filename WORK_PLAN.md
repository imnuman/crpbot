# Trading AI - Revised Work Plan

## Executive Summary

This plan prioritizes a shippable V1 using LSTM/Transformer + confidence calibration, with runtime/Telegram integrated early for iterative feedback. GAN and RL are optional enhancements that can be added after go-live.

**Timeline**: ~11.5-13.5 days for V1 (includes +2 day buffer for data/API friction + Phase 3 buffer + Phase 8 extension), +2-4 days optional for GAN/RL

---

## Phase 1: Infrastructure & Foundation (0.5-1d)

### 1.1 Repository Scaffolding
- [ ] Create full directory structure per blueprint
- [ ] Set up Python 3.10 environment with `uv` (or Poetry)
- [ ] Configure `pyproject.toml` with pinned dependencies
- [ ] Create `.env.example` with all required variables including:
  - `KILL_SWITCH=false` (safety rail)
  - `MAX_SIGNALS_PER_HOUR=10` (rate limiting)
  - `MAX_SIGNALS_PER_HOUR_HIGH=5` (tier-specific limits)

### 1.2 Development Tooling
- [ ] Create `Makefile` with all common tasks
- [ ] Configure pre-commit hooks (ruff, mypy, bandit)
- [ ] Set up `.pre-commit-config.yaml`
- [ ] Add deterministic seed configuration (numpy, torch, cudnn)

### 1.3 Configuration System
- [ ] Implement Pydantic-based config system (`libs/config/`)
- [ ] Support config files + env overrides
- [ ] **Schema validation at boot** (fail fast on invalid env vars)
- [ ] **`--config` flag** for CI to run seeded configs for smoke tests
- [ ] Ensure reproducibility for CI/runtime

### 1.4 CI/CD Pipeline
- [ ] GitHub Actions workflow for linting/type checking
- [ ] Smoke test workflow (5-minute backtest with seeded determinism)
- [ ] Security scanning (bandit)
- [ ] Optional: Docker build workflow for runtime image

### 1.5 Infrastructure Files
- [ ] Dockerfile for runtime
- [ ] Devcontainer configuration (optional)
- [ ] Systemd service unit
- [ ] Deployment scripts

---

## Phase 2: Data Pipeline & Empirical FTMO Execution Model (1-1.5d)

### 2.1 Data Ingestion
- [ ] Implement Binance API connector for 1m candles (BTC/ETH/BNB, **2020-present**)
  - **End date**: Current date at time of initial data collection
  - Plan for ongoing data updates (weekly/monthly refresh)
- [ ] Create data cleaning and validation pipeline
- [ ] Set up DVC for data versioning (or Git LFS initially)
- [ ] Implement walk-forward split logic:
  - Train: 2020-2023
  - Val: 2024
  - Test: 2025-present (or rolling window, e.g., last 6 months)

### 2.2 Feature Engineering
- [ ] Technical indicators (ATR, spread, volume)
- [ ] **Session features** (critical):
  - Encode session (Tokyo/London/NY)
  - Day-of-week
  - Volatility regime (high/medium/low)
- [ ] Feature normalization pipeline
- [ ] Handle NaN values and edge cases
- [ ] **Feature store-lite**: Versioned parquet for engineered features
  - Store in `data/features/features_YYYY-MM-DD.parquet`
  - Tie to DVC remote for versioning
  - Avoid recompute drift between training and backtest runs
  - Enable fast reload without recomputation

### 2.3 Empirical FTMO Execution Model (CRITICAL)
**Do NOT hardcode 12bps spread + 3bps slippage globally**

- [ ] Create execution measurement system:
  - Measure spreads/slippage per pair (BTC/ETH/BNB)
  - Measure per session (Asia/EU/US)
  - Track distributions (mean, p50, p90)
- [ ] Store measurements in **versioned, date-stamped JSON** (`data/execution_metrics_YYYY-MM-DD.json`)
  - Enable rollback to previous calibration if needed
- [ ] Create nightly job to recompute metrics from FTMO bridge:
  - **DECISION: VPS cron** (simpler for production readiness, runs on same infrastructure as runtime)
  - **Authentication**: Use read-only FTMO credentials (separate from trading account)
  - Persist versioned JSON with date stamps
  - Update symlink to latest (`data/execution_metrics.json` → latest dated file)
  - Cron job: `0 2 * * *` (2 AM daily) via `infra/scripts/nightly_exec_metrics.sh`
- [ ] Update execution model to sample from distributions:
  - `libs/rl_env/execution_model.py` uses empirical data
  - Backtests sample from session-specific distributions
  - Runtime uses same model for consistency
- [ ] Integration with training and backtesting pipelines

### 2.4 Data Quality Checks
- [ ] Leakage test: assert no features derived from future data (T+1)
- [ ] Data completeness validation
- [ ] Missing value detection and handling

---

## Phase 3: LSTM/Transformer Models + Evaluation Gates (2.5-3d) ⚠️ **BUFFERED**

**Note**: Realistically 3-4d with per-coin training, full evaluation framework, and calibration. This phase includes significant work.

### 3.1 LSTM Model
- [ ] Direction prediction model (15-min horizon)
- [ ] Per-coin training (BTC, ETH, BNB)
- [ ] Training loop with walk-forward validation splits
- [ ] Save/load model weights with experiment tracking
- [ ] Model versioning by hash + params

### 3.2 Transformer Model
- [ ] Trend strength prediction
- [ ] Multi-coin training capability
- [ ] Attention mechanism implementation
- [ ] Experiment tracking integration

### 3.3 Evaluation Framework
- [ ] Backtest engine with empirical FTMO execution model
- [ ] **Enhanced metrics** (beyond accuracy):
  - Per-tier precision/recall
  - Brier score / calibration curves
  - Average/Max drawdown
  - Hit rate by session
  - Latency-adjusted PnL vs naive baseline
- [ ] **Latency measurement**: record end-to-end decision latency
- [ ] **Latency budget SLA**: Penalize stale fills in simulator
  - If decision latency > N ms (configurable, e.g., 500ms), degrade entry by p90 slippage
  - Apply penalty in backtest PnL calculations
- [ ] Model promotion gates:
  - Validation accuracy ≥0.68 per coin (time-split)
  - Calibration gate: tier MAE ≤5% on validation
  - No leakage detected

### 3.4 Experiment Tracking & Model Versioning
- [ ] Simple CSV/TensorBoard index for runs
- [ ] Name models by hash + params (e.g., `lstm_btc_a3f2d1_epochs10_hidden64.pt`)
- [ ] Log all hyperparameters and metrics
- [ ] **Model versioning strategy**:
  - Store in `models/` with semantic tags: `v1.0.0`, `v1.1.0`, etc.
  - Create model registry JSON: `models/registry.json` (maps tag → file path, hash, metrics, deployment date)
  - Tag promoted models: `models/promoted/` (symlink to versioned file)
  - Enable quick rollback: `make rollback-model TAG=v1.0.0`

---

## Phase 4: Runtime + Telegram (Dry-Run) (0.5-1d) ⬆️ **MOVED UP**

**Rationale**: Get dry-run signals early to observe formatting, thresholds, and rule enforcement during model iteration.

**⚠️ IMPORTANT**: This phase creates runtime scaffolding with mock/stub models. Real validated models (from Phase 3) are integrated post-Phase 7 (after micro-lot validation). This allows:
- Early feedback on signal formatting and Telegram UX
- Testing FTMO rules enforcement without risk
- Iterating on confidence thresholds before models are ready
- Full integration happens in Phase 8 (Go-Live) once models pass all gates

### 4.1 Runtime Loop
- [ ] 2-minute scanning cycle
- [ ] Coin scanning → prediction → confidence scoring
- [ ] FTMO rule enforcement (dry-run mode)
- [ ] Signal generation logic
- [ ] **Kill-switch**: env var that halts emissions instantly
- [ ] **Rate limiting**: max N/hour per tier (configurable)
- [ ] **Backoff logic**: after 2 losses within 60 minutes, reduce risk by X% for session

### 4.2 Telegram Bot
- [ ] Command handlers:
  - `/check` - system status
  - `/stats` - performance metrics
  - `/ftmo_status` - FTMO account status
  - `/threshold <n>` - adjust confidence threshold
  - `/kill_switch <on|off>` - emergency stop
- [ ] Message sending and formatting
- [ ] Error handling and logging
- [ ] Dry-run mode indicator in messages

### 4.3 Observability
- [ ] Structured JSON logs with fields:
  - `pair`, `tier`, `conf`, `entry`, `tp`, `sl`, `rr`
  - `lat_ms`, `spread_bps`, `slip_bps`
  - **`mode`**: Tag every message/log with `mode=dryrun` vs `mode=live`
  - Prevents mixing stats and makes promotion audits trivial
- [ ] Nightly export to S3/local for analysis
- [ ] `/healthz` endpoint for uptime checks

### 4.4 MT5 Bridge Interface (Pre-Work)
- [ ] **Create thin interface + mock NOW** (`apps/mt5_bridge/interface.py`)
  - Abstract interface with mock implementation for development
  - Prevents blocking if Python MT5 wheel has Linux issues
- [ ] Initial implementation (Python MetaTrader5 module)
- [ ] **Fallback strategy**: Add README note for Windows VM + REST shim path
  - Document setup: MT5 on Windows VM, expose local REST API
  - Alternative: MetaAPI provider
  - Ensure swapping providers doesn't ripple through codebase
- [ ] Add `apps/mt5_bridge/README.md` with fallback instructions

---

## Phase 5: Confidence System + FTMO Rules + Database (0.5-1d)

### 5.1 Enhanced Confidence Scoring
- [ ] Ensemble method with **fallback weights**:
  - **With RL**: 35% LSTM, 40% Transformer, 25% RL
  - **Without RL** (if Phase 9 skipped): 50% LSTM, 50% Transformer
  - Configurable via `ENSEMBLE_WEIGHTS` env var or config file
- [ ] Conservative bias application (-5%)
- [ ] **Platt/Isotonic scaling** option if calibration error >5%
- [ ] **Tier hysteresis**: require 2 consecutive scans > threshold to upgrade tier
- [ ] **Per-pattern sample floor**: cap influence from <N observations
- [ ] Historical pattern adjustment

### 5.2 FTMO Rules Library
- [ ] Daily loss limits (4.5% of account)
- [ ] Total loss limits (9% of account)
- [ ] Position sizing helpers
- [ ] Integration with all execution paths
- [ ] **Guardrail test**: try to emit signal after synthetic losses push over limits → assert it blocks

### 5.3 Database & Auto-Learning
- [ ] **Database schema** (detailed design):
  - **Patterns table**: `id`, `name`, `pattern_hash`, `wins`, `total`, `created_at`, `updated_at`
  - **Risk book snapshots table**: 
    - `signal_id`, `pair`, `tier`, `entry_time`, `entry_price`, `tp_price`, `sl_price`, `rr_expected`
    - `result` (W/L/null), `exit_time`, `exit_price`, `r_realized`, `time_to_tp_sl_seconds`
    - `slippage_bps`, `slippage_expected_bps`, `spread_bps`, `latency_ms`, `mode` (dryrun/live)
    - Indexes: `signal_id` (unique), `entry_time`, `pair`, `mode`
  - **Model deployments table**: `version`, `model_path`, `deployed_at`, `metrics_json`, `rollback_reason` (nullable)
- [ ] **Retention policy**:
  - Risk book: Keep all live trades indefinitely, archive dry-run after 90 days
  - Patterns: Keep all (small dataset)
  - Model deployments: Keep all (audit trail)
  - Monthly archival to S3/parquet for long-term analysis
- [ ] **Query patterns** (optimize for):
  - Recent performance (last 7/30 days) for `/stats` command
  - Pattern win rates (filtered by sample floor)
  - Model comparison (accuracy by version)
  - Session analysis (performance by Tokyo/London/NY)
- [ ] **Backup strategy**:
  - Daily automated backup to S3 (via `infra/scripts/backup_db.sh`)
  - Point-in-time recovery capability
  - Test restore procedure quarterly
- [ ] Auto-learning system:
  - Pattern tracking and result recording
  - Historical adjustment logic
  - Sample floor enforcement

---

## Phase 6: Testing & Validation (1d)

### 6.1 Unit Tests
- [ ] Test all core functions
- [ ] Mock external dependencies
- [ ] Test FTMO rules logic
- [ ] Test confidence calibration

### 6.2 Smoke Tests
- [ ] 5-minute backtest simulation (seeded for determinism)
- [ ] Fast validation in CI

### 6.3 Integration Tests
- [ ] End-to-end runtime simulation
- [ ] Model loading and inference tests
- [ ] Telegram bot command tests

### 6.4 Critical Tests
- [ ] **Leakage test**: train on T, assert any feature derived from >T is absent
- [ ] **FTMO guardrail test**: synthetic losses push over limits → assert signal blocked
- [ ] **Calibration gate**: predicted vs actual tier win rate MAE ≤5% on validation

### 6.5 Security & Compliance
- [ ] Secrets management: `.env` locally, GitHub Environments for Actions
- [ ] Never print tokens in logs
- [ ] FTMO TOS compliance check (no automation violations)

---

## Phase 7: Micro-Lot Testing & Calibration (2-3d)

### 7.1 Micro-Lot Live Testing
- [ ] 0.01 lot positions ($1-$2 risk)
- [ ] 2-3 day observation period
- [ ] Track simulated vs real fills
- [ ] Adjust slippage model based on real data
- [ ] Record latency measurements
- [ ] Track all metrics (win rate, R realized, time-to-TP/SL)

### 7.2 FTMO Challenge Simulation
- [ ] Run 20 simulated challenge attempts
- [ ] Require ≥70% pass rate
- [ ] Record average days to pass
- [ ] Log all rule breaches
- [ ] Final confidence calibration (±5% error target)

### 7.3 Acceptance Gate
- [ ] ≥68% win rate over **≥100 trades** (statistical significance requirement)
- [ ] Latency-adjusted performance
- [ ] Before raising risk, validate all metrics
- [ ] **Note**: 100+ trades needed for narrow confidence intervals at 68% win rate

---

## Phase 8: Go-Live (1d) ⚠️ **EXTENDED**

### 8.1 Final Validation
- [ ] Final model validation
- [ ] Runtime configuration review
- [ ] **Alerting minimalism** (two alerts only):
  - **Kill-switch required**: Alert if kill-switch set but signals still attempted
  - **Guardrail breach**: Alert on any attempt to emit when daily/total loss would be violated
  - Use structured logs as source (no separate monitoring stack needed)
- [ ] Kill-switch tested
- [ ] Rate limiting verified

### 8.2 Model Integration & Deployment
- [ ] **Integrate validated models** from Phase 3 (replacing P4 stubs):
  - Load models from `models/promoted/` (symlinked versions)
  - Update runtime config to use validated model paths
  - Verify model loading and inference latency
- [ ] **Model rollback procedure** (documented and tested):
  - Create `infra/scripts/rollback_model.sh` script
  - Update `models/registry.json` with deployment timestamp
  - Rollback command: `make rollback-model TAG=<previous_version>`
  - Test rollback flow: deploy v1.1 → rollback to v1.0 → verify runtime uses v1.0
  - Document rollback triggers (accuracy drop, guardrail violations, etc.)
- [ ] VPS deployment (systemd)
- [ ] Environment variables configured
- [ ] Logging and observability active
- [ ] Health checks operational

---

## Phase 9: Optional - GAN & RL Enhancement (2-4d) ⚠️ **OPTIONAL**

**Only proceed if V1 is stable and you want additional edge. These can introduce instability.**

### 9.1 GAN for Synthetic Data
- [ ] Generator and discriminator networks
- [ ] Synthetic data generation (capped at ≤10-20% of dataset)
- [ ] **Always tag synthetic rows** for tracking
- [ ] Quality checks and sanity validation
- [ ] **Ablation tests**: run with 0% synth to prove it helps
- [ ] Watch for artifacts and regime hallucinations

### 9.2 RL Environment
- [ ] Gymnasium environment for trading
- [ ] Action space (buy, sell, hold)
- [ ] Observation space with market features
- [ ] Reward function design (avoid leakage)
- [ ] PPO implementation with execution model
- [ ] Gradient clipping and stability checks

### 9.3 RL Validation Requirements
- [ ] **Require reward curve stability**
- [ ] **Out-of-sample improvement over supervised ensemble**
- [ ] If not meeting requirements, keep as research branch only
- [ ] Do not integrate into runtime until validated

### 9.4 Re-Validation
- [ ] Re-run all validation tests
- [ ] Update confidence system if RL proves beneficial
- [ ] Update acceptance gates

---

## Acceptance Gates (MANDATORY - DO NOT SKIP)

### Validation Gates
- [ ] **Val floor**: ≥0.68 accuracy per coin (time-split validation)
- [ ] **Calibration**: tier MAE ≤5% on validation
- [ ] **FTMO guardrails**: zero violations across 20 challenge sims
- [ ] **Pass rate**: ≥70% challenge pass rate; log all breaches
- [ ] **Micro-lot**: ≥68% win rate over **≥100 trades** (statistical significance), latency-adjusted, before raising risk

### Quality Gates
- [ ] No leakage detected (leakage test passes)
- [ ] No reward leakage in RL (if using RL)
- [ ] No data snooping in confidence calibration
- [ ] All smoke tests pass in CI
- [ ] All unit tests pass
- [ ] FTMO rules enforced in all execution paths

### Operational Gates
- [ ] Kill-switch tested and working
- [ ] Rate limiting configured and tested
- [ ] Structured logging operational
- [ ] Health checks operational
- [ ] Secrets management secure (no tokens in logs/git)

---

## Key Implementation Notes

### Execution Model (Critical)
- **DO NOT** hardcode spread/slippage values
- **DO** measure per pair and per session
- **DO** use distributions (mean, p50, p90)
- **DO** update nightly from real FTMO data:
  - **DECISION: VPS cron** (2 AM daily, simpler for production readiness)
  - Use read-only FTMO credentials
  - Versioned, date-stamped JSON (enables rollback)
- **DO** use same model in backtests and runtime

### Data Rigor
- **Walk-forward splits** (not single split)
- **Session features** explicitly encoded
- **Seeded determinism** for CI reproducibility
- **Latency measurement** included in backtests
- **Latency budget SLA**: Penalize stale fills (>N ms → degrade by p90 slippage)
- **Feature store-lite**: Versioned parquet to avoid recompute drift (tied to DVC)

### Confidence System
- Keep -5% conservative bias
- Add Platt/Isotonic scaling if needed
- Tier hysteresis (2 consecutive scans)
- Per-pattern sample floor

### Ops & Safety
- Kill-switch env var
- Rate limiting per tier
- Backoff after losses
- Structured JSON logs with `mode` tag (dryrun/live)
- Nightly export for analysis
- **Alerting minimalism**: Two alerts only (kill-switch violation, guardrail breach)

### MT5 Bridge
- **Create thin interface + mock early** (prevents blocking)
- Abstract interface (swap providers easily)
- Plan fallback (Windows VM + REST shim or MetaAPI) - document in README
- Handle Linux MT5 issues gracefully

### Config System
- Schema validation at boot (fail fast on invalid env)
- `--config` flag for CI seeded configs
- Pydantic-based for type safety

### Model Versioning & Rollback
- Semantic versioning: `v1.0.0`, `v1.1.0`, etc.
- Model registry JSON tracks all versions with metrics and deployment dates
- Promoted models in `models/promoted/` (symlinks)
- Rollback procedure: `make rollback-model TAG=<version>`
- Test rollback flow before go-live

### Model Retraining Schedule
**Retraining triggers** (automatic or manual):
- **Weekly schedule**: Retrain on Sunday (low market activity) with latest data
- **Performance degradation threshold**: 
  - If validation accuracy drops below 0.65 (5% below gate) for 3 consecutive days
  - If micro-lot win rate drops below 65% for 30+ trades
  - If confidence calibration error exceeds 7% (threshold + 2%)
- **Data drift detection**: If feature distributions shift significantly (Kolmogorov-Smirnov test)
- **Manual trigger**: Via `/retrain` Telegram command (with confirmation)
- **Retraining process**:
  - Use walk-forward validation (train on latest full period, validate on most recent month)
  - Must pass all promotion gates (≥0.68 accuracy, ≤5% calibration error) before deployment
  - Version new model (increment patch version: v1.0.0 → v1.0.1)
  - Deploy via Phase 8 process (integration + rollback tested)

### Runtime/Model Integration
- Phase 4: Runtime scaffolding with mock/stub models (early UX feedback)
- Phase 8: Real validated models integrated post-micro-lot validation
- Clear separation prevents deploying unvalidated models to production

### Database Design
- Detailed schema with retention policies
- Query optimization for common patterns (stats, session analysis)
- Daily automated backups to S3
- Point-in-time recovery capability

### GAN & RL Reality Check
- GAN: Cap at ≤10-20%, tag synthetic rows, run ablations
- RL: Require stability and improvement, else keep as research branch

---

## Timeline Summary

| Phase | Duration | Status |
|-------|----------|--------|
| P1: Infra | 0.5-1d | - |
| P2: Data + Exec Model | 1-1.5d | - |
| P3: LSTM/Transformer | 2.5-3d | ⚠️ Buffered (+1d) |
| P4: Runtime + Telegram | 0.5-1d | - |
| P5: Confidence + Rules + DB | 0.5-1d | - |
| P6: Tests/Validation | 1d | - |
| P7: Micro-lot + Calibration | 2-3d | - |
| P8: Go-Live | 1d | ⚠️ Extended (+0.5d) |
| **Buffer (data/API friction)** | **+2d** | - |
| **V1 Total** | **11.5-13.5d** | - |
| P9: RL + GAN (optional) | 2-4d | - |
| **With Optional** | **13.5-17.5d** | - |

---

## Next Steps

1. Review and confirm this plan
2. Set up infrastructure (Phase 1)
3. Begin with data pipeline and empirical execution model (Phase 2)
4. Iterate with early runtime feedback (Phase 4 after Phase 3)

Ready to proceed when you are!


