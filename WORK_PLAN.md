# Trading AI - Revised Work Plan

## Executive Summary

This plan prioritizes a shippable V1 using LSTM/Transformer + confidence calibration, with runtime/Telegram integrated early for iterative feedback. V2 features (multi-timeframe, sentiment, Kafka) can be integrated pre-go-live (Option A) or post-go-live (Option B). GAN and RL are optional enhancements that can be added after go-live.

**V2 Integration Strategy (Recommended: Option A)**:
- **Option A**: Full integration pre-go-live (faster unified system)
  - P3.5: Multi-TF features + retrain models
  - P4-P6: Runtime, Confidence, Tests with V2 features
  - P6.5: Silent Observation (monitor V2 features)
  - P7-P8: Micro-lot + Go-Live (V2 features proven)
- **Option B**: V1 go-live first, add V2 later (safer, incremental)
  - P4-P8: V1 only (no multi-TF, no sentiment, no Kafka)
  - P8: Go-Live V1
  - P9: Add V2 features incrementally after validation

**Timeline**: 
- **V1 Only**: ~14.5-18.5 days (includes +2 day buffer + Phase 3 buffer + Phase 8 extension + 3-5 day silent observation period)
- **V1 + V2 (Option A)**: ~27.5-41.5 days (V1 + ~13-23 days for V2 features)
- **Optional**: +2-4 days for GAN/RL

---

## Infrastructure & Resource Requirements

### GPU Training Execution Details

**Training Strategy**:
- **Option 1: Cloud GPU Rental (Recommended)**
  - **Provider**: RunPod, Vast.ai, or Lambda Labs (cost-effective)
  - **GPU Type**: NVIDIA RTX 4090 or A100 (for Transformer training)
  - **Cost**: ~$0.50-2.00/hour (RTX 4090) or ~$1.50-4.00/hour (A100)
  - **Estimated Training Time**: 
    - LSTM: 2-4 hours per coin (BTC/ETH/BNB) = 6-12 hours total
    - Transformer: 4-8 hours (multi-coin) = 4-8 hours total
    - **Total GPU Cost**: ~$15-40 (one-time for initial training)
  - **Setup**: Pre-configured PyTorch Docker images available
  - **Data Transfer**: Upload data/features to cloud storage (S3/GCS) or use cloud storage directly

- **Option 2: Local GPU (if available)**
  - **Requirements**: NVIDIA GPU with 8GB+ VRAM (RTX 3060/3070 or better)
  - **Cost**: $0 (one-time hardware purchase)
  - **Setup**: Install CUDA, PyTorch with CUDA support

- **Option 3: CPU Training (Fallback)**
  - **Performance**: 5-10x slower than GPU
  - **Use Case**: Only for small models or testing
  - **Not recommended for production training**

**Parallel Processing**:
- **Data Fetching**: Parallel API calls for multiple symbols (BTC/ETH/BNB) using `asyncio` or `concurrent.futures`
- **Feature Engineering**: Parallel processing per symbol using `multiprocessing` (I/O bound, benefits from parallelization)
- **Model Training**: Sequential per model (LSTM → Transformer → RL) due to GPU memory constraints
- **Backtesting**: Parallel evaluation across multiple time periods using `multiprocessing`
- **Hyperparameter Search**: Optional parallel trials using Ray Tune or Optuna

**GPU Memory Management**:
- Batch size tuning to fit GPU memory
- Gradient accumulation for large effective batch sizes
- Mixed precision training (FP16) to reduce memory usage
- Model checkpointing to resume training if interrupted

### VPS Specifications

**Minimum Requirements**:
- **CPU**: 2+ cores (4+ recommended)
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 20GB SSD minimum (50GB recommended for data/models)
- **Network**: Stable connection with low latency to exchanges
- **OS**: Ubuntu 22.04 LTS (recommended)

**Recommended Specifications**:
- **CPU**: 4 cores (Intel/AMD x86_64)
- **RAM**: 8GB (16GB for larger datasets)
- **Storage**: 50GB SSD (for data, models, logs)
- **Network**: 100Mbps+ with <50ms latency to Coinbase/FTMO
- **Uptime**: 99.9% SLA (consider managed VPS)

**VPS Providers** (Cost-Effective Options):
- **DigitalOcean**: $12-24/month (4GB RAM, 2 vCPU)
- **Linode**: $12-24/month (4GB RAM, 2 vCPU)
- **Hetzner**: €8-16/month (4-8GB RAM, 2-4 vCPU) - Best value
- **AWS Lightsail**: $10-20/month (2GB RAM, 1 vCPU) - Basic tier

**Setup Requirements**:
- Python 3.10+ installed
- Systemd for service management
- Cron for scheduled jobs (nightly metrics, backups)
- Log rotation configured (logrotate)
- Firewall configured (UFW)
- Monitoring: Basic health checks (systemd, `/healthz` endpoint)

### FTMO Account Purchase Timing

**Recommended Timeline**:
- **Purchase Timing**: **After Phase 2.3 (Execution Model)** and **Before Phase 7 (Micro-Lot Testing)**
- **Rationale**: 
  - Need execution model ready to measure real spreads/slippage
  - Want to measure real FTMO execution before micro-lot testing
  - Allows time for account verification and setup (1-2 days)
  - Can start with demo account for Phase 2.3, upgrade to live for Phase 7

**Account Options**:
- **Demo Account**: Free (for Phase 2.3 execution metrics)
- **Challenge Account**: $155-500 (depending on account size: 10K, 25K, 50K, 100K)
- **Recommendation**: Start with 10K demo for testing, purchase 10K challenge for micro-lot phase

**Account Setup Checklist**:
- [ ] Create FTMO account
- [ ] Complete verification (KYC)
- [ ] Purchase challenge (Phase 7 timing)
- [ ] Configure MT5 credentials
- [ ] Test connection (Phase 4 MT5 bridge)
- [ ] Set up read-only credentials for execution metrics (Phase 2.3)

### Silent Observation Period

**Timing**: **After Phase 4 (Runtime + Telegram) setup, before Phase 7 (Micro-Lot Testing)**

**Duration**: 3-5 days of silent observation

**Purpose**:
- Monitor system stability without risk
- Verify signal generation quality
- Check FTMO rule enforcement
- Validate Telegram notifications
- Ensure no unexpected errors or crashes
- Confirm data pipeline reliability
- Validate execution model accuracy

**Activities During Observation**:
- [ ] Run runtime in dry-run mode (no actual trades)
- [ ] Log all signals that would be generated
- [ ] Track confidence scores and tier assignments
- [ ] Monitor FTMO rule checks (daily/total loss limits)
- [ ] Verify Telegram bot responsiveness
- [ ] Check structured logging output
- [ ] Validate execution cost calculations
- [ ] Monitor system resource usage (CPU, memory, disk)
- [ ] Review logs for any errors or warnings

**Success Criteria**:
- ✅ Zero crashes or unexpected errors
- ✅ All signals properly formatted and logged
- ✅ FTMO rules correctly enforced
- ✅ Telegram bot responsive and accurate
- ✅ System stable over 3-5 day period
- ✅ Ready to proceed to micro-lot testing

**Integration Point**:
- **MUST** happen after Phase 6 (Testing & Validation) completion
- **MUST** happen before Phase 7 (Micro-Lot Testing) starts
- Sequence: **P6 → P6.5 (3-5 days) → P7**
- No code changes needed (just monitoring)
- Allows confidence building before risking real capital

### Budget Summary

**One-Time Costs**:
- **FTMO Challenge**: $155-500 (10K-100K account size)
  - Recommendation: Start with 10K ($155) for micro-lot testing
- **GPU Training**: $15-40 (cloud rental, one-time for initial training)
  - Alternative: $0 if using local GPU
- **Development Tools**: $0 (open source stack)
- **Total One-Time**: **$170-540**

**Monthly Recurring Costs**:
- **VPS**: $12-24/month (recommended: Hetzner €8-16 or DigitalOcean $12)
- **Database**: $0 (SQLite for Phase 2-5, PostgreSQL optional later)
- **Cloud Storage**: $0-5/month (optional: S3 for backups, DVC)
- **Monitoring**: $0 (self-hosted logs, Telegram for alerts)
- **Total Monthly**: **$12-29/month**

**Optional Costs** (Phase 9+):
- **Additional GPU Training**: $15-40 per retraining cycle (weekly/monthly)
- **PostgreSQL Database**: $0-10/month (if migrating from SQLite)
- **Cloud Monitoring**: $0-20/month (optional: Datadog, Grafana Cloud)

**Total Estimated Budget**:
- **Initial Setup**: $170-540 (FTMO + GPU training)
- **Monthly Operations**: $12-148/month (depends on sentiment API choice)
  - **Lean** (no sentiment): $12-29/month
  - **Standard** (Reddit free): $12-29/month
  - **Premium** (Twitter $100/mo): $112-148/month
- **Year 1 Total**: $314-2,316
  - **Lean**: $314-888/year
  - **Standard**: $314-888/year
  - **Premium**: $1,370-2,316/year

**Cost Optimization Tips**:
- Use SQLite instead of PostgreSQL initially (Phase 2-5)
- Use free tier cloud storage (GitHub LFS, Git LFS)
- Use demo FTMO account for Phase 2.3 execution metrics
- Rent GPU only for training (not 24/7)
- Use cost-effective VPS providers (Hetzner, DigitalOcean)
- Self-host monitoring (no external services)

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
  - **Parallel Processing**: Fetch multiple symbols in parallel using `asyncio` or `concurrent.futures`
- [ ] Create data cleaning and validation pipeline
  - **Parallel Processing**: Process multiple symbols/periods in parallel where possible
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
- [ ] **Parallel Processing**: Engineer features for multiple symbols in parallel
- [ ] **Feature store-lite**: Versioned parquet for engineered features
  - Store in `data/features/features_YYYY-MM-DD.parquet`
  - Tie to DVC remote for versioning
  - Avoid recompute drift between training and backtest runs
  - Enable fast reload without recomputation

### 2.3 Empirical FTMO Execution Model (CRITICAL)
**Do NOT hardcode 12bps spread + 3bps slippage globally**

**Note**: Can start with **FTMO demo account** (free) for initial measurements. Upgrade to live challenge account before Phase 7.

- [ ] Create execution measurement system:
  - Measure spreads/slippage per pair (BTC/ETH/BNB)
  - Measure per session (Asia/EU/US)
  - Track distributions (mean, p50, p90)
  - **Use FTMO demo account initially** (free, no purchase needed yet)
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

**⚠️ IMPORTANT**: Phase 3 trains models on single-timeframe (1m) features. Phase 3.5 (V2) adds multi-timeframe features and retrains models.

### 3.1 LSTM Model
- [ ] Direction prediction model (15-min horizon)
- [ ] Per-coin training (BTC, ETH, BNB)
- [ ] **GPU Training**: Use cloud GPU rental (RunPod/Vast.ai) or local GPU
  - Estimated time: 2-4 hours per coin = 6-12 hours total
  - GPU cost: ~$5-15 (one-time)
- [ ] Training loop with walk-forward validation splits
- [ ] Save/load model weights with experiment tracking
- [ ] Model versioning by hash + params

### 3.2 Transformer Model
- [ ] Trend strength prediction
- [ ] Multi-coin training capability
- [ ] **GPU Training**: Use cloud GPU rental (recommend A100 for Transformer)
  - Estimated time: 4-8 hours total
  - GPU cost: ~$10-25 (one-time)
- [ ] Attention mechanism implementation
- [ ] Experiment tracking integration

### 3.3 Evaluation Framework
- [ ] Backtest engine with empirical FTMO execution model
- [ ] **Parallel Processing**: Run backtests across multiple time periods in parallel
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

## Phase 3.5: Multi-Timeframe Features + Model Retraining (V2) (2-3d) 🔄 **OPTIONAL V2**

**Timing**: After Phase 3 completion, before Phase 4 (if using Option A - Full Integration Pre-Go-Live)

**Purpose**: Add multi-timeframe features to improve model accuracy and retrain models.

**⚠️ Note**: This is a V2 enhancement. Can be skipped if using Option B (V1 go-live first).

### 3.5.1 Multi-Timeframe Feature Engineering
- [ ] Fetch data for multiple timeframes (1m, 5m, 15m, 1h, 4h)
- [ ] Engineer features for each timeframe
- [ ] Create multi-TF feature aggregation:
  - Higher timeframe trend signals
  - Multi-TF momentum alignment
  - Timeframe convergence/divergence
- [ ] Update feature store with multi-TF features

### 3.5.2 Model Retraining
- [ ] Retrain LSTM models with multi-TF features
- [ ] Retrain Transformer models with multi-TF features
- [ ] Validate improvement (should see ≥2% accuracy improvement)
- [ ] Update model versions

**Success Criteria**:
- ✅ Multi-TF features integrated
- ✅ Models retrained and validated
- ✅ Accuracy improvement confirmed (≥2% vs single-TF baseline)
- ✅ Model versions updated

---

## Phase 4: Runtime + Telegram (Dry-Run) (0.5-1d) ⬆️ **MOVED UP**

**Rationale**: Get dry-run signals early to observe formatting, thresholds, and rule enforcement during model iteration.

**⚠️ IMPORTANT**: This phase creates runtime scaffolding with mock/stub models. Real validated models (from Phase 3) are integrated post-Phase 7 (after micro-lot validation). This allows:
- Early feedback on signal formatting and Telegram UX
- Testing FTMO rules enforcement without risk
- Iterating on confidence thresholds before models are ready
- Full integration happens in Phase 8 (Go-Live) once models pass all gates

**VPS Setup**:
- Deploy to VPS (see Infrastructure & Resource Requirements for specs)
- Configure systemd service
- Set up log rotation and monitoring

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

### 4.4 Kafka Integration (V2 - Optional)
- [ ] **Kafka Topics**: `signals`, `trades`, `metrics`, `sentiment`
- [ ] Producer for runtime events
- [ ] Consumer for downstream processing (if needed)
- [ ] **Kafka Lag Monitoring**:
  - Consumer group lag metric (`kafka_lag_ms`)
  - Alert if lag >1s for 5+ minutes
  - `/healthz` includes Kafka broker status
  - Structured logs include `kafka_lag_ms`
- [ ] Self-hosted Kafka (Docker) - $0 cost

### 4.5 MT5 Bridge Interface (Pre-Work)
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
  - **With RL**: 30% LSTM, 35% Transformer, 25% RL, 10% Sentiment
  - **Without RL** (if Phase 9 skipped): 50% LSTM, 50% Transformer
  - Configurable via `ENSEMBLE_WEIGHTS` env var or config file
- [ ] Conservative bias application (-5%)
- [ ] **Platt/Isotonic scaling** option if calibration error >5%
- [ ] **Tier hysteresis**: require 2 consecutive scans > threshold to upgrade tier
- [ ] **Per-pattern sample floor**: cap influence from <N observations
- [ ] Historical pattern adjustment
- [ ] **FREE Boosters** (V2):
  - Multi-timeframe alignment bonus
  - Session timing boost (Tokyo/London/NY optimal times)
  - Volatility regime adjustment

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

## Phase 6.5: Silent Observation Period (3-5 days) 🔍 **MANDATORY**

**Timing**: **After Phase 6 (Testing & Validation) completion, BEFORE Phase 7 (Micro-Lot Testing)**

**⚠️ CRITICAL**: This period MUST happen after all tests pass and BEFORE micro-lot testing begins.

**Purpose**: Monitor system stability and signal quality without risk before risking real capital.

**Activities**:
- [ ] Run runtime in dry-run mode for 3-5 days
- [ ] Log all signals that would be generated
- [ ] Monitor FTMO rule enforcement
- [ ] Verify Telegram bot responsiveness
- [ ] Review structured logs for errors
- [ ] Validate execution cost calculations
- [ ] Monitor system resource usage
- [ ] **FTMO Account**: Ensure FTMO account is purchased and configured by end of this period

**Success Criteria**:
- ✅ Zero crashes or unexpected errors
- ✅ All signals properly formatted
- ✅ FTMO rules correctly enforced
- ✅ System stable over observation period
- ✅ Ready to proceed to micro-lot testing

---

## Phase 7: Micro-Lot Testing & Calibration (2-3d)

**Prerequisites**:
- ✅ Silent observation period completed
- ✅ FTMO challenge account purchased and configured
- ✅ System validated and stable

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

## Phase 9: Optional - V2 Enhancements (Sentiment, RL Agent) (2-4d) ⚠️ **OPTIONAL**

**Only proceed if V1 is stable and you want additional edge. These can introduce instability.**

### 9.1 Sentiment Integration (V2)

**Sentiment Source Decision**:
Choose ONE:
1. **Reddit (FREE, recommended)** ⭐
   - Use `praw` library
   - Subreddits: r/cryptocurrency, r/bitcoin
   - 60 req/min (free tier)
   - Cost: $0/month
2. **Twitter ($100/mo)**
   - Better influencer coverage, but paid
   - Requires Twitter API v2 access
   - Cost: ~$100/month
3. **CryptoPanic (FREE)**
   - Simple news headline feed
   - Cost: $0/month

**Recommendation**: Start with Reddit → add Twitter later if budget allows.

**Implementation**:
- [ ] Integrate sentiment API (Reddit/Twitter/CryptoPanic)
- [ ] Store daily sentiment scores (rolling 1h mean)
- [ ] Apply sentiment filter before ensemble weighting
- [ ] Add sentiment score to structured logs
- [ ] Update ensemble weights to include sentiment (10% weight if enabled)

### 9.2 RL Agent (V2)

### 9.2.1 RL Environment
- [ ] Gymnasium environment for trading
- [ ] Action space (buy, sell, hold)
- [ ] Observation space with market features
- [ ] Reward function design (avoid leakage)
- [ ] PPO implementation with execution model
- [ ] Gradient clipping and stability checks

### 9.2.2 RL Validation Requirements
- [ ] **Require reward curve stability**
- [ ] **Out-of-sample improvement ≥2% over supervised ensemble**
- [ ] If not meeting requirements, keep as research branch only
- [ ] Do not integrate into runtime until validated

### 9.2.3 RL Agent Validation & Fallback Strategy
**If RL validates (≥2% OOS improvement)**:
- ✅ Ensemble: 30% LSTM, 35% Transformer, 25% RL, 10% Sentiment
- ✅ Proceed to production
- ✅ Update model registry

**If RL fails (<2% improvement)**:
- ❌ Keep as research branch
- ❌ Fall back to: 50% LSTM, 50% Transformer (or 30% LSTM, 35% Transformer, 35% Sentiment if sentiment enabled)
- ❌ Document results in `models/rl_failure_report.md`
- ✅ Continue operations with no delay

### 9.3 GAN for Synthetic Data (Optional)
- [ ] Generator and discriminator networks
- [ ] Synthetic data generation (capped at ≤10-20% of dataset)
- [ ] **Always tag synthetic rows** for tracking
- [ ] Quality checks and sanity validation
- [ ] **Ablation tests**: run with 0% synth to prove it helps
- [ ] Watch for artifacts and regime hallucinations

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

### V2 Integration Strategy
**Decision Required**: Choose integration path

**Option A (Recommended - Full Integration Pre-Go-Live)**:
- P3.5: Multi-TF features added + retrain models
- P4: Runtime (with Kafka if enabled)
- P5: Confidence (with FREE boosters)
- P6: Tests (validate V2 features)
- P6.5: Silent Observation (monitor V2 features)
- P7: Micro-lot (validate V2 impact)
- P8: Go-Live (V2 features proven)
- P9: RL agent (optional post-go-live enhancement)

**Option B (Safer - V1 Go-Live First, Add V2 Later)**:
- P4-P8: V1 only (no multi-TF, no sentiment, no Kafka)
- P8: Go-Live V1
- P9: Add V2 features incrementally after validation

**Recommendation**: Go with **Option A** to reach a unified system faster, unless risk or resource limits justify Option B.

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
| **P3.5: Multi-TF (V2)** | **2-3d** | 🔄 **Optional V2** |
| P4: Runtime + Telegram | 0.5-1d | - |
| P5: Confidence + Rules + DB | 0.5-1d | - |
| P6: Tests/Validation | 1d | - |
| P6.5: Silent Observation | 3-5d | 🔍 **Mandatory** |
| P7: Micro-lot + Calibration | 2-3d | - |
| P8: Go-Live | 1d | ⚠️ Extended (+0.5d) |
| **Buffer (data/API friction)** | **+2d** | - |
| **V1 Total** | **14.5-18.5d** | ⚠️ Updated (includes observation period) |
| **V1 + V2 (Option A)** | **27.5-41.5d** | 🔄 Includes P3.5 and V2 features |
| P9: Sentiment + RL (optional) | 2-4d | - |
| **With All Optional** | **29.5-45.5d** | - |

---

## Next Steps

1. Review and confirm this plan
2. Set up infrastructure (Phase 1)
3. Begin with data pipeline and empirical execution model (Phase 2)
4. Iterate with early runtime feedback (Phase 4 after Phase 3)

Ready to proceed when you are!


