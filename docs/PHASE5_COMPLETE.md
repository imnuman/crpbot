# Phase 5: Confidence System + Database - Implementation Complete

## ✅ Completed Components

### Phase 5.1: Enhanced Confidence Scoring ✅
- **File**: `libs/confidence/enhanced.py`
- **Features**:
  - Enhanced confidence scorer with calibration support
  - Tier hysteresis (require 2 consecutive scans > threshold)
  - FREE boosters (multi-TF alignment, session timing, volatility regime)
  - Per-pattern sample floor (via database integration)
  - Historical pattern adjustment
  - Ensemble weights with fallback (50/50 LSTM/Transformer if no RL)

- **Platt/Isotonic Scaling**: `libs/confidence/scaling.py`
  - Platt scaling (logistic regression)
  - Isotonic scaling (non-parametric)
  - Calibration error calculation (ECE)
  - Auto-apply scaling if calibration error > 5%

### Phase 5.2: FTMO Rules Library ✅
- **Status**: Already implemented in Phase 4
- **File**: `apps/runtime/ftmo_rules.py`
- **Features**: Daily/total loss limits, position sizing, state tracking

### Phase 5.3: Database & Auto-Learning ✅
- **Database Models**: `libs/db/models.py`
  - `Pattern` table: Pattern tracking with win rates
  - `RiskBookSnapshot` table: Trade tracking with full metrics
  - `ModelDeployment` table: Model versioning and rollback tracking

- **Database Connection**: `libs/db/database.py`
  - SQLAlchemy-based database management
  - SQLite support (default) and PostgreSQL support
  - Session management and transaction handling

- **Auto-Learning System**: `libs/db/auto_learning.py`
  - Pattern tracking and result recording
  - Pattern win rate retrieval
  - Trade recording and result updating
  - Statistics calculation
  - Sample floor enforcement

- **Retention & Archival**: `libs/db/retention.py`
  - Dry-run trade archival (after 90 days)
  - Export to Parquet for long-term analysis
  - Monthly archival support

## 📊 Database Schema

### Patterns Table
- `id`: Primary key
- `name`: Pattern name
- `pattern_hash`: Unique pattern hash (SHA256)
- `wins`: Number of wins
- `total`: Total occurrences
- `win_rate`: Calculated win rate
- `created_at`, `updated_at`: Timestamps

### Risk Book Snapshots Table
- `signal_id`: Primary key (unique signal ID)
- `pair`: Trading pair
- `tier`: Confidence tier (high/medium/low)
- `entry_time`, `entry_price`: Entry details
- `tp_price`, `sl_price`: Take-profit and stop-loss
- `rr_expected`: Expected risk:reward ratio
- `result`: Trade result (win/loss/null)
- `exit_time`, `exit_price`: Exit details
- `r_realized`: Realized R
- `time_to_tp_sl_seconds`: Time to TP/SL
- `slippage_bps`, `spread_bps`: Execution metrics
- `latency_ms`: Decision latency
- `mode`: Mode tag (dryrun/live)

### Model Deployments Table
- `id`: Primary key
- `version`: Model version (unique)
- `model_path`: Path to model file
- `model_type`: Model type (lstm/transformer/ensemble)
- `deployed_at`: Deployment timestamp
- `metrics_json`: Deployment metrics (JSON)
- `rollback_reason`: Rollback reason (nullable)
- `is_promoted`: Promotion status
- `is_active`: Active status

## 🔧 Features

### Enhanced Confidence Scoring
- ✅ Ensemble weighting (LSTM + Transformer + RL + Sentiment)
- ✅ Conservative bias (-5%)
- ✅ Platt/Isotonic scaling (if calibration error > 5%)
- ✅ Tier hysteresis (2 consecutive scans)
- ✅ FREE boosters (multi-TF, session timing, volatility)
- ✅ Pattern win rate adjustment (with sample floor)

### Auto-Learning
- ✅ Pattern tracking (hash-based)
- ✅ Result recording (win/loss)
- ✅ Win rate calculation
- ✅ Sample floor enforcement
- ✅ Historical pattern adjustment

### Database
- ✅ SQLite support (default, for development)
- ✅ PostgreSQL support (for production)
- ✅ Retention policy (dry-run: 90 days, live: indefinite)
- ✅ Backup strategy (daily backups, S3 support)
- ✅ Export to Parquet (for analysis)

## 📝 Usage

### Initialize Database
```bash
python scripts/init_database.py
```

### Test Database
```bash
python scripts/test_database.py
```

### Backup Database
```bash
# Local backup
bash infra/scripts/backup_db.sh

# S3 backup (if configured)
S3_BUCKET=your-bucket bash infra/scripts/backup_db.sh
```

### Use Enhanced Confidence Scorer
```python
from libs.confidence.enhanced import EnhancedConfidenceScorer
from datetime import datetime

scorer = EnhancedConfidenceScorer(
    ensemble_weights={"lstm": 0.35, "transformer": 0.40, "rl": 0.25},
    enable_calibration=True,
    enable_hysteresis=True,
    enable_boosters=True,
)

confidence = scorer.score(
    lstm_pred=0.72,
    transformer_pred=0.70,
    rl_pred=0.0,
    timestamp=datetime.now(),
    volatility_regime="medium",
)

tier = scorer.get_tier(confidence)
```

### Use Auto-Learning System
```python
from libs.db.auto_learning import AutoLearningSystem

auto_learning = AutoLearningSystem()

# Record pattern result
features = {"feature1": 0.5, "feature2": 0.7}
auto_learning.record_pattern_result(features, "win", "my_pattern")

# Get pattern win rate
win_rate, sample_count = auto_learning.get_pattern_win_rate(features)

# Record trade
auto_learning.record_trade(
    signal_id="signal_001",
    pair="BTC-USD",
    tier="high",
    entry_time=datetime.now(),
    entry_price=50000.0,
    tp_price=51000.0,
    sl_price=49000.0,
    rr_expected=2.0,
    mode="dryrun",
)
```

## 🎯 Integration Points

### Runtime Integration
- Enhanced confidence scorer can be used in runtime loop
- Auto-learning system records patterns and trades
- Database tracks all trades and patterns

### Model Training
- Calibration models can be fitted during training
- Pattern win rates can be used for confidence adjustment

### Telegram Bot
- Statistics from database can be displayed in `/stats` command
- FTMO status can include database-backed metrics

## 📊 Retention Policy

- **Risk Book (Live)**: Keep all trades indefinitely
- **Risk Book (Dry-run)**: Archive after 90 days
- **Patterns**: Keep all (small dataset)
- **Model Deployments**: Keep all (audit trail)
- **Monthly Archival**: Export to Parquet for long-term analysis

## 🔧 Configuration

### Database URL
```bash
# SQLite (default)
DB_URL=sqlite:///tradingai.db

# PostgreSQL
DB_URL=postgresql+psycopg://user:pass@localhost:5432/tradingai
```

### Ensemble Weights
```bash
# Default: 35% LSTM, 40% Transformer, 25% RL
ENSEMBLE_WEIGHTS=0.35,0.40,0.25

# Without RL: 50% LSTM, 50% Transformer
ENSEMBLE_WEIGHTS=0.5,0.5,0.0
```

## ⚠️ Important Notes

1. **Database**: SQLite is used by default for development. For production, use PostgreSQL.

2. **Calibration**: Platt/Isotonic scaling requires scikit-learn. Calibration models are fitted during training/validation.

3. **Pattern Tracking**: Patterns are hashed from features. Ensure features are consistent across runs.

4. **Retention**: Dry-run trades are automatically archived after 90 days. Live trades are kept indefinitely.

## ✅ Phase 5 Status: COMPLETE

All Phase 5 components have been implemented and tested. The enhanced confidence system and database are ready for integration with the runtime.

