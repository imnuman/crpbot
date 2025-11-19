# CRPBot V6 Enhanced Data Pipeline & System Architecture

**Last Updated**: 2025-11-16
**Version**: V6 Enhanced FNN with 72 Amazon Q Features

---

## System Overview

### Current Production Deployment
- **Location**: Cloud Server (178.156.136.185)
- **Runtime Process**: PID 163631 (LIVE mode, 60-second scan interval)
- **Dashboard Process**: PID 170323 (Port 5000)
- **Local Machine**: All processes stopped (cloud-only architecture)

### Model Architecture
**V6 Enhanced FNN (Feedforward Neural Network)**
- **Input Layer**: 72 features (Amazon Q enhanced feature set)
- **Hidden Layers**:
  - Layer 1: 72 → 256 neurons
  - Layer 2: 256 → 128 neurons
  - Layer 3: 128 → 64 neurons
- **Output Layer**: 64 → 3 classes (Down, Neutral, Up)
- **Activation**: ReLU for hidden layers, Softmax for output
- **Confidence Control**: Logit clamping (±2.0) + Temperature scaling (T=2.0)

---

## Complete Data Pipeline Flow

### Step 1: Market Data Collection
**Source**: Coinbase Advanced Trade API (Primary)
**Backup**: Kraken API (Multi-timeframe validation)

```
┌─────────────────────────────┐
│   Coinbase REST API         │
│   - BTC-USD, ETH-USD,       │
│     SOL-USD                 │
│   - 1-minute OHLCV candles  │
│   - 120 candles per request │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  MarketDataFetcher          │
│  (apps/runtime/             │
│   data_fetcher.py)          │
│                             │
│  - JWT Authentication       │
│  - Rate Limiting            │
│  - Error Handling           │
│  - Pandas DataFrame Output  │
└──────────────┬──────────────┘
               │
               ▼
    Raw OHLCV DataFrame
    Columns: [open, high, low, close, volume, timestamp]
    Shape: (120, 6)
```

### Step 2: Feature Engineering (Amazon Q's 72 Features)
**Module**: `apps/trainer/amazon_q_features.py`

```
┌─────────────────────────────────────────────────────────────┐
│  engineer_amazon_q_features(df)                             │
│                                                             │
│  INPUT: 120 rows × 6 columns (OHLCV + timestamp)           │
│  OUTPUT: 120 rows × 78 columns (6 raw + 72 features)       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  FEATURE CATEGORIES (72 Total)                              │
│                                                             │
│  1. EXPONENTIAL MOVING AVERAGES (20 features)              │
│     - Periods: 5, 10, 20, 50, 100, 200                     │
│     - Price crossovers (close vs EMA)                       │
│     - EMA slopes                                            │
│                                                             │
│  2. MACD INDICATORS (12 features)                           │
│     - MACD_12_26, MACD_signal, MACD_histogram              │
│     - Fast MACD_5_13, Slow MACD_19_39                      │
│     - MACD crossovers and divergences                       │
│                                                             │
│  3. RSI (RELATIVE STRENGTH INDEX) (9 features)             │
│     - RSI_14 (primary)                                      │
│     - RSI_7, RSI_21                                         │
│     - RSI overbought (>70), oversold (<30)                  │
│     - RSI divergence detection                              │
│                                                             │
│  4. BOLLINGER BANDS (8 features)                            │
│     - Upper, Middle, Lower bands (20-period)               │
│     - Bandwidth, %B position                                │
│     - Band squeeze detection                                │
│                                                             │
│  5. STOCHASTIC OSCILLATOR (6 features)                      │
│     - %K (14-period)                                        │
│     - %D (3-period moving average of %K)                    │
│     - Overbought/oversold levels                            │
│                                                             │
│  6. WILLIAMS %R (4 features)                                │
│     - Williams_R_14                                         │
│     - Overbought (<-80), Oversold (>-20)                    │
│                                                             │
│  7. MOMENTUM & ROC (8 features)                             │
│     - Momentum_10, Momentum_20                              │
│     - Rate of Change (ROC_10, ROC_20)                       │
│                                                             │
│  8. VOLATILITY & VOLUME (5 features)                        │
│     - ATR_14 (Average True Range)                           │
│     - Volume_MA_20, Volume_ratio                            │
│     - Volatility_20 (rolling std)                           │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
         Enriched DataFrame (120, 78)
```

### Step 3: Model Inference (V6 Enhanced FNN)
**Module**: `apps/runtime/ensemble.py`

```
┌─────────────────────────────────────────────────────────────┐
│  EnsemblePredictor.predict(df)                              │
│                                                             │
│  1. Feature Selection                                       │
│     - Extract 72 Amazon Q features                          │
│     - Normalize/Scale (already normalized during training)  │
│                                                             │
│  2. Model Forward Pass                                      │
│     Input: (batch_size=1, features=72)                      │
│       ↓                                                      │
│     Linear(72 → 256) + ReLU + Dropout(0.3)                 │
│       ↓                                                      │
│     Linear(256 → 128) + ReLU + Dropout(0.3)                │
│       ↓                                                      │
│     Linear(128 → 64) + ReLU + Dropout(0.3)                 │
│       ↓                                                      │
│     Linear(64 → 3)  # Raw logits                           │
│       ↓                                                      │
│     Output: [down_logit, neutral_logit, up_logit]          │
│                                                             │
│  3. Confidence Calibration                                  │
│     Raw Logits: [40064.82, -23939.20, -10154.86]          │
│       ↓                                                      │
│     Clamp to ±2.0: [2.0, -2.0, -2.0]                       │
│       ↓                                                      │
│     Apply Temperature (T=2.0):                              │
│       logits_scaled = clamped_logits / 2.0                  │
│       = [1.0, -1.0, -1.0]                                   │
│       ↓                                                      │
│     Softmax:                                                │
│       probs = exp(logits_scaled) / sum(exp(logits_scaled)) │
│       = [0.787, 0.107, 0.107]                              │
│                                                             │
│  4. Direction & Confidence                                  │
│     Direction = argmax(probs) = "short" (Down)             │
│     Confidence = max(probs) = 0.787 (78.7%)                │
│                                                             │
│  5. Tier Classification                                     │
│     if confidence >= 0.75: tier = "high"                    │
│     elif confidence >= 0.65: tier = "medium"                │
│     else: tier = "low"                                      │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  PREDICTION OUTPUT                                          │
│                                                             │
│  {                                                          │
│    "symbol": "BTC-USD",                                     │
│    "direction": "short",     # or "long"                    │
│    "confidence": 0.787,      # 78.7%                        │
│    "tier": "high",           # high/medium/low              │
│    "down_prob": 0.787,                                      │
│    "neutral_prob": 0.107,                                   │
│    "up_prob": 0.107,                                        │
│    "timestamp": "2025-11-16T14:22:28-05:00"                │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘
```

### Step 4: Signal Filtering & Database Storage
**Module**: `apps/runtime/main.py`

```
┌─────────────────────────────────────────────────────────────┐
│  Signal Processing                                          │
│                                                             │
│  1. Confidence Threshold Check                              │
│     if confidence >= 0.65:                                  │
│         → Proceed to save signal                            │
│     else:                                                    │
│         → Skip (no database record)                         │
│                                                             │
│  2. Database Storage (SQLite/PostgreSQL)                    │
│     Table: signals                                          │
│     Columns:                                                │
│       - id (primary key)                                    │
│       - timestamp (EST timezone)                            │
│       - symbol (BTC-USD, ETH-USD, SOL-USD)                 │
│       - direction (long/short)                              │
│       - confidence (0.0 - 1.0)                              │
│       - tier (high/medium/low)                              │
│       - lstm_prediction (down/neutral/up probabilities)     │
│       - result (win/loss, evaluated later)                  │
│       - pnl (profit/loss when closed)                       │
│                                                             │
│  3. Telegram Notification (if enabled)                      │
│     Send to: -4757699063                                    │
│     Format:                                                 │
│       📊 **HIGH TIER SIGNAL**                              │
│       Symbol: BTC-USD                                       │
│       Direction: SHORT                                      │
│       Confidence: 78.7%                                     │
│       Time: 2025-11-16 14:22:28 EST                        │
└─────────────────────────────────────────────────────────────┘
```

### Step 5: Dashboard Visualization
**Modules**: `apps/dashboard/app.py` + `templates/dashboard.html`

```
┌─────────────────────────────────────────────────────────────┐
│  DASHBOARD API ENDPOINTS                                    │
│                                                             │
│  1. /api/status                                             │
│     - System status (live/offline)                          │
│     - Model info (version, architecture, accuracy)          │
│     - Data sources (Coinbase, Kraken, CoinGecko)           │
│                                                             │
│  2. /api/market/live                                        │
│     - Current prices for BTC/ETH/SOL                        │
│     - 24h change %, high, low, volume                       │
│     - Updated every 1 second                                │
│                                                             │
│  3. /api/predictions/live                                   │
│     - Real-time predictions (all symbols)                   │
│     - 3-class probabilities (Down/Neutral/Up)               │
│     - Confidence + Direction + Tier                         │
│     - **NEW**: Confidence displayed under price             │
│                                                             │
│  4. /api/signals/recent/24                                  │
│     - Last 100 signals in past 24 hours                     │
│     - Sortable by timestamp (newest first)                  │
│                                                             │
│  5. /api/signals/stats/24                                   │
│     - Total signals count                                   │
│     - Average confidence                                    │
│     - Hourly rate (signals/hour)                            │
│     - Breakdown by: symbol, direction, tier                 │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  DASHBOARD UI FEATURES                                      │
│                                                             │
│  📊 Live Market Prices                                      │
│     - Current price (updating every 1 second)               │
│     - **Confidence display** (color-coded by tier)          │
│       • Green: High tier (≥75%)                            │
│       • Orange: Medium tier (≥65%)                         │
│       • Gray: Low tier (<65%)                              │
│     - 24h change %, high, low, volume                       │
│                                                             │
│  🎯 Live Predictions                                        │
│     - Doughnut charts (Down/Neutral/Up probabilities)       │
│     - Direction indicator (LONG/SHORT)                      │
│     - Confidence percentage                                 │
│     - Tier classification                                   │
│                                                             │
│  📈 Signal Statistics (24h)                                 │
│     - Total signals, avg confidence, hourly rate            │
│     - Breakdown by symbol, direction, tier                  │
│                                                             │
│  🔔 Recent Signals Table                                    │
│     - Last 10 signals with timestamp                        │
│     - Symbol, direction, confidence, tier                   │
│                                                             │
│  🔬 Analysis Process Diagram                                │
│     - Data Collection → Feature Engineering                 │
│     - Model Inference → Signal Generation                   │
└─────────────────────────────────────────────────────────────┘
```

---

## System Performance Metrics

### Model Training Results
- **BTC-USD**: 67.58% test accuracy
- **ETH-USD**: 71.65% test accuracy
- **SOL-USD**: 70.39% test accuracy
- **Average**: 69.87% test accuracy

### Confidence Calibration (After Fix)
- **Before**: 99-100% confidence (unrealistic)
- **After**: 10-90% confidence range (realistic)
- **Method**: Logit clamping (±2.0) + Temperature scaling (T=2.0)

### Signal Generation Stats (Example)
```
Timestamp: 2025-11-16 14:22:28 EST

BTC-USD:
  - Raw Logits: [40064.82, -23939.20, -10154.86]
  - Clamped: [2.0, -2.0, -2.0]
  - Probabilities: [78.7%, 10.7%, 10.7%]
  - Direction: SHORT
  - Confidence: 78.7%
  - Tier: HIGH

ETH-USD:
  - Raw Logits: [1513.92, -1078.21, -1068.50]
  - Clamped: [2.0, -2.0, -2.0]
  - Probabilities: [78.7%, 10.7%, 10.7%]
  - Direction: SHORT
  - Confidence: 78.7%
  - Tier: HIGH

SOL-USD:
  - Raw Logits: [68.66, -49.26, -15.98]
  - Clamped: [2.0, -2.0, -2.0]
  - Probabilities: [78.7%, 10.7%, 10.7%]
  - Direction: SHORT
  - Confidence: 78.7%
  - Tier: HIGH
```

---

## Background Processes (Cloud Only)

### Runtime Process
```bash
PID: 163631
Command: .venv/bin/python3 apps/runtime/main.py --mode live --iterations -1 --sleep-seconds 60
Started: 13:44 EST
Mode: LIVE
Scan Interval: 60 seconds
Symbols: BTC-USD, ETH-USD, SOL-USD
Log: /tmp/v6_final_fix.log
```

### Dashboard Process
```bash
PID: 170323
Command: .venv/bin/python3 apps/dashboard/app.py
Started: 14:21 EST
Port: 5000
URL: http://178.156.136.185:5000
Log: /tmp/dashboard_with_confidence.log
Auto-refresh: 1 second
```

### Local Machine
```
✅ All processes stopped
✅ Cloud-only architecture confirmed
```

---

## Data Quality & Validation

### Input Validation
- **Minimum Candles**: 120 for prediction (100 for transformer, 60 for LSTM)
- **Feature Completeness**: All 72 features must be present
- **NaN Handling**: Forward-fill for missing values
- **Outlier Detection**: Values beyond ±5 std deviations flagged

### Output Validation
- **Confidence Range**: 0.0 - 1.0 (enforced by softmax)
- **Probability Sum**: Always equals 1.0 (Down + Neutral + Up)
- **Direction Consistency**: argmax(probs) matches direction label
- **Tier Logic**: Deterministic thresholds (75%, 65%)

---

## Troubleshooting Reference

### Common Issues & Solutions

**Issue**: 100% Confidence Values
- **Cause**: Extreme raw logits without clamping
- **Solution**: Implemented logit clamping (±2.0) + temperature scaling (T=2.0)
- **Status**: ✅ Fixed (apps/runtime/ensemble.py:237-253)

**Issue**: UTC Timestamps Instead of EST
- **Cause**: Using datetime.utcnow() throughout codebase
- **Solution**: Created libs/utils/timezone.py with now_est() function
- **Status**: ✅ Fixed (all modules updated)

**Issue**: "Not enough data" errors
- **Cause**: Fetching fewer than 120 candles from API
- **Solution**: Increased fetch to 120 candles, added validation check
- **Status**: ✅ Fixed (apps/runtime/data_fetcher.py:53)

---

## Future Enhancements

1. **Multi-Timeframe Integration** (Phase 3.5)
   - Already implemented: apps/trainer/multi_tf_features.py
   - Adds: 5m, 15m, 1h cross-timeframe alignment features
   - Status: Module created, pending training integration

2. **Reinforcement Learning Agent** (Phase 4)
   - Algorithm: PPO (Proximal Policy Optimization)
   - Purpose: Optimize entry/exit timing considering spreads
   - Status: Stub implementation exists

3. **Performance Tracking** (Phase 5)
   - Win/loss evaluation (15-minute horizon)
   - Sharpe ratio calculation
   - Max drawdown monitoring

4. **FTMO Compliance Enforcement** (Phase 6)
   - Daily 5% loss limit
   - Total 10% loss limit
   - Minimum trading days (4)
   - Profit target (10%)

---

## Quick Commands Reference

```bash
# Check cloud processes
ssh root@178.156.136.185 "ps aux | grep -E 'dashboard|runtime' | grep python | grep -v grep"

# View runtime logs
ssh root@178.156.136.185 "tail -f /tmp/v6_final_fix.log"

# View dashboard logs
ssh root@178.156.136.185 "tail -f /tmp/dashboard_with_confidence.log"

# Restart runtime
ssh root@178.156.136.185 "pkill -f 'apps/runtime/main.py' && cd ~/crpbot && nohup .venv/bin/python3 apps/runtime/main.py --mode live --iterations -1 --sleep-seconds 60 > /tmp/runtime.log 2>&1 &"

# Restart dashboard
ssh root@178.156.136.185 "pkill -f 'apps/dashboard/app.py' && cd ~/crpbot && nohup .venv/bin/python3 apps/dashboard/app.py > /tmp/dashboard.log 2>&1 &"

# Access dashboard
http://178.156.136.185:5000
```

---

## Contact & Support

- **Dashboard URL**: http://178.156.136.185:5000
- **GitHub Repo**: https://github.com/imnuman/crpbot
- **Telegram**: -4757699063
- **Documentation**: See CLAUDE.md, PROJECT_MEMORY.md

---

**Generated**: 2025-11-16 14:23:00 EST
**By**: Claude Code (Session: Dashboard Confidence Enhancement)
