# Amazon Q - Project Status Summary
**Last Updated**: 2025-11-16 10:20 EST
**Project**: CRPBot V6 Rebuild - Multi-Source Trading AI

---

## 🎯 Current Mission

**Goal**: Rebuild V6 LSTM models with >68% accuracy using multi-source data and 155 features

**Status**: Phase 1 COMPLETE ✅ | Phase 2 READY TO START ⏳

---

## 🖥️ Environment Status

### Local Machine (`/home/numan/crpbot`)
- **Purpose**: Development, training, data collection
- **Git Branch**: `main` (commit: 952d16b)
- **Status**: ✅ Fully updated
- **Data Sources**: Coinbase ✅ | Kraken ✅ | CoinGecko ✅
- **Role**: Primary development environment

### Cloud Machine (`root@178.156.136.185:~/crpbot`)
- **Purpose**: Production runtime (currently broken, fixing later)
- **Git Branch**: `main` (commit: 952d16b)
- **Status**: ✅ Synced with local
- **Data Sources**: Coinbase ⚠️ (401 errors) | Kraken ✅ | CoinGecko ✅
- **Role**: Production deployment (after V6 training)

### Sync Status
- ✅ Code: Synced via Git (commit 952d16b)
- ✅ .env: All 3 API keys configured on both machines
- ✅ Dependencies: pycoingecko installed, CCXT available
- ✅ Scripts: V6 rebuild scripts deployed

---

## 📊 Data Sources Configuration

### 1. Coinbase Advanced Trade API
**Status**: ✅ Working locally | ❌ Broken on cloud (401 errors)

**Local Configuration**:
```
COINBASE_API_KEY_NAME=organizations/.../apiKeys/feb545c9-0580-4a7a-90f2-035bda8d3923
COINBASE_API_PRIVATE_KEY=-----BEGIN EC PRIVATE KEY-----...
```

**Capabilities**:
- Historical OHLCV (1m, 5m, 15m, 1h)
- Rate limit: 10 req/s
- High volume, accurate pricing
- **Role**: Primary OHLCV source

**Cloud Issue**:
- `apps/runtime/data_fetcher.py` uses basic REST without ECDSA authentication
- Needs JWT signing implementation
- **Fix scheduled**: After V6 training

---

### 2. Kraken API
**Status**: ✅ Public API working on both machines

**Configuration**:
```
KRAKEN_API_KEY=Jk/k4kllL3...
KRAKEN_API_SECRET=... (configured but permissions need fixing for private API)
```

**Capabilities**:
- Historical OHLCV (1m, 5m, 15m, 1h)
- Public API: 1 req/s
- Private API: 15-20 req/s (needs "Query Funds" permission)
- 1,368 markets available
- **Role**: Backup OHLCV + validation

**Note**: Public API sufficient for V6 data collection

---

### 3. CoinGecko Professional API
**Status**: ✅ Working on both machines

**Configuration**:
```
COINGECKO_API_KEY=CG-VQhq64e59sGxchtK8mRgdxXW
```

**Capabilities**:
- Market cap, volume, ATH data
- Rate limit: 500 calls/min
- Historical data: 7-365 days
- 10 fundamental features
- **Role**: Fundamental enhancement

**Test Results** (2025-11-16):
```
BTC-USD: ATH distance -24.4%, price $95,349
ETH-USD: ATH distance -35.9%, price $3,170
SOL-USD: ATH distance -52.3%, price $140
```

---

## 🔧 V6 Rebuild Infrastructure

### Phase 1: Infrastructure ✅ COMPLETE

**Completed Components**:

1. **Multi-Source Data Collection** ✅
   - Script: `scripts/collect_multi_source_data.py`
   - Sources: Coinbase + Kraken
   - Features: Quality comparison, gap detection, validation
   - Test: 7 days successful (168 candles each source)

2. **Enhanced Features Module** ✅
   - Script: `apps/trainer/enhanced_features.py`
   - Features Added: 72 advanced indicators
   - Categories: Momentum, volatility, price action, microstructure, entropy, trends
   - Test: Passed with 1000 sample candles

3. **CoinGecko Integration** ✅
   - Module: `apps/runtime/coingecko_fetcher.py`
   - Features: 10 fundamental indicators
   - Test: All 3 symbols working
   - Historical: 169 data points per 7-day fetch

4. **Environment Sync Tool** ✅
   - Script: `scripts/sync_environments.sh`
   - Commands: all, code, env, deps, v6, status, verify
   - Usage: `./scripts/sync_environments.sh all`

---

## 📈 Feature Count Breakdown

| Category | Features | Status |
|----------|----------|--------|
| Base OHLCV | 5 | ✅ |
| Session features | 7 | ✅ |
| Spread features | 4 | ✅ |
| Volume features | 3 | ✅ |
| Technical indicators | 20 | ✅ |
| Volatility regime | 4 | ✅ |
| Multi-timeframe | 30 | ✅ |
| **Enhanced features** | **72** | ✅ |
| **CoinGecko fundamentals** | **10** | ✅ |
| **TOTAL** | **155 features** | ✅ |

**Improvement over V5**: +368% (31 → 155 features)

---

## 📋 Next Steps - Phase 2

### Immediate Tasks (Ready to Execute)

**1. Collect 2 Years of Historical Data**
```bash
for symbol in BTC/USD ETH/USD SOL/USD; do
    uv run python scripts/collect_multi_source_data.py \
        --symbol $symbol \
        --timeframe 1m \
        --days 730 \
        --sources kraken coinbase \
        --output-dir data/v6_rebuild
done
```

**Expected Output**:
- 6 parquet files (3 symbols × 2 sources)
- ~1M candles per symbol
- ~100-150 MB per file
- Time: ~2 hours

**2. Engineer All Features**
```bash
uv run python scripts/v6_rebuild_feature_engineering.py \
    --input data/v6_rebuild/ \
    --output data/v6_features/
```

**Expected Output**:
- 3 feature files (BTC, ETH, SOL)
- 155 columns per file
- ~1M rows per file
- Time: ~30-45 minutes

**3. Train V6 Models**
```bash
for coin in BTC ETH SOL; do
    uv run python apps/trainer/main.py \
        --task lstm \
        --coin $coin \
        --epochs 20 \
        --features-dir data/v6_features/ \
        --model-version v6_rebuild
done
```

**Expected Output**:
- 3 LSTM models (BTC, ETH, SOL)
- Input: 155 features
- Architecture: 3-layer LSTM, hidden=128
- Target: >68% accuracy
- Time: ~2-3 hours (3 models on CPU)

---

## 🚨 Known Issues & Fixes

### Issue 1: Cloud Coinbase 401 Errors
**Status**: ⚠️ Deferred (not blocking V6 rebuild)

**Root Cause**: `apps/runtime/data_fetcher.py` doesn't implement ECDSA JWT authentication

**Impact**: Cloud runtime can't fetch live data

**Workaround**: Use Kraken API or train locally first

**Fix**: Implement ECDSA auth after V6 training succeeds

---

### Issue 2: Kraken Private API Permission Denied
**Status**: ⚠️ Not critical (public API sufficient)

**Root Cause**: API key missing "Query Funds" permission

**Impact**: Slower rate limits (1 req/s vs 15-20 req/s)

**Workaround**: Use public API (works fine, just slower)

**Fix**: Enable permissions on Kraken website if faster collection needed

---

## 🎯 Success Criteria

### V6 Rebuild Targets

| Metric | V5 Current | V6 Target | Improvement |
|--------|------------|-----------|-------------|
| Features | 31 | 155 | +368% |
| Data Sources | 1 | 3 | +200% |
| Accuracy | ~50% | >68% | +36% |
| Calibration Error | Unknown | <5% | N/A |
| Data Quality | Basic | Multi-source validated | High |

### Promotion Gates

Models must pass to reach production:
- ✅ Win rate ≥68% on test set
- ✅ Calibration error ≤5%
- ✅ Backtest Sharpe >1.0
- ✅ Max drawdown <15%

---

## 📁 Key Files Reference

### Documentation
- `V6_REBUILD_STATUS.md` - Detailed rebuild status
- `V6_DATA_SOURCE_STRATEGY.md` - Data source strategy
- `V6_REBUILD_PLAN.md` - Original rebuild plan
- `AMAZON_Q_STATUS.md` - This file (for Amazon Q)

### Scripts
- `scripts/collect_multi_source_data.py` - Multi-source data collector
- `scripts/v6_rebuild_feature_engineering.py` - Feature engineering pipeline
- `scripts/sync_environments.sh` - Environment sync tool
- `test_kraken_connection.py` - Kraken connectivity test
- `test_coingecko_integration.py` - CoinGecko integration test

### Code
- `apps/trainer/enhanced_features.py` - 72 advanced features
- `apps/runtime/coingecko_fetcher.py` - CoinGecko data fetcher
- `apps/runtime/multi_tf_fetcher.py` - Multi-timeframe data fetcher
- `apps/runtime/runtime_features.py` - Runtime feature engineering

---

## 🤖 For Amazon Q

**Current Context**:
- We are in V6 rebuild Phase 1 → Phase 2 transition
- All infrastructure is ready and tested
- 3 data sources configured and working (Coinbase, Kraken, CoinGecko)
- 155 features engineered and tested
- Both local and cloud machines are synced

**Immediate Goal**:
- Collect 2 years of OHLCV data from Coinbase + Kraken
- Add CoinGecko fundamental features
- Engineer all 155 features
- Train 3 LSTM models with >68% accuracy target

**Recommended Action**:
Start Phase 2 data collection with the command provided above.

**If You See This Document**:
You are working on the V6 rebuild project. The user (Numan) is rebuilding models to achieve >68% accuracy. All infrastructure is ready. Focus on data collection → feature engineering → training pipeline.

---

**Last Sync**: 2025-11-16 10:20 EST (commit 952d16b)
**Sync Command**: `./scripts/sync_environments.sh all`
