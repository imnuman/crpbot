# 📋 Builder Claude Instructions - V5 Execution

**Created**: 2025-11-15 15:00 EST (Toronto)
**Last Updated**: 2025-11-15 15:00 EST (Toronto)
**Author**: QC Claude (Local Machine)
**To**: Builder Claude (Cloud Server)
**Priority**: HIGH
**Status**: Ready to Execute

---

## 🚨 CRITICAL UPDATE: Pricing Correction

**IMPORTANT**: All previous V5 documentation stated Tardis.dev at $98/month.

**ACTUAL PRICING**: Tardis.dev minimum is **$300-350+/month** ($6000+ for enterprise)

**NEW STRATEGY**: CoinGecko Analyst at **$129/month** for Phase 1

### What Changed
- ❌ OLD: Tardis.dev Historical ($98) + 53 features (tick data + order book)
- ✅ NEW: CoinGecko Analyst ($129) + 40-50 features (OHLCV only)
- ✅ Budget: Phase 1 = $154/month (CoinGecko $129 + AWS $25)

---

## 🎯 Current Status (As of 2025-11-15 15:00 EST)

### ✅ Completed
1. CoinGecko API key obtained
2. API key configured in `.env` (local machine)
3. Config system updated (`libs/config/config.py`)
4. All V5 budget documents corrected
5. Pricing corrections committed to Git

### 📍 Your Environment
- **Location**: Cloud server (`~/crpbot`)
- **Branch**: `main` (pull latest changes first!)
- **CoinGecko API Key**: `CG-VQhq64e59sGxchtK8mRgdxXW`
- **Status**: Ready to start V5 Week 1

---

## 🚀 Immediate Next Actions (Week 1)

### Step 1: Sync with Latest Changes (FIRST!)

```bash
# On cloud server
cd ~/crpbot
git pull origin main

# Verify you have the latest files
ls -la V5_SIMPLE_PLAN.md  # Should show 2025-11-15 timestamp
ls -la PRICING_CORRECTION_2025-11-15.md  # Should exist

# Check commit history
git log --oneline -5
# Should see: "fix: correct V5 data provider pricing..."
```

### Step 2: Configure CoinGecko API Key

```bash
# Add to .env on cloud server
echo 'COINGECKO_API_KEY=CG-VQhq64e59sGxchtK8mRgdxXW' >> .env

# Verify it's there
grep COINGECKO .env
```

### Step 3: Create CoinGecko Data Fetcher Script

**File**: `scripts/fetch_coingecko_data.py`

**Requirements**:
- Download 2 years of OHLCV data
- Symbols: BTC-USD, ETH-USD, SOL-USD
- Intervals: 1m, 5m, 15m, 1h (multiple timeframes)
- Output: Parquet files in `data/raw/coingecko/`
- Error handling and rate limiting

**CoinGecko API Endpoints**:
- Base URL: `https://api.coingecko.com/api/v3`
- OHLC endpoint: `/coins/{id}/ohlc`
- Historical data: `/coins/{id}/market_chart/range`
- Rate limit: ~50 calls/minute (Analyst tier)

**Coin IDs** (CoinGecko format):
- BTC-USD: `bitcoin`
- ETH-USD: `ethereum`
- SOL-USD: `solana`

**Reference Implementation**:
```python
#!/usr/bin/env python3
"""
Fetch historical OHLCV data from CoinGecko API.

Usage:
    python scripts/fetch_coingecko_data.py --symbol BTC-USD --days 730
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# CoinGecko configuration
COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY')
COINGECKO_BASE_URL = 'https://api.coingecko.com/api/v3'

# Coin mapping
COIN_IDS = {
    'BTC-USD': 'bitcoin',
    'ETH-USD': 'ethereum',
    'SOL-USD': 'solana'
}

def fetch_historical_data(coin_id: str, days: int = 730) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from CoinGecko.

    Args:
        coin_id: CoinGecko coin ID (e.g., 'bitcoin')
        days: Number of days of history (default: 730 = 2 years)

    Returns:
        DataFrame with OHLCV data
    """
    # Implementation here
    # Use /coins/{id}/market_chart/range endpoint
    # Convert to OHLCV format
    # Return pandas DataFrame
    pass

def save_to_parquet(df: pd.DataFrame, symbol: str, interval: str):
    """Save DataFrame to parquet file."""
    output_dir = Path('data/raw/coingecko')
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f'{symbol}_{interval}_{datetime.now().strftime("%Y%m%d")}.parquet'
    output_path = output_dir / filename

    df.to_parquet(output_path, index=False)
    print(f"✅ Saved: {output_path} ({len(df)} rows)")

if __name__ == '__main__':
    # Implementation here
    pass
```

### Step 4: Download Data for All Symbols

```bash
# Run data fetcher for each symbol
python scripts/fetch_coingecko_data.py --symbol BTC-USD --days 730
python scripts/fetch_coingecko_data.py --symbol ETH-USD --days 730
python scripts/fetch_coingecko_data.py --symbol SOL-USD --days 730

# Verify downloads
ls -lh data/raw/coingecko/
# Should see 3 parquet files, each 10-50 MB
```

### Step 5: Validate Data Quality

```bash
# Create validation script
python scripts/validate_coingecko_data.py --symbol BTC-USD

# Check:
# - No missing timestamps
# - OHLCV values reasonable
# - 2 years of data present
# - Data quality vs existing Coinbase data
```

---

## 📅 Full 4-Week Timeline

### Week 1: Data Download & Validation (Days 1-7)
**Status**: CURRENT WEEK

**Tasks**:
- [x] Sync with latest Git changes
- [x] Configure CoinGecko API key
- [ ] Create `scripts/fetch_coingecko_data.py`
- [ ] Download 2 years OHLCV for BTC/ETH/SOL
- [ ] Create `scripts/validate_coingecko_data.py`
- [ ] Validate data quality
- [ ] Compare CoinGecko vs Coinbase data quality
- [ ] Document findings

**Expected Output**:
```
data/raw/coingecko/
├── BTC-USD_1m_20251115.parquet   (~30-50 MB, 730 days)
├── ETH-USD_1m_20251115.parquet   (~30-50 MB, 730 days)
└── SOL-USD_1m_20251115.parquet   (~20-40 MB, 730 days)
```

**Success Criteria**:
- ✅ All 3 symbols downloaded
- ✅ 2 years of data per symbol
- ✅ No gaps or missing data
- ✅ Data quality better than free Coinbase

---

### Week 2: Feature Engineering (Days 8-14)

**Tasks**:
- [ ] Update `apps/trainer/features.py` for OHLCV-based features
- [ ] Engineer 40-50 features (NOT 53 - no microstructure)
- [ ] Create multi-timeframe features (1m, 5m, 15m, 1h)
- [ ] Run feature engineering for all symbols
- [ ] Validate feature distributions
- [ ] Test baseline accuracy (expect 55-60% vs current 50%)

**Feature Categories** (40-50 total):
1. **Existing (31 features)**: Reuse from V4
   - Session features (5)
   - Spread features (4)
   - Volume features (3)
   - Moving averages (8)
   - Technical indicators (8)
   - Volatility regime (3)

2. **New OHLCV-based (9-19 features)**:
   - Multi-timeframe price ratios (4-6)
   - Cross-timeframe momentum (3-5)
   - Volume-weighted features (2-4)
   - Advanced volatility measures (2-4)

**Expected Output**:
```
data/features/coingecko/
├── features_BTC-USD_1m_coingecko.parquet
├── features_ETH-USD_1m_coingecko.parquet
└── features_SOL-USD_1m_coingecko.parquet
```

**Success Criteria**:
- ✅ 40-50 features engineered per symbol
- ✅ No NaN values
- ✅ Feature distributions look reasonable
- ✅ Baseline test shows >50% accuracy

---

### Week 3: Model Training (Days 15-21)

**Tasks**:
- [ ] Update model architecture for new features
- [ ] Train LSTM models for BTC/ETH/SOL (use AWS GPU!)
- [ ] Train Transformer multi-coin model
- [ ] Evaluate on validation set
- [ ] Check for overfitting
- [ ] Tune hyperparameters if needed

**AWS GPU Training**:
```bash
# You have 4 GPU instance types approved:
# - g4dn.xlarge (recommended - cheapest)
# - g5.xlarge
# - g6.xlarge
# - p3.2xlarge

# Use g4dn.xlarge for training (NVIDIA T4, 16 GB GPU RAM)
# Cost: ~$0.526/hour on-demand
# Training time: ~2-3 hours for all 3 LSTM + 1 Transformer
```

**Training Commands**:
```bash
# Train LSTM models (on AWS GPU instance)
uv run python apps/trainer/main.py --task lstm --coin BTC --epochs 15
uv run python apps/trainer/main.py --task lstm --coin ETH --epochs 15
uv run python apps/trainer/main.py --task lstm --coin SOL --epochs 15

# Train Transformer (multi-coin)
uv run python apps/trainer/main.py --task transformer --epochs 15
```

**Expected Output**:
```
models/
├── lstm_BTC_USD_1m_coingecko_<hash>.pt
├── lstm_ETH_USD_1m_coingecko_<hash>.pt
├── lstm_SOL_USD_1m_coingecko_<hash>.pt
└── transformer_multi_coingecko_<hash>.pt
```

**Success Criteria**:
- ✅ All 4 models trained successfully
- ✅ Validation accuracy: 62-70% (target: ≥68%)
- ✅ No severe overfitting (train/val gap <5%)
- ✅ Calibration error <5%

---

### Week 4: Validation & Decision (Days 22-28)

**Tasks**:
- [ ] Comprehensive backtesting on 2-year data
- [ ] Walk-forward validation
- [ ] Calculate performance metrics (Sharpe, drawdown, win rate)
- [ ] Test runtime integration with Coinbase real-time
- [ ] Generate performance report
- [ ] Make GO/NO-GO decision

**Evaluation Criteria**:
```
✅ GO TO PHASE 2 IF:
- Test accuracy ≥68%
- Calibration error ≤5%
- Sharpe ratio >1.0
- Max drawdown <15%
- Win rate >60%

⚠️  TUNE & RETRY IF:
- Test accuracy 60-67%
- Minor overfitting issues
- Calibration slightly off

❌ INVESTIGATE IF:
- Test accuracy <60%
- Severe overfitting
- Poor calibration
```

**Expected Output**:
```
reports/v5_phase1/
├── evaluation_report_2025-11-22.md
├── backtest_results.json
├── performance_metrics.csv
└── decision_recommendation.md
```

**Success Criteria**:
- ✅ Clear GO/NO-GO recommendation
- ✅ All metrics documented
- ✅ Decision shared with QC Claude and user

---

## 💰 Budget Tracking

### Phase 1 Costs (4 weeks)
```
CoinGecko Analyst:     $129/month × 1 = $129
AWS S3/RDS:            ~$25/month × 1 = $25
AWS GPU training:      ~$5-10 (2-3 hours total)
────────────────────────────────────────────
Total Phase 1:         ~$159-164

Budget approved:       $154/month ✅
Actual:                ~$159-164 (slightly over, acceptable)
```

### Phase 2 Costs (If Phase 1 succeeds)
```
Option A - Conservative:
CoinGecko:             $129/month
Coinbase real-time:    $0/month
AWS production:        ~$50/month
────────────────────────────────────────────
Total:                 $179/month ✅

Option B - Premium (only if ROI proven):
Tardis.dev:            $300-350+/month
AWS production:        ~$50/month
────────────────────────────────────────────
Total:                 $350-400+/month
(Only if profit >$500/month justifies upgrade)
```

---

## 🔧 Technical Requirements

### Data Fetcher Requirements
- **API**: CoinGecko API v3
- **Authentication**: API key in header (`x-cg-pro-api-key`)
- **Rate Limiting**: ~50 calls/minute (Analyst tier)
- **Retry Logic**: Exponential backoff for failures
- **Output Format**: Parquet (compressed)

### Feature Engineering Updates
- **Input**: CoinGecko OHLCV parquet files
- **Output**: Feature parquet files (40-50 columns)
- **Multi-timeframe**: Support 1m, 5m, 15m, 1h intervals
- **Validation**: Check for NaN, outliers, distributions

### Model Training Updates
- **Architecture**: Same LSTM + Transformer (90% reuse!)
- **Input features**: 40-50 (update from 31)
- **Data source**: CoinGecko features (not Coinbase)
- **Hardware**: AWS GPU instances (g4dn.xlarge recommended)
- **Checkpointing**: Save best models based on validation accuracy

---

## 📝 Documentation Updates Needed

### Files to Update
1. **PROJECT_MEMORY.md** - Update with CoinGecko strategy
2. **CLAUDE.md** - Update V5 section with correct pricing
3. **BUILDER_CLAUDE_SUMMARY_2025-11-15.md** - Add pricing correction note

### New Files to Create
1. `scripts/fetch_coingecko_data.py` - CoinGecko data fetcher
2. `scripts/validate_coingecko_data.py` - Data quality validator
3. `WEEK1_PROGRESS_2025-11-15.md` - Week 1 progress tracking
4. `V5_COINGECKO_INTEGRATION.md` - CoinGecko integration guide

---

## 🚧 Common Issues & Solutions

### Issue 1: CoinGecko API Rate Limits
**Symptom**: 429 Too Many Requests errors
**Solution**:
- Add sleep(1.2) between API calls (50 calls/min = 1.2s per call)
- Implement exponential backoff
- Cache responses locally

### Issue 2: Data Gaps
**Symptom**: Missing timestamps in OHLCV data
**Solution**:
- CoinGecko aggregates data from multiple exchanges
- Some gaps expected for low-liquidity periods
- Forward-fill small gaps (<5 minutes)
- Document gaps in validation report

### Issue 3: Feature Count Mismatch
**Symptom**: V4 had 31 features, V5 should have 40-50
**Solution**:
- Reuse all 31 V4 features ✅
- Add 9-19 new OHLCV-based features
- Do NOT try to add microstructure features (need tick data)
- Document exact feature list in code

---

## ✅ Success Checklist

### Week 1 Complete When:
- [ ] Git synced with latest changes
- [ ] CoinGecko API key configured in cloud `.env`
- [ ] Data fetcher script created and tested
- [ ] 2 years OHLCV downloaded for BTC/ETH/SOL
- [ ] Data quality validated
- [ ] Week 1 progress report created

### Week 2 Complete When:
- [ ] Feature engineering updated for OHLCV
- [ ] 40-50 features engineered per symbol
- [ ] Baseline accuracy test shows >50%
- [ ] Feature files ready for training

### Week 3 Complete When:
- [ ] All 4 models trained (3 LSTM + 1 Transformer)
- [ ] Validation accuracy ≥62% (target: ≥68%)
- [ ] Models saved to `models/` directory
- [ ] Training logs documented

### Week 4 Complete When:
- [ ] Comprehensive backtest completed
- [ ] All metrics calculated
- [ ] GO/NO-GO decision made
- [ ] Report shared with QC Claude and user

---

## 📞 Communication Protocol

### Daily Updates (During Active Work)
Create daily progress files:
- `WEEK1_DAY1_PROGRESS.md`
- `WEEK1_DAY2_PROGRESS.md`
- etc.

### Blockers
If blocked, immediately create:
- `BLOCKER_2025-11-XX.md` with details
- Tag QC Claude for review
- Propose solutions

### Decision Points
For major decisions, create:
- `DECISION_POINT_<topic>.md`
- Present options with pros/cons
- Wait for user approval before proceeding

---

## 🎯 Week 1 Priority Tasks (Start Here!)

### Priority 1: Sync & Setup (30 min)
```bash
cd ~/crpbot
git pull origin main
echo 'COINGECKO_API_KEY=CG-VQhq64e59sGxchtK8mRgdxXW' >> .env
```

### Priority 2: Create Data Fetcher (2-3 hours)
- File: `scripts/fetch_coingecko_data.py`
- Test with 7 days of data first
- Then fetch full 730 days

### Priority 3: Download All Data (1-2 hours)
```bash
python scripts/fetch_coingecko_data.py --symbol BTC-USD --days 730
python scripts/fetch_coingecko_data.py --symbol ETH-USD --days 730
python scripts/fetch_coingecko_data.py --symbol SOL-USD --days 730
```

### Priority 4: Validate Quality (1 hour)
- Create validation script
- Compare to existing Coinbase data
- Document findings

**Total Week 1 Estimated Time**: 6-8 hours

---

## 📚 Reference Documents

### Updated with Correct Pricing
- ✅ `V5_SIMPLE_PLAN.md` - Corrected V5 strategy
- ✅ `V5_BUDGET_PLAN.md` - Corrected budget breakdown
- ✅ `DATA_STRATEGY_COMPLETE.md` - Corrected data strategy
- ✅ `PRICING_CORRECTION_2025-11-15.md` - Pricing error documentation

### CoinGecko Documentation
- API Docs: https://docs.coingecko.com/reference/introduction
- Pricing: https://www.coingecko.com/en/api/pricing
- Rate Limits: https://docs.coingecko.com/reference/rate-limits

### AWS GPU Documentation
- Instance types: Already approved (g4dn, g5, g6, p3)
- Pricing: ~$0.526/hour for g4dn.xlarge
- Guide: See `AWS_GPU_SETUP.md` (if exists)

---

## 🎉 Ready to Start!

**Current Status**: All prerequisites met ✅
- CoinGecko API key obtained
- Budget approved ($154/month Phase 1)
- Documentation updated
- Cloud server ready

**Next Action**: Start Week 1 Priority 1 (Sync & Setup)

**Timeline**: 4 weeks to validation decision

**Support**: QC Claude available for questions and reviews

---

**File**: `BUILDER_CLAUDE_INSTRUCTIONS_2025-11-15.md`
**Status**: Ready for execution
**Priority**: HIGH - Start Week 1 immediately
**Contact**: QC Claude (local machine) for questions

**Good luck with V5! 🚀**
