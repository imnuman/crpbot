# V5 Hybrid Data Strategy - Coinbase + CoinGecko

**Date**: 2025-11-15 16:10 EST
**Decision**: Use BOTH Coinbase (1m OHLCV) + CoinGecko (market context)
**Status**: CoinGecko subscription already active ✅
**Goal**: Combine tactical + strategic features for higher accuracy

---

## 🎯 Hybrid Approach: Best of Both Worlds

### Data Source #1: Coinbase (Tactical - FREE)
**Purpose**: High-frequency price action and volume patterns
**Granularity**: 1-minute candles
**Timeframe**: 2 years (2023-11-15 to 2025-11-15)
**Cost**: $0/month

**Features** (30-35 features):
- OHLCV raw (5)
- Technical indicators (RSI, MACD, Bollinger Bands, etc.) (15-20)
- Volume patterns (volume MA, ratios, trends) (5)
- Session features (Tokyo, London, NY, weekday) (5)
- Spread features (bid-ask, ATR) (3-5)

**Use Case**: **Tactical trading signals** (15-minute prediction horizon)

---

### Data Source #2: CoinGecko (Strategic - $129/month)
**Purpose**: Market context, sentiment, and cross-market dynamics
**Granularity**: Daily (updated every 24h)
**Timeframe**: 2 years (align with Coinbase)
**Cost**: $129/month

**Features** (15-20 features):

#### Market Metadata (5-7 features)
1. **Market Cap** (USD)
2. **Market Cap Rank** (1-100)
3. **Market Cap 24h Change** (%)
4. **Circulating Supply** (% of total)
5. **BTC Dominance** (% of total crypto market)
6. **ETH Dominance** (% of total crypto market)
7. **All-Time High Distance** (% below ATH)

#### Multi-Exchange Data (5-6 features)
8. **Global Average Price** (aggregated across exchanges)
9. **Coinbase vs Global Spread** (% difference)
10. **Cross-Exchange Price Variance** (std dev across exchanges)
11. **Global 24h Volume** (USD)
12. **Volume-Weighted Average Price** (VWAP)
13. **Exchange Count** (how many exchanges trading)

#### Sentiment & Market Mood (3-5 features)
14. **Fear & Greed Index** (0-100 scale)
15. **24h Volume Change** (% vs 7-day average)
16. **Social Volume Trend** (Twitter/Reddit mentions - if available)
17. **Trending Rank** (is coin trending today?)

#### Category Context (2 features)
18. **Category Performance** (DeFi/Layer1 sector trend)
19. **Market-Wide Trend** (total crypto market cap change)

**Use Case**: **Strategic context** (broader market conditions)

---

## 📊 Combined Feature Set (50-60 Total)

### Feature Engineering Pipeline

```
1. Coinbase 1-minute OHLCV (Raw)
   ↓
2. Engineer Tactical Features (30-35 features)
   - RSI, MACD, Bollinger Bands (1m, 5m, 15m, 1h)
   - Volume patterns
   - Session indicators
   ↓
3. CoinGecko Daily Data (Raw)
   ↓
4. Engineer Strategic Features (15-20 features)
   - Market cap changes
   - BTC/ETH dominance trends
   - Fear & Greed index
   - Cross-exchange spreads
   ↓
5. Merge Daily Features into 1m Dataset
   - Forward-fill daily values for all 1m candles in that day
   - Align timestamps (UTC)
   ↓
6. Final Dataset: 1-minute granularity with daily context
   - 50-60 features total
   - ~1,050,000 rows per symbol
   - Ready for model training
```

---

## 🔧 Implementation Plan

### Week 1: Data Collection (In Progress)

**Task 1.1**: Coinbase 1m OHLCV ✅ (Running now)
```bash
# BTC-USD, ETH-USD, SOL-USD
# Status: Downloading (~2-3 hours remaining)
# Output: data/raw/coinbase/*.parquet
```

**Task 1.2**: CoinGecko Daily Data 🆕 (Start after Coinbase completes)
```bash
# Fetch 730 days of market data for BTC, ETH, SOL
# Endpoints:
#   - /coins/{id}/market_chart/range (price, market_cap, volume)
#   - /coins/{id} (metadata, ranks, supply)
#   - /global (BTC dominance, fear & greed)
#   - /search/trending (trending status)

# Script: scripts/fetch_coingecko_metadata.py (need to create)
# Output: data/raw/coingecko_daily/*.parquet
```

**Estimated Time**: 2-3 hours (API rate limits: 50 calls/min)

---

### Week 1-2: Feature Engineering

**Task 2.1**: Engineer Coinbase Tactical Features
```bash
# Use existing script (already working)
uv run python scripts/engineer_features.py \
  --input data/raw/coinbase/BTC-USD_1m_*.parquet \
  --symbol BTC-USD \
  --interval 1m

# Output: data/features/features_BTC-USD_1m_coinbase.parquet
# Features: 30-35 (OHLCV + technical indicators)
```

**Task 2.2**: Engineer CoinGecko Strategic Features 🆕
```bash
# New script needed
uv run python scripts/engineer_coingecko_features.py \
  --input data/raw/coingecko_daily/BTC-USD_daily_*.parquet \
  --symbol BTC-USD

# Output: data/features/features_BTC-USD_daily_coingecko.parquet
# Features: 15-20 (market context, sentiment)
```

**Task 2.3**: Merge Tactical + Strategic Features 🆕
```bash
# Merge daily CoinGecko features into 1m Coinbase data
# Forward-fill daily values to match 1m granularity

uv run python scripts/merge_features.py \
  --tactical data/features/features_BTC-USD_1m_coinbase.parquet \
  --strategic data/features/features_BTC-USD_daily_coingecko.parquet \
  --output data/features/features_BTC-USD_hybrid.parquet

# Output: 1m granularity with daily context columns
# Total features: 50-60
```

---

### Week 3: Model Training (Enhanced)

**Same as before, but with hybrid dataset**:
```bash
# Train LSTM with 50-60 features (instead of 30-35)
uv run python apps/trainer/main.py --task lstm --coin BTC --epochs 15

# Train Transformer with 50-60 features
uv run python apps/trainer/main.py --task transformer --epochs 15
```

**Expected Improvement**: +2-5% accuracy from strategic context

---

## 📋 New Scripts Needed

### 1. `scripts/fetch_coingecko_metadata.py` 🆕
**Purpose**: Fetch daily market data from CoinGecko
**Endpoints**:
- `/coins/{id}/market_chart/range` - Price, market cap, volume (daily)
- `/coins/{id}` - Metadata (rank, supply, ATH)
- `/global` - BTC/ETH dominance, total market cap
- `/search/trending` - Trending status

**Output**: Daily parquet files for each symbol

---

### 2. `scripts/engineer_coingecko_features.py` 🆕
**Purpose**: Engineer strategic features from CoinGecko raw data
**Features**:
- Market cap changes (daily, 7d, 30d)
- Dominance trends (BTC, ETH)
- Fear & Greed index
- Cross-exchange spreads
- Trending indicators

**Output**: Engineered daily features (parquet)

---

### 3. `scripts/merge_features.py` 🆕
**Purpose**: Merge daily CoinGecko features into 1m Coinbase data
**Logic**:
```python
# Pseudo-code
coinbase_1m = pd.read_parquet('features_BTC-USD_1m_coinbase.parquet')
coingecko_daily = pd.read_parquet('features_BTC-USD_daily_coingecko.parquet')

# Convert timestamps to dates
coinbase_1m['date'] = coinbase_1m['timestamp'].dt.date

# Merge on date (many-to-one)
hybrid = coinbase_1m.merge(
    coingecko_daily,
    left_on='date',
    right_on='date',
    how='left'
)

# Forward-fill any missing daily values
hybrid = hybrid.fillna(method='ffill')

# Save
hybrid.to_parquet('features_BTC-USD_hybrid.parquet')
```

**Output**: Hybrid dataset (1m granularity + daily context)

---

## 🎯 Updated Week 1 Tasks

### ✅ Completed
- [x] Download Coinbase 1m OHLCV (in progress, ~2-3h remaining)

### 🆕 New Tasks (After Coinbase completes)
- [ ] Create `scripts/fetch_coingecko_metadata.py`
- [ ] Fetch CoinGecko daily data (730 days, BTC/ETH/SOL)
- [ ] Create `scripts/engineer_coingecko_features.py`
- [ ] Engineer CoinGecko strategic features
- [ ] Create `scripts/merge_features.py`
- [ ] Merge Coinbase + CoinGecko features
- [ ] Validate hybrid dataset

**Estimated Additional Time**: 4-6 hours (scripting + data fetching)

---

## 💰 Updated Budget

### Phase 1 (4 weeks)
```
Coinbase API:       $0/month    ✅ FREE
CoinGecko Analyst:  $129/month  ✅ Already subscribed
AWS (S3 + RDS):     ~$25/month
──────────────────────────────────────
Total:              $154/month  ✅
```

**This is the original budget** - we're back on track!

---

## 🎯 Expected Benefits

### Hypothesis: Strategic Features Improve Accuracy

**Baseline** (Coinbase only):
- Features: 30-35 (OHLCV + technical)
- Expected accuracy: 65-70%

**Enhanced** (Coinbase + CoinGecko):
- Features: 50-60 (OHLCV + technical + market context)
- Expected accuracy: 67-75% (+2-5% improvement)

**Rationale**:
1. **Market context matters**: Fear & Greed index can signal regime changes
2. **BTC dominance**: When BTC dominance rises, altcoins often suffer
3. **Cross-exchange spreads**: Can detect when Coinbase is lagging/leading
4. **Market cap trends**: Large cap changes can precede price moves

**Examples**:
- **Scenario 1**: Fear & Greed = 10 (Extreme Fear) → More likely to bounce
- **Scenario 2**: BTC dominance spiking → Sell signal for ETH/SOL
- **Scenario 3**: Coinbase price > Global average → Potential correction

---

## 📊 Feature Categories Summary

| Category | Source | Count | Granularity | Examples |
|----------|--------|-------|-------------|----------|
| **OHLCV** | Coinbase | 5 | 1-minute | open, high, low, close, volume |
| **Technical** | Coinbase | 20-25 | 1-minute | RSI, MACD, BB, SMA, volume ratios |
| **Session** | Coinbase | 5 | 1-minute | Tokyo, London, NY, weekday, weekend |
| **Market Context** | CoinGecko | 7-10 | Daily | market cap, rank, dominance, ATH distance |
| **Cross-Exchange** | CoinGecko | 5-6 | Daily | global spread, VWAP, variance |
| **Sentiment** | CoinGecko | 3-5 | Daily | fear & greed, trending, volume change |
| **Total** | **Both** | **50-60** | **1-minute** | **Hybrid dataset** |

---

## ✅ Action Plan (Updated)

### Today (2025-11-15)
1. ✅ Wait for Coinbase downloads to complete (~18:00 EST)
2. 🆕 Create CoinGecko metadata fetcher script
3. 🆕 Start CoinGecko daily data download (730 days, 3 symbols)
4. ✅ Validate Coinbase data quality

### Tomorrow (2025-11-16)
1. 🆕 Verify CoinGecko data downloaded successfully
2. 🆕 Create CoinGecko feature engineering script
3. 🆕 Engineer strategic features (market cap, dominance, sentiment)
4. 🆕 Create merge script (combine Coinbase + CoinGecko)
5. 🆕 Generate hybrid datasets (BTC, ETH, SOL)

### Week 2
1. ✅ Validate hybrid dataset quality
2. ✅ Start model training with 50-60 features
3. ✅ Compare: Coinbase-only vs Hybrid accuracy
4. ✅ Document findings

---

## 🎉 Summary

**Decision**: Use HYBRID approach (Coinbase + CoinGecko) ✅

**Benefits**:
- **Tactical + Strategic** features
- **Potentially +2-5% accuracy** improvement
- **Use existing subscription** (already paid for!)
- **More robust models** (market context awareness)

**Timeline**:
- Week 1: Data collection (Coinbase ✅ + CoinGecko 🆕)
- Week 2: Feature engineering (hybrid datasets)
- Week 3: Model training (enhanced features)
- Week 4: Validation & GO/NO-GO decision

**Budget**: $154/month Phase 1 (as originally planned)

---

**File**: `V5_HYBRID_DATA_STRATEGY_2025-11-15.md`
**Status**: CoinGecko subscription active, ready to fetch metadata
**Next**: Create CoinGecko fetcher script after Coinbase downloads complete
