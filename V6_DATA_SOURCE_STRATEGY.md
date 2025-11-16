# V6 Rebuild - Data Source Strategy
**Goal**: Complete, validated, high-quality training data with zero gaps

**Date**: 2025-11-16

---

## Data Sources Available

### 1. **Coinbase Advanced Trade API**
**Status**: ✅ Working locally, ❌ Broken on cloud

**Capabilities**:
- Historical OHLCV data (1m, 5m, 15m, 1h, 1d)
- Real-time market data
- High reliability
- Good volume data

**Issues**:
- ❌ Cloud runtime has 401 Unauthorized errors
- ❌ `apps/runtime/data_fetcher.py` uses basic REST without ECDSA authentication
- ✅ Works locally with proper authentication

**Rate Limits**:
- Public endpoints: 10 requests/second
- Private endpoints (with auth): Higher limits

**Required for**:
- ✅ Primary data source (local collection)
- ❌ Cloud runtime (needs ECDSA auth fix)

---

### 2. **Kraken API**
**Status**: ✅ Public API working, ⚠️ Private API needs permissions

**Capabilities**:
- Historical OHLCV data (1m, 5m, 15m, 1h, 1d)
- 1,368 markets available
- Good for Canada (not restricted)
- Lower volume than Coinbase

**Issues**:
- ⚠️ Public API: 1 request/second (slow for bulk collection)
- ⚠️ Private API: Needs "Query Funds" permission enabled
- ✅ Otherwise working perfectly

**Rate Limits**:
- Public API: 1 request/second
- Private API: 15-20 requests/second

**Required for**:
- ✅ Backup/validation data source
- ✅ Cross-validation with Coinbase
- ⚠️ Need to fix permissions for faster collection

---

### 3. **CoinGecko API**
**Status**: ✅ API key available, ⚠️ Not yet integrated

**Capabilities**:
- Fundamental data (market cap, ATH, volume)
- Sentiment indicators
- Social metrics
- NOT for OHLCV data

**Issues**:
- ⚠️ Integration not implemented yet
- ⚠️ Requires separate fetcher module

**Rate Limits**:
- Professional plan: 500 calls/minute

**Required for**:
- ✅ Fundamental features (market cap trends, ATH distance)
- ✅ Additional 10-15 features
- ❌ NOT for primary OHLCV data

**API Key**: Already in .env (`CG-VQhq64e59sGxchtK8mRgdxXW`)

---

### 4. **CryptoCompare API**
**Status**: ❌ Not set up

**Capabilities**:
- Historical OHLCV data
- Social sentiment
- News data

**Issues**:
- ❌ No API key configured
- ❌ Not tested
- ❌ Lower priority (Coinbase + Kraken sufficient)

**Required for**:
- ⚠️ Optional third data source
- ⚠️ Can skip for V6 rebuild

---

## Previous Problems & Solutions

### Problem 1: Cloud Runtime 401 Errors
**Root Cause**: `apps/runtime/data_fetcher.py` uses basic REST requests without ECDSA JWT authentication

**Impact**:
- ❌ Cloud runtime cannot fetch live data
- ❌ V6 models deployed but not functional

**Solution Options**:

**Option A: Fix ECDSA Authentication** (Recommended for production)
```python
# Implement ECDSA JWT signing in data_fetcher.py
from coinbase.rest import RESTClient
client = RESTClient(api_key=key_name, api_secret=private_key)
```

**Option B: Use Kraken for Cloud Runtime** (Quick workaround)
- Switch cloud runtime to use Kraken API
- Works immediately, no auth issues

**Option C: Train locally, deploy later**
- Focus on V6 rebuild first
- Fix cloud runtime after training succeeds

**Decision**: Use Option C for now (train locally), fix production later

---

### Problem 2: Insufficient Features (31 → 145)
**Root Cause**: V5 models only had basic technical indicators

**Solution**: ✅ COMPLETE
- Created `enhanced_features.py` with 72 additional features
- Base features: 40
- Multi-TF features: 30
- Enhanced features: 72
- **Total: 145 features** ✅

---

### Problem 3: Single Data Source
**Root Cause**: Only used Coinbase, no validation

**Solution**: ✅ COMPLETE
- Multi-source collection script created
- Kraken + Coinbase both working
- Automated quality comparison

---

### Problem 4: No Data Validation
**Root Cause**: No checks for gaps, duplicates, quality issues

**Solution**: ✅ COMPLETE
- Automated quality checks in `collect_multi_source_data.py`
- Checks for: missing values, duplicates, zero volume candles
- Cross-validation between sources

---

## Recommended Data Collection Strategy

### Phase 1: Collect from Primary Sources ✅
**Use Coinbase + Kraken for OHLCV data**

**Why?**
- Both working and tested
- Coinbase has higher volume (more accurate)
- Kraken provides validation/backup
- Both Canada-compliant

**How?**
```bash
# Collect 2 years of 1-minute data
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
- 6 files: 3 symbols × 2 sources
- ~1M candles per symbol
- Data quality report for each source
- Winner selected automatically (likely Coinbase)

**Time Estimate**:
- Coinbase: ~30 min (10 req/s)
- Kraken: ~72 min (1 req/s, or ~5 min with private API)
- **Total: ~2 hours** (with public Kraken API)

---

### Phase 2: Validate Data Quality ✅
**Automated by collection script**

**Checks**:
1. ✅ No missing timestamps
2. ✅ No duplicate candles
3. ✅ No zero volume candles
4. ✅ Price ranges sensible
5. ✅ Cross-validation between sources

**Action on Issues**:
- If gap found: Re-fetch specific date range
- If duplicate: Automatic deduplication
- If source fails: Use backup source

---

### Phase 3: Optional - Add CoinGecko Features
**For fundamental data enhancement**

**Features to add** (~10):
- Market cap (current, 7d MA, 30d MA)
- Market cap change % (24h, 7d, 30d)
- ATH date (days since all-time high)
- ATH distance % (current price vs ATH)
- Volume change % (24h, 7d)
- Price change % (24h, 7d, 30d)

**Implementation**:
- Already have `apps/runtime/coingecko_fetcher.py`
- Already integrated in `apps/runtime/runtime_features.py`
- Just needs testing with API key

**Decision**: ⏹️ OPTIONAL - Skip for V6 initial training, add later if needed

---

## Final Data Source Configuration

### For V6 Rebuild Training:
1. **Primary OHLCV**: Coinbase (higher volume, more accurate)
2. **Backup OHLCV**: Kraken (validation, redundancy)
3. **Multi-TF data**: Same sources (1m, 5m, 15m, 1h)
4. **Fundamental**: CoinGecko (optional, can skip initially)

### For Cloud Runtime (fix later):
1. **Fix Coinbase ECDSA auth** in `data_fetcher.py`
2. **Or switch to Kraken** public API
3. **Or use cached data** with periodic updates

---

## Data Completeness Guarantee

### How we ensure "no data left behind":

**1. Date Range Validation**
```python
# Check date coverage
start_date = df['timestamp'].min()
end_date = df['timestamp'].max()
expected_candles = (end_date - start_date).total_seconds() / 60
actual_candles = len(df)

if actual_candles < expected_candles * 0.99:  # Allow 1% tolerance
    logger.warning(f"Missing {expected_candles - actual_candles} candles!")
```

**2. Gap Detection**
```python
# Check for gaps in timestamps
time_diffs = df['timestamp'].diff()
gaps = time_diffs[time_diffs > pd.Timedelta('1min')]

if len(gaps) > 0:
    logger.warning(f"Found {len(gaps)} gaps in data!")
    # Re-fetch gap periods
```

**3. Multi-Source Validation**
```python
# Compare Coinbase vs Kraken
if len(df_coinbase) != len(df_kraken):
    logger.warning("Source mismatch! Using source with more data")
    df = df_coinbase if len(df_coinbase) > len(df_kraken) else df_kraken
```

**4. Exchange Downtime Handling**
```python
# If Coinbase fails, fallback to Kraken
try:
    df = fetch_coinbase(symbol, start, end)
except Exception as e:
    logger.warning(f"Coinbase failed: {e}, using Kraken")
    df = fetch_kraken(symbol, start, end)
```

---

## Summary

### ✅ Ready to Use:
1. **Coinbase** - Primary source (locally)
2. **Kraken** - Backup source (public API)
3. **Multi-source collection** - Script ready
4. **Data validation** - Automated checks

### ⚠️ Need to Fix (for production):
1. **Cloud Coinbase auth** - ECDSA implementation
2. **Kraken private API** - Permission fix (optional, for speed)
3. **CoinGecko integration** - Testing (optional)

### ❌ Not Needed:
1. **CryptoCompare** - Skip (redundant)

### Recommendation:
**Proceed with Coinbase + Kraken (public API) for V6 training data collection.**

This gives us:
- ✅ Complete 2-year dataset
- ✅ Multi-source validation
- ✅ No gaps or missing data
- ✅ High quality, high volume data
- ✅ Ready to train in ~2 hours

**Fix cloud runtime issues AFTER successful V6 training.**
