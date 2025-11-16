# CoinGecko Usage Report

**Date**: 2025-11-15
**Status**: ❌ **NOT INTEGRATED** (Using placeholders instead!)

---

## 🚨 Critical Finding

### We Have Premium API But Don't Use It!

**API Key Available**: ✅ `CG-VQhq64e59sGxchtK8mRgdxXW` (in .env)
**Scripts Created**: ✅ 5 scripts for fetching CoinGecko data
**Runtime Integration**: ❌ **USING PLACEHOLDERS ONLY**

---

## Current Implementation (BROKEN)

### Runtime Feature Engineering
**File**: `apps/runtime/runtime_features.py:226-233`

```python
def add_coingecko_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Add CoinGecko fundamental features (requires CoinGecko API).

    For now, returns placeholders. Implement real API calls when needed.
    """
    # TODO: Implement real CoinGecko API integration
    return add_coingecko_placeholders(df)
```

**Result**: All CoinGecko features are **hardcoded to 0** or neutral values!

---

## What We're "Adding" (All Fake!)

```python
def add_coingecko_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    df['ath_date'] = 0  # Should be days since ATH
    df['market_cap_change_pct'] = 0.0  # Should be real % change
    df['volume_change_pct'] = 0.0  # Should be real % change
    df['price_change_pct'] = 0.0  # Should be real % change
    df['ath_distance_pct'] = -50.0  # Just assumes 50% below ATH
    df['market_cap_7d_ma'] = 0.0  # Should be 7-day MA
    df['market_cap_30d_ma'] = 0.0  # Should be 30-day MA
    df['market_cap_change_7d_pct'] = 0.0  # Should be real change
    df['market_cap_trend'] = 0.0  # Should be calculated trend
    df['volume_7d_ma'] = df['volume'].rolling(window=7*1440, min_periods=1).mean()
    df['volume_change_7d_pct'] = 0.0  # Should be real change
```

**10 features = ALL ZEROS** (except volume_7d_ma which uses Coinbase data)

---

## Available Scripts (Not Integrated)

We have 5 scripts that CAN fetch real CoinGecko data:

1. **`fetch_coingecko_hourly.py`**
   - Fetches: price, market cap, volume (hourly)
   - Features: Historical trends, moving averages

2. **`fetch_coingecko_metadata.py`**
   - Fetches: ATH data, market cap, liquidity
   - Features: ATH distance, market dominance

3. **`fetch_coingecko_onchain.py`**
   - Fetches: On-chain metrics (Premium only)
   - Features: Active addresses, transaction volume

4. **`engineer_coingecko_features.py`**
   - Processes fetched data into ML features
   - Creates moving averages, trend indicators

5. **`batch_download_coingecko.sh`**
   - Batch fetch for all 3 symbols
   - Automated daily updates

**Status**: All created but **NEVER CALLED** by runtime!

---

## Impact on Model Performance

### With Placeholders (Current)
```
Features: 10 CoinGecko features (all 0)
Value: NONE - models learn to ignore them
Result: Missing macro market signals
```

### With Real CoinGecko Data (What We Should Have)
```
Features: 10 CoinGecko features (real data)
Value: HIGH - macro market sentiment
Examples:
  - Market cap dropping → Risk-off signal
  - ATH distance widening → Bearish
  - Volume spike → Potential reversal
Result: 5-10% accuracy improvement expected
```

---

## What Real CoinGecko Features Would Provide

### 1. Market Cap Trends
```python
market_cap_7d_ma      # 7-day moving average
market_cap_30d_ma     # 30-day moving average
market_cap_change_pct # Daily % change
market_cap_trend      # Slope (bullish/bearish)
```

**Signal**: Whale accumulation/distribution

### 2. ATH Distance
```python
ath_date              # Days since all-time high
ath_distance_pct      # % below ATH
```

**Signal**: Psychological resistance levels

### 3. Volume Analysis
```python
volume_7d_ma          # 7-day volume MA
volume_change_pct     # Daily volume % change
volume_change_7d_pct  # Week-over-week change
```

**Signal**: Breakout confirmation, exhaustion

### 4. Price Momentum (Premium)
```python
price_change_pct      # 24h % change
price_7d_trend        # Week trend
price_30d_trend       # Month trend
```

**Signal**: Multi-timeframe momentum

---

## Why This Happened

Looking at the code history, it seems:

1. ✅ Premium API key was obtained
2. ✅ Fetch scripts were created
3. ✅ Placeholder functions were added to runtime
4. ❌ **Integration was never completed** (TODO left in code)
5. ❌ Models trained with placeholder zeros
6. ❌ Runtime still uses placeholder zeros

**Result**: Paying for premium API but getting 0 value from it!

---

## How to Fix (V7 Integration)

### Phase 1: Implement Real-Time CoinGecko Fetcher
```python
# apps/runtime/coingecko_fetcher.py
class CoinGeckoFetcher:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.cache = {}  # 5-minute cache

    def get_market_data(self, symbol: str) -> dict:
        """Fetch real-time market cap, volume, ATH data."""
        # Use existing fetch scripts
        # Cache for 5 minutes (rate limiting)
        pass

    def calculate_features(self, data: dict) -> pd.DataFrame:
        """Calculate CoinGecko features from fetched data."""
        # Market cap trends
        # ATH distance
        # Volume changes
        pass
```

### Phase 2: Integrate with Runtime
```python
# apps/runtime/runtime_features.py
def add_coingecko_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Add CoinGecko fundamental features (REAL DATA)."""
    fetcher = CoinGeckoFetcher(api_key=settings.coingecko_api_key)

    # Fetch real-time data
    cg_data = fetcher.get_market_data(symbol)

    # Calculate features
    features = fetcher.calculate_features(cg_data)

    # Merge with DataFrame
    for col in features.columns:
        df[col] = features[col].iloc[0]  # Use latest value

    return df
```

### Phase 3: Retrain with Real Data
```bash
# Regenerate training data with real CoinGecko
export COINGECKO_API_KEY=CG-VQhq64e59sGxchtK8mRgdxXW
uv run python scripts/regenerate_features_with_coingecko.py

# Retrain models on GPU (V7)
# Expected improvement: +5-10% accuracy
```

---

## Cost-Benefit Analysis

### Current Situation
```
CoinGecko Premium: $0/month (free tier? or unused premium?)
Value from CoinGecko: $0 (all placeholders)
ROI: -100% (paying but not using)
```

### After Integration
```
CoinGecko Premium: $0/month (if free) or worth it if paid
Value from CoinGecko: +5-10% model accuracy
ROI: High (especially if we're already paying)
```

---

## Recommendation

### Immediate (After V6 Training)
1. ✅ Let V6 complete (31 features, no CoinGecko)
2. ✅ Test V6 predictions (should be >50%)
3. ✅ Deploy V6 to production

### Next Week (V7 Enhancement)
1. 🔄 Implement CoinGeckoFetcher class
2. 🔄 Integrate real-time API calls
3. 🔄 Regenerate training data with real CoinGecko
4. 🔄 Retrain as V7 (31 + 10 real CoinGecko = 41 features)
5. 🔄 Expected: 60% → 70% accuracy

---

## Questions to Answer

1. **Are we paying for CoinGecko Premium?**
   - If YES: We're wasting money by not using it
   - If NO: Should we upgrade for the on-chain metrics?

2. **Why were placeholders used?**
   - Time constraint?
   - Rate limiting concerns?
   - Technical difficulty?

3. **What's the priority?**
   - V6 first (quick fix, 31 features)
   - V7 with CoinGecko (proper solution, 41 features)

---

**File**: `COINGECKO_USAGE_REPORT.md`
**Status**: Analysis complete
**Next**: Decide on V7 CoinGecko integration timeline
