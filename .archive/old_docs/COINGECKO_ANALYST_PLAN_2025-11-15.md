# CoinGecko Analyst Subscription - Full Utilization Plan

**Date**: 2025-11-15 16:20 EST
**Subscription**: CoinGecko Analyst (ACTIVE)
**Cost**: $129/month
**Status**: ✅ Ready to leverage premium features!

---

## 📦 What We Have Access To

### Plan Features
```
✅ 500k call credits/month  (plenty for our needs!)
✅ 500 rate limit/min       (10x faster than free tier)
✅ 70+ market data endpoints
✅ 10 years historical data (vs 2 years from Coinbase)
✅ Exclusive data endpoints  (premium features)
✅ Priority email support
```

---

## 🎯 How We'll Use It for V5

### Primary Use Case: Market Context & Sentiment Features

**Goal**: Enhance 1-minute Coinbase OHLCV with daily market intelligence

**Data to Fetch** (Daily granularity, 730 days):

#### 1. **Market Metadata** (7-10 features)
- Market cap (USD)
- Market cap rank (1-10,000)
- Market cap 24h change (%)
- Circulating supply (% of total)
- Total supply
- All-time high price & date
- Distance from ATH (%)

**Endpoint**: `/coins/{id}` (1 call per symbol per day)
**Total calls**: 3 symbols × 730 days = 2,190 calls

---

#### 2. **Historical Market Data** (5-7 features)
- Daily OHLC (for validation against Coinbase)
- Daily volume (total across exchanges)
- Market cap historical

**Endpoint**: `/coins/{id}/market_chart/range`
**Total calls**: 3 symbols × ~3 chunks = ~10 calls
**Features**: Price changes, volume changes, market cap trends

---

#### 3. **Global Market Context** (5-8 features)
- BTC dominance (% of total crypto market)
- ETH dominance (% of total crypto market)
- Total crypto market cap
- Total crypto market cap 24h change
- Active cryptocurrencies count
- Fear & Greed Index (if available via integration)

**Endpoint**: `/global` (1 call per day)
**Total calls**: 730 calls (one per day)

---

#### 4. **Exchange-Level Data** (3-5 features) 🔥 EXCLUSIVE
- Price across multiple exchanges (Coinbase, Binance, Kraken, etc.)
- Cross-exchange spread (% difference)
- Volume-weighted average price (VWAP)
- Exchange variance (price std dev across exchanges)

**Endpoint**: `/coins/{id}/tickers` (exclusive endpoint)
**Total calls**: 3 symbols × 730 days = 2,190 calls
**Benefit**: Detect when Coinbase is leading/lagging market

---

#### 5. **Social Sentiment** (2-4 features) 🔥 EXCLUSIVE (if available)
- Twitter followers count
- Reddit subscribers
- GitHub stars/commits (for coins with repos)
- Social mention volume

**Endpoint**: `/coins/{id}` (social metrics included)
**Already covered**: In metadata call
**Benefit**: Detect hype cycles, FOMO/FUD events

---

#### 6. **On-Chain Metrics** (0-3 features) 🔥 EXCLUSIVE (if available)
- Active addresses (24h)
- Transaction volume (24h)
- Hash rate (for PoW coins like BTC)

**Endpoint**: `/coins/{id}/market_chart/range` or exclusive endpoints
**Total calls**: May require additional endpoints (~500-1000 calls)
**Benefit**: Fundamental network strength indicators

---

#### 7. **Trending & Category Data** (2-3 features)
- Is coin trending today? (binary flag)
- Category (DeFi, Layer1, etc.)
- Category performance (how is category trending?)

**Endpoint**: `/search/trending` + `/coins/categories`
**Total calls**: 730 days = ~1,000 calls
**Benefit**: Sector rotation signals

---

## 📊 Total API Call Budget

| Data Type | Calls | % of 500k | Notes |
|-----------|-------|-----------|-------|
| Market metadata | 2,190 | 0.4% | 3 symbols × 730 days |
| Historical OHLC | 10 | <0.01% | Chunked requests |
| Global context | 730 | 0.1% | 1 per day |
| Exchange tickers | 2,190 | 0.4% | Cross-exchange data |
| Trending data | 1,000 | 0.2% | Category trends |
| **Subtotal** | **6,120** | **1.2%** | **Well within limits!** |
| **Buffer (retries, testing)** | 10,000 | 2% | Safety margin |
| **Total** | **~16,000** | **3.2%** | **Plenty of headroom** |

**Conclusion**: We'll use <5% of our monthly quota! 🎉

---

## ⚡ Download Speed Advantage

### With 500 calls/min rate limit:
- **Single symbol metadata (730 days)**: ~1.5 minutes
- **All 3 symbols metadata**: ~5 minutes
- **Exchange tickers (730 × 3)**: ~5 minutes
- **Global + trending**: ~2 minutes

**Total CoinGecko download time**: ~15-20 minutes (vs hours with free tier!)

---

## 🔧 Implementation: CoinGecko Metadata Fetcher

### Script: `scripts/fetch_coingecko_metadata.py` 🆕

```python
#!/usr/bin/env python3
"""
Fetch daily market metadata from CoinGecko Analyst API.

Features:
- Market cap, rank, supply (metadata)
- Cross-exchange prices & spreads
- Global market context (BTC dominance, etc.)
- Social sentiment (if available)
- Trending status

Rate limit: 500 calls/min (Analyst tier)
"""

import os
import time
import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path

# CoinGecko configuration
API_KEY = os.getenv('COINGECKO_API_KEY')
BASE_URL = 'https://api.coingecko.com/api/v3'
HEADERS = {
    'accept': 'application/json',
    'x-cg-pro-api-key': API_KEY
}

# Coin mapping
COIN_IDS = {
    'BTC-USD': 'bitcoin',
    'ETH-USD': 'ethereum',
    'SOL-USD': 'solana'
}

def fetch_market_chart(coin_id, days=730):
    """Fetch historical price, market cap, volume (daily)."""
    url = f"{BASE_URL}/coins/{coin_id}/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily'
    }
    response = requests.get(url, headers=HEADERS, params=params)
    response.raise_for_status()
    data = response.json()

    # Parse prices, market_caps, total_volumes
    df = pd.DataFrame({
        'timestamp': [p[0] for p in data['prices']],
        'price': [p[1] for p in data['prices']],
        'market_cap': [m[1] for m in data['market_caps']],
        'total_volume': [v[1] for v in data['total_volumes']]
    })
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    return df

def fetch_coin_metadata(coin_id):
    """Fetch current metadata (rank, supply, ATH, etc.)."""
    url = f"{BASE_URL}/coins/{coin_id}"
    params = {
        'localization': 'false',
        'tickers': 'false',  # Fetch separately for historical
        'market_data': 'true',
        'community_data': 'true',
        'developer_data': 'true'
    }
    response = requests.get(url, headers=HEADERS, params=params)
    response.raise_for_status()
    return response.json()

def fetch_exchange_tickers(coin_id):
    """Fetch prices across exchanges (cross-exchange spread)."""
    url = f"{BASE_URL}/coins/{coin_id}/tickers"
    params = {
        'include_exchange_logo': 'false',
        'depth': 'true'
    }
    response = requests.get(url, headers=HEADERS, params=params)
    response.raise_for_status()
    return response.json()

def fetch_global_data():
    """Fetch global market data (BTC dominance, total market cap)."""
    url = f"{BASE_URL}/global"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json()

def fetch_trending():
    """Fetch trending coins."""
    url = f"{BASE_URL}/search/trending"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json()

# Rate limiting (500 calls/min = ~0.12s per call)
def rate_limit():
    time.sleep(0.12)

# Main execution
if __name__ == '__main__':
    # TODO: Implement full fetcher with daily loop
    pass
```

**Estimated development time**: 2-3 hours

---

## 📋 Updated Week 1 Timeline

### Today (2025-11-15) - REVISED
| Time | Task | Status |
|------|------|--------|
| 15:43 | Started Coinbase downloads | 🟢 Running |
| 16:00 | Created hybrid data strategy | ✅ Done |
| **16:30** | **Create CoinGecko metadata fetcher** | 🆕 Next |
| **17:00** | **Start CoinGecko data download** | 🆕 Pending |
| **17:20** | **CoinGecko download complete** | 🆕 Pending (~20 min) |
| **18:00** | **Coinbase downloads complete** | ⏳ Waiting |
| **18:30** | **Validate both datasets** | 🆕 Pending |
| **19:00** | **Create merge script** | 🆕 Pending |
| **20:00** | **Generate hybrid datasets** | 🆕 Pending |
| **21:00** | **Week 1 completion report** | 🆕 Pending |

**Goal**: Complete all data collection + preparation TODAY! ✅

---

## 🎯 Final Feature Set (Hybrid)

### Coinbase Tactical Features (30-35)
- OHLCV (5)
- Technical indicators (RSI, MACD, BB, SMA, etc.) (15-20)
- Volume patterns (5)
- Session features (5)

### CoinGecko Strategic Features (20-25) 🆕
- Market metadata (7-10): market cap, rank, supply, ATH
- Cross-exchange (5-6): spread, VWAP, variance
- Global context (5-8): BTC dominance, total market cap, fear & greed
- Sentiment (2-4): social metrics (if available)
- Trending (2-3): trending status, category performance

### **Total: 50-60 Features** ✅

---

## 💡 Key Advantages of Hybrid Approach

### 1. **Multi-Timeframe Intelligence**
- **1-minute tactical**: RSI, MACD (short-term patterns)
- **Daily strategic**: Market cap trends, BTC dominance (macro context)
- **Best of both**: React to micro + macro signals

### 2. **Cross-Market Awareness**
- **Coinbase-only**: Single exchange view (limited context)
- **+ CoinGecko**: See global market (Binance, Kraken, etc.)
- **Benefit**: Detect when Coinbase is out of sync

### 3. **Sentiment Signals**
- **OHLCV-only**: Price action (reactive)
- **+ Social data**: Detect hype cycles early (proactive)
- **Example**: Twitter spike + price increase = FOMO (potential reversal)

### 4. **Regime Detection**
- **Fear & Greed Index**: Market mood (extreme fear = buy opportunity?)
- **BTC Dominance**: Altseason vs BTC season
- **Market cap trends**: Whale accumulation signals

---

## 📊 Expected Accuracy Boost

### Conservative Estimate
- **Baseline** (Coinbase only): 65-70% accuracy
- **Hybrid** (Coinbase + CoinGecko): 67-75% accuracy
- **Improvement**: +2-5% absolute accuracy

### Optimistic Estimate (if strategic features are highly predictive)
- **Baseline**: 65-70%
- **Hybrid**: 70-78%
- **Improvement**: +5-8%

**Target for Phase 1**: ≥68% accuracy (promotion gate)

**With hybrid approach, we have better odds of clearing the gate!** 🎯

---

## ✅ Next Steps (Immediate)

1. **Create CoinGecko metadata fetcher** (~2 hours)
   - Script: `scripts/fetch_coingecko_metadata.py`
   - Leverage 500 calls/min rate limit
   - Fetch 730 days × 3 symbols (~15-20 min download)

2. **Create CoinGecko feature engineer** (~1 hour)
   - Script: `scripts/engineer_coingecko_features.py`
   - Generate 20-25 strategic features

3. **Create feature merge script** (~1 hour)
   - Script: `scripts/merge_features.py`
   - Combine Coinbase 1m + CoinGecko daily
   - Output: Hybrid dataset (50-60 features, 1m granularity)

4. **Validate hybrid datasets** (~30 min)
   - Check for missing values
   - Verify feature distributions
   - Ensure proper alignment

5. **Update Week 1 report** (~30 min)
   - Document hybrid approach
   - Show feature breakdown
   - Summarize data collection

**Total additional time**: ~5 hours (can finish today!)

---

## 💰 Value Justification

**CoinGecko Cost**: $129/month
**Benefit**: +2-5% accuracy improvement
**ROI Calculation**:
- If V5 achieves 70% with Coinbase alone → Passes gate
- If V5 achieves 75% with hybrid → Higher Sharpe, more confidence
- Potential to reduce drawdowns, increase win rate
- **Justifies the $129/month expense** ✅

**Plus**: We use <5% of our API quota, so plenty of headroom for experimentation!

---

## 🎉 Summary

**Decision**: Leverage full CoinGecko Analyst subscription ✅

**Plan**:
1. ✅ Download Coinbase 1m OHLCV (in progress)
2. 🆕 Download CoinGecko daily metadata (500 calls/min!)
3. 🆕 Engineer tactical + strategic features
4. 🆕 Merge into hybrid dataset (50-60 features)
5. ✅ Train models with enhanced feature set
6. 🎯 Target: 67-75% accuracy (beat the 68% gate!)

**Timeline**: Complete data collection + feature engineering today!

**Budget**: $154/month (CoinGecko + AWS) - as originally planned

**Confidence**: HIGH - We have the tools and data to build a strong V5! 🚀

---

**File**: `COINGECKO_ANALYST_PLAN_2025-11-15.md`
**Status**: Ready to create CoinGecko fetcher
**Next**: Build `scripts/fetch_coingecko_metadata.py`
