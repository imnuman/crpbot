# CoinGecko Analyst Upgrade Plan - Analysis & Implementation Strategy

**Date**: 2025-11-16
**Status**: Planning Phase
**API**: CoinGecko Analyst (ACTIVE - $129/mo)

---

## 📊 CURRENT STATE

### ✅ What We Have

**Models:**
- LSTM (V5, V6, V7 Enhanced)
- FNN (V6 Enhanced - 4 layer)
- Current accuracy: ~70% (V7: 70.2%)

**Symbols:**
- 3 coins: BTC-USD, ETH-USD, SOL-USD
- Data source: Coinbase Advanced Trade API

**Features:**
- 72 features (V6/V7 Enhanced)
- Technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands, etc.
- Basic CoinGecko: 10 features (market cap, price changes, ATH distance)
- Timeframe: 1-minute candles

**Infrastructure:**
- AWS GPU training (g4dn.xlarge)
- S3 storage for models/data
- Live runtime on cloud server (178.156.136.185)
- CoinGecko API: ACTIVE & PAID ✅

### ❌ What We Don't Have

**Models:**
- XGBoost, LightGBM, CatBoost (have reference code in v3_ultimate, not in production pipeline)

**Symbols:**
- 7 additional coins: BNB, ADA, XRP, MATIC, AVAX, DOGE, DOT

**Features (proposed +150):**
- Advanced CoinGecko (60): Social, developer, community metrics
- Cross-asset (40): Correlations, dominance, rankings
- Engineered (50): Multi-timeframe, advanced TA

**Architecture:**
- Specialist models for different market conditions
- Social divergence signals
- 15-minute timeframe option

---

## 🎯 PROPOSED UPGRADES (from instructions)

### Phase 1: Add 3 Models + 7 Coins
**Goal**: 55% → 60% WR (+5%)

**Models to add:**
- XGBoost
- LightGBM
- CatBoost

**Coins to add:**
- BNB, ADA, XRP, MATIC, AVAX, DOGE, DOT

**Feasibility**: ⚠️ MEDIUM
- ✅ Have reference code for models (v3_ultimate)
- ⚠️ Need to integrate into main training pipeline
- ⚠️ BNB not available on Coinbase (would need Kraken/Binance)
- ✅ CoinGecko can provide data for all coins

---

### Phase 2: CoinGecko Features (+60)
**Goal**: 60% → 64% WR (+4%)

**Market Data (15):**
- ✅ Market cap ranking - AVAILABLE
- ✅ Market cap dominance - AVAILABLE
- ✅ Volume ranking - AVAILABLE
- ✅ Volume/market cap ratio - CALCULABLE
- ✅ Circulating supply % - AVAILABLE
- ✅ Fully diluted valuation - AVAILABLE
- ✅ ATH distance - ALREADY HAVE
- ✅ ATL distance - AVAILABLE
- ✅ 7d/30d price change - AVAILABLE
- ✅ Price change percentile - CALCULABLE

**Social Metrics (15):**
- ✅ Twitter followers - endpoint: /coins/{id}
- ✅ Reddit subscribers - endpoint: /coins/{id}
- ⚠️ Reddit active users - LIMITED (basic count only)
- ⚠️ Reddit posts/hour - NOT AVAILABLE (no real-time Reddit API)
- ⚠️ Telegram members - LIMITED (static count only)
- ❌ Social mentions spike - NOT AVAILABLE (would need external API)
- ❌ Social dominance % - NOT AVAILABLE
- ❌ Social sentiment - NOT AVAILABLE (would need LunarCrush/Santiment)
- ⚠️ Trending rank - AVAILABLE (coins/trending)

**Developer Activity (15):**
- ✅ GitHub commits - AVAILABLE (for supported coins)
- ✅ GitHub stars - AVAILABLE
- ✅ GitHub forks - AVAILABLE
- ✅ Contributors count - AVAILABLE
- ⚠️ Code additions/deletions - LIMITED (commit count only)
- ⚠️ Pull requests merged - NOT DIRECTLY AVAILABLE
- ⚠️ Closed issues - NOT DIRECTLY AVAILABLE
- ✅ Development score - AVAILABLE (CoinGecko's proprietary score)

**Community Metrics (15):**
- ✅ Community score - AVAILABLE
- ⚠️ Public interest score - LIMITED
- ⚠️ Liquidity score - NOT AVAILABLE
- ✅ CoinGecko rank changes - AVAILABLE
- ❌ Watchlist adds - NOT AVAILABLE
- ❌ Search trends - NOT AVAILABLE (would need Google Trends)

**Feasibility**: ⚠️ MEDIUM (30-35 features realistically available)
- ✅ Market data: 15 features (100% achievable)
- ⚠️ Social metrics: 5-7 features (50% achievable)
- ✅ Developer activity: 8-10 features (60% achievable)
- ⚠️ Community metrics: 3-5 features (30% achievable)

**Realistic total: ~30-35 features (not 60)**

---

### Phase 3: Cross-Asset Features (+40)
**Goal**: 64% → 67% WR (+3%)

**Correlation Features (20):**
- ✅ BTC correlation - CALCULABLE
- ✅ ETH correlation - CALCULABLE
- ✅ Market-wide correlation - CALCULABLE
- ✅ Sector correlation - CALCULABLE (if we add more L1s)
- ✅ Correlation breakdown detection - CALCULABLE
- ✅ Correlation regime shifts - CALCULABLE

**Dominance Features (10):**
- ✅ BTC dominance trend - AVAILABLE (global API)
- ✅ ETH dominance trend - AVAILABLE
- ✅ Altcoin season index - CALCULABLE
- ✅ Market cap concentration - CALCULABLE
- ✅ Volume concentration - CALCULABLE

**Ranking Features (10):**
- ✅ Rank momentum - CALCULABLE
- ✅ Rank volatility - CALCULABLE
- ✅ Distance to top 10/20/50 - CALCULABLE
- ✅ Rank acceleration - CALCULABLE
- ✅ Relative strength vs ranking - CALCULABLE

**Feasibility**: ✅ HIGH (100% achievable with current data)

---

### Phase 4: Engineered Features (+50)
**Goal**: 67% → 70% WR (+3%)

**Multi-timeframe (20):**
- ✅ Same indicators on 5m/15m/1h/4h - ALREADY PARTIALLY IMPLEMENTED
- ✅ Timeframe alignment - CALCULABLE
- ✅ Support/resistance confluence - CALCULABLE
- ✅ Trend alignment score - CALCULABLE

**Advanced TA (20):**
- ✅ Ichimoku Cloud - ta library supports
- ✅ Fibonacci levels - CALCULABLE
- ⚠️ Elliott Wave - COMPLEX (simplified version possible)
- ⚠️ Market profile - REQUIRES ORDER BOOK DATA (not available)
- ⚠️ Volume profile - REQUIRES TICK DATA (not available)
- ✅ VWAP deviations - CALCULABLE
- ❌ Order flow proxy - NOT AVAILABLE (needs Level 2 data)

**Volatility/Momentum (10):**
- ✅ Volatility regime - ALREADY HAVE
- ✅ Volatility percentile - CALCULABLE
- ✅ Momentum divergence - CALCULABLE
- ✅ Acceleration - CALCULABLE
- ✅ Jerk (3rd derivative) - CALCULABLE

**Feasibility**: ✅ HIGH (~40-45 features achievable)

---

### Phase 5: Specialized Models
**Goal**: 70% → 73% WR (+3%)

**5 Specialists:**
1. High Volume Model - ✅ FEASIBLE
2. Social Spike Model - ⚠️ LIMITED (need better social data)
3. Developer Activity Model - ✅ FEASIBLE
4. Ranking Model - ✅ FEASIBLE
5. Mean Reversion Model - ✅ FEASIBLE

**Feasibility**: ⚠️ MEDIUM (3-4 specialists realistic)

---

### Phase 6: Social Divergence Signals
**Goal**: 73% → 75% WR (+2%)

**Requirements:**
- Real-time social metrics (Twitter, Reddit, Telegram)
- Trend detection algorithms
- Divergence detection (price vs social)

**Feasibility**: ⚠️ LOW
- ❌ Real-time social data not available via CoinGecko
- ❌ Would need LunarCrush ($99+/mo) or Santiment ($229+/mo)
- ⚠️ Can use daily social counts from CoinGecko (limited value)

---

### Phase 7: Quality Filters + 15min
**Goal**: 75% → 77% WR (+2%)

**Changes:**
- Timeframe: 1-min → 15-min
- Confidence threshold: >75%
- Volume filter: >2x average
- Model agreement: 3+ models
- Social score: >60
- Dev score: >50

**Feasibility**: ✅ HIGH (easily implementable)

---

## 📈 REALISTIC EXPECTATIONS

### What the Instructions Promise
- Start: 55% WR
- End: 75-77% WR (+20-22%)
- Features: 50 → 200+
- Time: 5 weeks
- Cost: $130/mo (already paid)

### Reality Check

**Current State:**
- Actual WR: ~70% (not 55%)
- Already optimized features (72)
- Already using CoinGecko (basic)

**Achievable Gains:**

| Phase | Proposed | Realistic | Reason |
|-------|----------|-----------|--------|
| Phase 1 (Models+Coins) | +5% | +1-2% | Already at 70%, gradient boosting may not improve much |
| Phase 2 (CoinGecko 60) | +4% | +1-2% | Only ~35 features available, low predictive power |
| Phase 3 (Cross-asset 40) | +3% | +1-2% | Useful for multi-coin strategies |
| Phase 4 (Engineered 50) | +3% | +1-2% | Some already implemented |
| Phase 5 (Specialists) | +3% | +1-2% | Good idea but limited by data |
| Phase 6 (Social) | +2% | +0-1% | Insufficient real-time data |
| Phase 7 (Filters+15min) | +2% | +1-2% | Timeframe change could help |

**Total realistic gain: +6-12% WR**
**Final expected WR: 76-82%** (from current 70%)

---

## 🛠️ RECOMMENDED IMPLEMENTATION PLAN

### Priority 1: HIGH IMPACT, LOW EFFORT (2-3 weeks)

**Week 1: Gradient Boosting Models**
- ✅ Add XGBoost, LightGBM, CatBoost to training pipeline
- ✅ Use existing 72 features
- ✅ Train ensemble with voting mechanism
- Expected gain: +1-2% WR
- Effort: Medium (have reference code)

**Week 2: CoinGecko Market Features**
- ✅ Add 15 market data features (market cap, dominance, rankings)
- ✅ Add 8-10 developer activity features
- ✅ Add 5-7 basic social features (follower counts, trends)
- ✅ Total: ~30 new features
- Expected gain: +1-2% WR
- Effort: Low (API already integrated)

**Week 3: Cross-Asset + Timeframe**
- ✅ Add 20 correlation features
- ✅ Add 10 dominance features
- ✅ Add 10 ranking features
- ✅ Switch to 15-minute timeframe
- Expected gain: +2-3% WR
- Effort: Medium (calculations only)

**Total Week 1-3: +4-7% WR → 74-77% WR**

---

### Priority 2: MEDIUM IMPACT, MEDIUM EFFORT (2-3 weeks)

**Week 4: Advanced TA**
- ⚠️ Add Ichimoku Cloud
- ⚠️ Add Fibonacci levels
- ⚠️ Add VWAP deviations
- ⚠️ Enhance multi-timeframe features
- Expected gain: +1-2% WR
- Effort: Medium-High

**Week 5: Specialist Models**
- ⚠️ High Volume specialist
- ⚠️ Ranking specialist
- ⚠️ Mean Reversion specialist
- ⚠️ Meta-model selector
- Expected gain: +1-2% WR
- Effort: High

**Total Week 4-5: +2-4% WR → 76-81% WR**

---

### Priority 3: LOW PRIORITY (optional)

**Additional Coins:**
- ⚠️ ADA, XRP, DOGE, DOT (Coinbase available)
- ❌ BNB, MATIC, AVAX (need alternative exchanges)
- Benefit: Portfolio diversification (not WR improvement)
- Effort: Medium (per coin)

**Social Divergence:**
- ❌ Requires paid social APIs ($100-200/mo extra)
- ❌ Limited historical data for training
- ❌ Not recommended at this time

---

## 💰 COST-BENEFIT ANALYSIS

### Current Investment
- CoinGecko Analyst: $129/mo ✅ PAID
- AWS GPU training: ~$5-10/mo ✅ ACTIVE
- Cloud server: ~$30/mo ✅ ACTIVE

**Total: ~$165/mo**

### Proposed Additional Costs
- LunarCrush (social data): $99/mo ❌ NOT RECOMMENDED
- Santiment (social data): $229/mo ❌ NOT RECOMMENDED
- Alternative exchanges: Free (API only) ✅ POSSIBLE

**Recommendation: Stay with current tools ($165/mo)**

---

## 🎯 FINAL RECOMMENDATION

### What to Implement (Realistic Plan)

**Phase A: Immediate (Week 1-2)**
1. Add XGBoost, LightGBM, CatBoost models
2. Add 30 CoinGecko features (market + dev + social)
3. Expected: 70% → 72-73% WR

**Phase B: Short-term (Week 3-4)**
1. Add 40 cross-asset features
2. Switch to 15-minute timeframe
3. Add quality filters
4. Expected: 72-73% → 75-77% WR

**Phase C: Medium-term (Week 5-6)**
1. Add 20-30 advanced TA features
2. Build 3 specialist models
3. Expected: 75-77% → 77-79% WR

**Total Timeline: 6 weeks**
**Expected Final WR: 77-79%** (from current 70%)
**Additional Cost: $0** (use existing tools)

---

## ⚠️ WHAT NOT TO DO

1. ❌ **Don't add 7 more coins yet**
   - Focus on improving WR first
   - Add coins after achieving 75%+ WR

2. ❌ **Don't subscribe to additional social APIs**
   - CoinGecko social data is sufficient
   - Real-time social signals have low predictive power

3. ❌ **Don't over-engineer features**
   - Quality > quantity
   - 120-150 features is optimal (not 200+)

4. ❌ **Don't expect 22% WR gain**
   - Already at 70%, not 55%
   - Realistic gain: 7-10% → 77-80% final WR

---

## ✅ SUCCESS METRICS

**Target WR by Phase:**
- Current: 70%
- Phase A (2 weeks): 72-73%
- Phase B (4 weeks): 75-77%
- Phase C (6 weeks): 77-79%

**Minimum Acceptable:**
- 75% WR for FTMO challenge
- 100+ signals/month with >65% confidence
- Risk-adjusted return >2.0 Sharpe ratio

**Stretch Goal:**
- 80% WR (if all phases exceed expectations)
- Live FTMO challenge by Week 8

---

## 📋 NEXT STEPS

1. **Review this analysis** with user
2. **Get approval** for Phase A (XGBoost + CoinGecko features)
3. **Create detailed task list** for Week 1-2
4. **Begin implementation** with TodoWrite tracking

**Ready to proceed?**
