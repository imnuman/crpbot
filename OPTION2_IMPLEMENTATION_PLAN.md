# Option 2: Full Implementation Plan

**Goal**: Train models with 7 years data + 15 sentiment features
**Timeline**: 2-3 weeks
**Cost**: $4.94/month + $0.61 one-time

---

## Summary: What We're Building

### Data Sources (ALL FREE)
1. ✅ **Coinbase** - 7 years OHLCV data
2. ⏳ **Reddit** - Community sentiment (6-layer filtered)
3. ⏳ **CryptoCompare** - Social metrics (250k calls/month)
4. ⏳ **CoinGecko** - Community & dev stats
5. ⏳ **Fear & Greed Index** - Market sentiment
6. ⏳ **Google Trends** - Retail interest
7. ⏳ **Santiment** - On-chain metrics (1k calls/month)

### Features
- **Current**: 50 features (multi-TF only)
- **Target**: 65 features (50 multi-TF + 15 sentiment)
- **Improvement**: +30% more features

### Models
- **BTC-USD**: 7 years data (2018-2025)
- **ETH-USD**: 7 years data (2018-2025)
- **SOL-USD**: 4.4 years data (2021-2025)

### Expected Results
- **Accuracy**: 60% → 70-75%
- **Sharpe Ratio**: 1.5 → 2.0-2.3
- **Training Time**: 3 minutes (GPU) vs 9 days (CPU)
- **Monthly Cost**: $4.94

---

## Implementation Phases

### ✅ Phase 1: Infrastructure Setup (DONE)
```
✅ AWS account configured
✅ S3 bucket created: crpbot-ml-data-20251110
✅ Versioning enabled
✅ Data uploading to S3 (in progress)
✅ GPU training scripts created
✅ Amazon Q integration ready
```

### ⏳ Phase 2: API Setup (TODAY - 2 hours)
```
⏳ Reddit API credentials
⏳ CryptoCompare API key
⏳ CoinGecko (no key needed)
⏳ Fear & Greed (no key needed)
⏳ Google Trends (no key needed)
⏳ Santiment API key
⏳ Test all APIs
```

**Deliverable**: All APIs working, credentials stored securely

### ⏳ Phase 3: Reddit Filtering Implementation (Week 1 - 6 hours)
```
⏳ Layer 1: Subreddit quality filtering
⏳ Layer 2: User reputation filtering
⏳ Layer 3: Content quality filtering
⏳ Layer 4: NLP-based filtering (spam, sentiment)
⏳ Layer 5: Time decay weighting
⏳ Layer 6: Engagement weighting
⏳ Test with sample data
```

**Deliverable**: Reddit pipeline that filters 70% noise → 85%+ signal

### ⏳ Phase 4: Historical Data Collection (Week 1-2 - 8 hours)
```
⏳ Fetch 7yr BTC-USD (1m, 5m, 15m, 1h) - 3 hours
⏳ Fetch 7yr ETH-USD (1m, 5m, 15m, 1h) - 3 hours
⏳ Fetch 4.4yr SOL-USD (1m, 5m, 15m, 1h) - 2 hours
⏳ Fetch Reddit historical (6 months) - 1 hour
⏳ Fetch CryptoCompare social (2 years) - 30 min
⏳ Fetch CoinGecko community (2 years) - 30 min
⏳ Fetch Fear & Greed (2 years) - 10 min
⏳ Fetch Google Trends (5 years) - 30 min
⏳ Upload all to S3
```

**Deliverable**:
- 3.5GB of historical OHLCV data
- 500MB of sentiment data
- All stored in S3 with versioning

### ⏳ Phase 5: Feature Engineering (Week 2 - 8 hours)
```
⏳ Engineer multi-TF features (50 features)
⏳ Engineer sentiment features (15 features)
⏳ Calculate derived features (momentum, divergence)
⏳ Validate feature quality
⏳ Handle missing data
⏳ Upload feature files to S3
```

**Deliverable**:
- 3 feature files (BTC, ETH, SOL)
- 65 features per symbol
- ~2.5GB total size

### ⏳ Phase 6: GPU Training (Week 3 - 10 minutes)
```
⏳ Launch p3.8xlarge instance
⏳ Download feature files from S3
⏳ Train all 3 models in parallel
⏳ Monitor training progress
⏳ Upload models to S3
⏳ Terminate instance
```

**Deliverable**:
- 3 trained models (BTC, ETH, SOL)
- Training logs with metrics
- Cost: $0.61

### ⏳ Phase 7: Evaluation & Testing (Week 3 - 1 day)
```
⏳ Evaluate on test set (68% accuracy gate)
⏳ Check calibration error (<5%)
⏳ Backtest with historical data
⏳ Compare with baseline models
⏳ Analyze feature importance
⏳ Generate evaluation report
```

**Deliverable**: Evaluation report with metrics, decision on model promotion

### ⏳ Phase 8: Deployment Preparation (Week 4)
```
⏳ Setup Kafka pipeline (local Docker)
⏳ Integrate sentiment data real-time
⏳ Test end-to-end system
⏳ Paper trading (1-2 weeks)
⏳ Monitor performance
⏳ Prepare for live trading
```

**Deliverable**: System ready for live trading

---

## Detailed Cost Analysis

### Free Tier Stack (RECOMMENDED)

**APIs (All FREE)**:
```
Reddit API:          FREE (1000 req/min)
CryptoCompare:      FREE (250k calls/month)
CoinGecko:          FREE (50 calls/min)
Fear & Greed:       FREE (unlimited)
Google Trends:      FREE (400 queries/hour)
Santiment:          FREE (1000 queries/month)
──────────────────────────────────────
TOTAL APIs:         $0/month ✅
```

**Infrastructure**:
```
S3 storage (3GB):              $2.50/month
Data transfer (download):      $0.27 (one-time)
GPU training (4x/month):       $2.44/month
──────────────────────────────────────
TOTAL Infrastructure:          $4.94/month ✅
```

**Total Free Tier**: **$4.94/month** + **$0.88 one-time**

### Premium Option (If Needed Later)

**LunarCrush Basic ($99/month)**:
- Pre-filtered Twitter sentiment
- Saves 20+ hours/month development
- Higher quality than free Reddit
- +5 additional features

**When to upgrade**:
- After models are profitable (Sharpe > 1.5)
- When monthly profit > $1,000 (10x the cost)
- Need Twitter data for edge cases

**Total Premium**: **$103.94/month**

---

## Reddit Noise Filtering: 6-Layer System

### Problem We're Solving
- **Raw Reddit**: 10,000 posts/day, 70% noise
- **Our Goal**: 500-1,000 posts/day, 85%+ signal

### Solution: Multi-Layer Filtering

**Layer 1: Subreddit Quality** ⭐⭐⭐⭐⭐
```
✅ High quality: r/bitcoin, r/ethereum (weight: 1.0)
⚠️ Medium quality: r/CryptoCurrency (weight: 0.8)
❌ Blacklist: r/CryptoMoonShots, r/SatoshiStreetBets
```

**Layer 2: User Reputation** ⭐⭐⭐⭐⭐
```
✅ Comment karma > 500
✅ Account age > 90 days
✅ Not suspended/deleted
❌ New accounts (bots)
❌ Low karma (spam)
```

**Layer 3: Content Quality** ⭐⭐⭐⭐
```
✅ Upvote ratio > 65%
✅ Score > 10 OR comments > 3
✅ Text length > 50 chars
❌ Spam keywords: "airdrop", "100x", "moonshot"
```

**Layer 4: NLP Filtering** ⭐⭐⭐⭐⭐
```
✅ Spam detection (BERT model)
✅ Language quality check
✅ Sentiment confidence > 60%
✅ Crypto relevance check
❌ Gibberish, sarcasm, off-topic
```

**Layer 5: Time Weighting** ⭐⭐⭐⭐
```
Recent posts matter more
Exponential decay: e^(-0.05 × hours)
Half-life: ~14 hours
```

**Layer 6: Engagement Weighting** ⭐⭐⭐⭐
```
Higher engagement = stronger signal
Weight = log(score + comments) × upvote_ratio
Prevents outlier manipulation
```

### Expected Results
```
Before: 10,000 posts/day, 30% signal, 70% noise
After:  500-1,000 posts/day, 85-90% signal, 10-15% noise
Quality: 3x better correlation with price
```

---

## Alternative Free Data Sources

### 1. CryptoCompare (FREE - 250k/month) ⭐⭐⭐⭐
```
✅ Social media statistics
✅ News sentiment (100+ sources)
✅ Price & volume data
✅ 2+ years historical
```

### 2. CoinGecko (FREE - Unlimited) ⭐⭐⭐⭐⭐
```
✅ Community stats (Twitter, Reddit, Telegram)
✅ Developer activity (GitHub)
✅ Market sentiment
✅ Trading volume
Quality: Excellent, pre-aggregated
```

### 3. Fear & Greed Index (FREE) ⭐⭐⭐⭐⭐
```
✅ Daily sentiment index (0-100)
✅ Market-wide sentiment
✅ Historical data available
Quality: Industry standard
```

### 4. Google Trends (FREE) ⭐⭐⭐⭐
```
✅ Search interest over time
✅ 5 years historical
✅ Good retail sentiment proxy
Rate limit: ~400/hour (use carefully)
```

### 5. Santiment Free Tier (FREE - 1k/month) ⭐⭐⭐⭐
```
✅ Social volume (1000+ sources)
✅ GitHub activity
✅ On-chain metrics
✅ Exchange flows
Limited free tier but high quality
```

**Total**: 5 free data sources + Reddit = 6 sources, $0/month

---

## Features Being Added

### Current Features (50)
```
Multi-TF OHLCV:     15 (5m, 15m, 1h)
Technical:          20 (RSI, MACD, BB, etc.)
Multi-TF derived:    8 (alignment, ATR, etc.)
Volatility:          7 (regime, percentile, etc.)
```

### New Sentiment Features (+15)
```
Reddit:              4 (sentiment, volume, engagement, quality)
CryptoCompare:       3 (Twitter, Reddit, sentiment)
CoinGecko:           3 (community, developer, telegram)
Fear & Greed:        1 (index)
Google Trends:       1 (search interest)
Derived:             3 (consensus, momentum, divergence)
```

### Total Features: **65** (50 + 15)

---

## Timeline & Milestones

### Week 1: Data Collection
```
Day 1: API setup (2 hours) ⏳
Day 2-3: Reddit filtering impl (6 hours) ⏳
Day 4-5: Fetch 7yr historical (8 hours) ⏳
Day 6-7: Fetch sentiment data (2 hours) ⏳
```

### Week 2: Feature Engineering
```
Day 8-10: Multi-TF features (50 features) ⏳
Day 11-13: Sentiment features (15 features) ⏳
Day 14: Validation & upload to S3 ⏳
```

### Week 3: Training & Evaluation
```
Day 15: GPU training (10 min, $0.61) ⏳
Day 16-17: Model evaluation ⏳
Day 18-21: Hyperparameter tuning (if needed) ⏳
```

### Week 4: Deployment
```
Day 22-24: Kafka pipeline setup ⏳
Day 25-28: Paper trading ⏳
```

---

## Risk Mitigation

### Data Quality Risks
**Risk**: Low-quality sentiment data
**Mitigation**: 6-layer filtering, multiple sources, quality scoring

### Cost Risks
**Risk**: Unexpected AWS charges
**Mitigation**: Budget alerts, auto-terminate GPU instances, S3 lifecycle policies

### Performance Risks
**Risk**: Models don't meet 68% accuracy gate
**Mitigation**: Multiple data sources, proper feature engineering, hyperparameter tuning

### Timeline Risks
**Risk**: Data collection takes longer than expected
**Mitigation**: Parallel processing, incremental approach, start with 2 years first

---

## Success Metrics

### Phase Gates
```
✅ Phase 2: All APIs working
✅ Phase 3: Reddit filter reduces noise by 60%+
✅ Phase 4: All historical data collected
✅ Phase 5: Features engineered, quality validated
✅ Phase 6: Models trained successfully
✅ Phase 7: Accuracy > 68%, calibration error < 5%
✅ Phase 8: Paper trading Sharpe > 1.5
```

### Final Success Criteria
```
✅ Model accuracy: > 68%
✅ Calibration error: < 5%
✅ Sharpe ratio (backtest): > 1.5
✅ Sharpe ratio (paper trade): > 1.2
✅ Max drawdown: < 15%
✅ Win rate: > 55%
✅ Monthly cost: < $10
```

---

## Next Immediate Steps

### Today (2 hours)
1. ✅ S3 setup complete
2. ⏳ Wait for S3 upload (~10 min remaining)
3. ⏳ Setup Reddit API
4. ⏳ Setup CryptoCompare API
5. ⏳ Setup other free APIs
6. ⏳ Test all APIs

### Tomorrow (6 hours)
7. ⏳ Implement Reddit 6-layer filtering
8. ⏳ Test with sample Reddit data
9. ⏳ Start fetching 7yr historical data

### This Week
10. ⏳ Complete historical data collection
11. ⏳ Engineer all features
12. ⏳ Upload to S3

### Next Week
13. ⏳ Train on GPU
14. ⏳ Evaluate models
15. ⏳ Deploy pipeline

---

## Questions Answered

### Q: How will we handle Reddit noise?
**A**: 6-layer filtering system:
- Subreddit quality + user reputation + content quality
- NLP spam detection + sentiment confidence
- Time decay + engagement weighting
- Expected: 70% noise reduction → 85%+ signal

### Q: What other free resources can we include?
**A**: 5 additional free sources:
- CryptoCompare (social metrics)
- CoinGecko (community stats)
- Fear & Greed Index (market sentiment)
- Google Trends (retail interest)
- Santiment (on-chain metrics)

### Q: If paid works better, can we do that?
**A**: Yes! Recommend **LunarCrush Basic ($99/month)** if:
- Free tier models are profitable (Sharpe > 1.5)
- Monthly profit > $1,000 (10x the cost)
- Need pre-filtered Twitter data
- Saves 20+ hours/month development

### Q: What else do we need to setup for Option 2?
**A**: Complete checklist:
1. API credentials (6 APIs)
2. Reddit filtering pipeline
3. 7 years historical data
4. Sentiment historical data
5. Feature engineering pipeline
6. GPU training setup (done)
7. Model evaluation framework
8. Kafka deployment (later)

**Total setup time**: 40-60 hours over 2-3 weeks
**Total cost**: $4.94/month + $0.88 one-time

---

**Ready to start Option 2? Next step: Setup APIs (2 hours)**
