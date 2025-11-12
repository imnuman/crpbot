# Sentiment Data Strategy & Quality Control

**Created**: 2025-11-10
**Status**: Implementation Plan

## Overview

Integrating sentiment data from multiple sources with proper noise filtering and quality control.

---

## 1. Reddit Data Quality & Noise Filtering

### Problem: Reddit Noise Sources
1. **Spam & Bot Posts**: Automated pump-and-dump posts
2. **Low-Quality Comments**: "To the moon!", "HODL", memes
3. **Manipulation**: Coordinated shilling campaigns
4. **Irrelevant Content**: Off-topic discussions
5. **Sarcasm**: "Bitcoin going to zero! 🚀" (actually bullish)

### Solution: Multi-Layer Filtering System

#### Layer 1: Source Quality Filtering
```python
# Filter by subreddit reputation
HIGH_QUALITY_SUBREDDITS = {
    "r/bitcoin": {"weight": 1.0, "min_karma": 100},
    "r/ethereum": {"weight": 1.0, "min_karma": 100},
    "r/CryptoCurrency": {"weight": 0.8, "min_karma": 500},  # Lower weight, more noise
    "r/CryptoMarkets": {"weight": 0.9, "min_karma": 200},
    "r/solana": {"weight": 0.9, "min_karma": 100},
}

# Filter out low-quality subreddits
BLACKLIST_SUBREDDITS = [
    "r/CryptoMoonShots",  # Pump and dump
    "r/SatoshiStreetBets",  # Meme-focused
    # ... more
]
```

#### Layer 2: User Quality Filtering
```python
def is_quality_user(author):
    """Filter users by reputation."""
    # Minimum karma threshold
    if author.comment_karma < 500:
        return False

    # Account age (avoid new bot accounts)
    if (datetime.now() - author.created_utc).days < 90:
        return False

    # Verify human (not deleted/suspended)
    if author.is_suspended or author.name == "[deleted]":
        return False

    return True
```

#### Layer 3: Content Quality Filtering
```python
def is_quality_content(post):
    """Filter by content quality."""
    # Minimum upvote ratio (avoid controversial spam)
    if post.upvote_ratio < 0.65:
        return False

    # Minimum engagement (avoid ignored posts)
    if post.score < 10 and post.num_comments < 3:
        return False

    # Length filtering (avoid low-effort posts)
    if len(post.selftext) < 50 and len(post.title) < 20:
        return False

    # Spam keywords
    spam_keywords = ["airdrop", "giveaway", "free", "pump", "100x", "moonshot"]
    text_lower = (post.title + " " + post.selftext).lower()
    if sum(1 for kw in spam_keywords if kw in text_lower) >= 2:
        return False

    return True
```

#### Layer 4: Advanced NLP Filtering
```python
from transformers import pipeline
import spacy

# Load models
sentiment_model = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
spam_detector = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sms-spam-detection")
nlp = spacy.load("en_core_web_sm")

def advanced_filtering(text):
    """Apply NLP-based filtering."""
    # 1. Spam detection
    spam_result = spam_detector(text[:512])[0]
    if spam_result["label"] == "spam" and spam_result["score"] > 0.8:
        return None, "spam"

    # 2. Language quality (detect gibberish)
    doc = nlp(text)
    if len(list(doc.sents)) == 0:  # No valid sentences
        return None, "gibberish"

    # 3. Sentiment with confidence
    sentiment_result = sentiment_model(text[:512])[0]
    if sentiment_result["score"] < 0.6:  # Low confidence = ambiguous/sarcasm
        return None, "low_confidence"

    # 4. Extract entities (ensure crypto-related)
    crypto_entities = {"BTC", "Bitcoin", "ETH", "Ethereum", "crypto"}
    has_crypto = any(ent.text in crypto_entities for ent in doc.ents)
    if not has_crypto and "crypto" not in text.lower():
        return None, "off_topic"

    return sentiment_result, "valid"
```

#### Layer 5: Time-Based Weighting
```python
def time_decay_weight(post_age_hours):
    """Recent posts matter more."""
    # Exponential decay: weight = e^(-λt)
    lambda_decay = 0.05  # Half-life ~14 hours
    return np.exp(-lambda_decay * post_age_hours)
```

#### Layer 6: Engagement-Based Weighting
```python
def engagement_weight(post):
    """Higher engagement = more signal."""
    # Normalized score: log scale to prevent outliers
    score_weight = np.log1p(post.score) / 10

    # Comment depth indicates discussion quality
    comment_weight = np.log1p(post.num_comments) / 5

    # Upvote ratio indicates agreement
    ratio_weight = post.upvote_ratio

    # Combined weight
    return (score_weight + comment_weight) * ratio_weight
```

### Complete Reddit Pipeline
```python
def process_reddit_post(post):
    """Complete quality filtering pipeline."""
    # Layer 1: Subreddit check
    if post.subreddit not in HIGH_QUALITY_SUBREDDITS:
        return None

    # Layer 2: User quality
    if not is_quality_user(post.author):
        return None

    # Layer 3: Content quality
    if not is_quality_content(post):
        return None

    # Layer 4: Advanced NLP
    text = f"{post.title} {post.selftext}"
    sentiment_result, status = advanced_filtering(text)
    if status != "valid":
        return None

    # Layer 5 & 6: Weighting
    age_weight = time_decay_weight((datetime.now() - post.created_utc).total_seconds() / 3600)
    engagement_weight_val = engagement_weight(post)
    subreddit_weight = HIGH_QUALITY_SUBREDDITS[post.subreddit]["weight"]

    # Final weighted sentiment
    final_sentiment = sentiment_result["label"]  # POS/NEG/NEU
    final_score = sentiment_result["score"] * age_weight * engagement_weight_val * subreddit_weight

    return {
        "timestamp": post.created_utc,
        "sentiment": final_sentiment,
        "score": final_score,
        "confidence": sentiment_result["score"],
        "engagement": post.score + post.num_comments,
        "subreddit": post.subreddit,
    }
```

### Expected Results
- **Before filtering**: 10,000 posts/day, 70% noise
- **After filtering**: 500-1,000 posts/day, 85-90% signal
- **Quality improvement**: 3-5x better correlation with price

---

## 2. Alternative Data Sources (Free)

### A. CryptoCompare API (FREE Tier) ✅ RECOMMENDED
```
Endpoint: https://min-api.cryptocompare.com/
Free Tier: 250,000 calls/month (8,333/day)
Data: Price, volume, social stats, news
Cost: $0
```

**What we get (FREE)**:
- Social media statistics (Twitter mentions, Reddit activity)
- News sentiment (aggregated from 100+ sources)
- Price & volume data (backup to Coinbase)
- Historical data (2+ years)

**Implementation**:
```python
import requests

def get_cryptocompare_social(symbol):
    url = f"https://min-api.cryptocompare.com/data/social/coin/latest?coinId={symbol}"
    response = requests.get(url)
    data = response.json()["Data"]

    return {
        "twitter_followers": data["Twitter"]["followers"],
        "twitter_statuses": data["Twitter"]["statuses"],
        "reddit_subscribers": data["Reddit"]["subscribers"],
        "reddit_active_users": data["Reddit"]["active_users"],
        "sentiment_score": data["General"]["Points"] / 1000,  # Normalized
    }
```

**Quality**: ⭐⭐⭐⭐ (4/5) - Pre-aggregated, low noise

---

### B. CoinGecko API (FREE) ✅ RECOMMENDED
```
Endpoint: https://api.coingecko.com/api/v3/
Free Tier: 10-50 calls/minute (no key needed)
Data: Price, volume, developer activity, community stats
Cost: $0
```

**What we get (FREE)**:
- Community statistics (Twitter, Reddit, Telegram)
- Developer activity (GitHub commits, stars, forks)
- Market sentiment (Fear & Greed derivative)
- Trading volume across exchanges

**Implementation**:
```python
def get_coingecko_community(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}?localization=false&tickers=false&market_data=false&community_data=true&developer_data=true"
    response = requests.get(url)
    data = response.json()

    return {
        "community_score": data["community_score"],
        "developer_score": data["developer_score"],
        "twitter_followers": data["community_data"]["twitter_followers"],
        "reddit_subscribers": data["community_data"]["reddit_subscribers"],
        "telegram_users": data["community_data"]["telegram_channel_user_count"],
        "github_stars": data["developer_data"]["stars"],
        "github_commits_4w": data["developer_data"]["commit_count_4_weeks"],
    }
```

**Quality**: ⭐⭐⭐⭐⭐ (5/5) - Excellent, low noise

---

### C. Alternative.me Crypto Fear & Greed Index (FREE) ✅ ALREADY PLANNED
```
Endpoint: https://api.alternative.me/fng/
Free Tier: Unlimited
Data: Daily sentiment index (0-100)
Cost: $0
```

**Quality**: ⭐⭐⭐⭐⭐ (5/5) - Market-wide sentiment

---

### D. Santiment Free Tier
```
Endpoint: https://api.santiment.net/graphql
Free Tier: 1,000 queries/month
Data: On-chain metrics, social volume, development activity
Cost: $0
```

**What we get (FREE)**:
- Social volume (mentions across 1000+ sources)
- GitHub activity
- Daily active addresses (on-chain)
- Exchange flow (deposits/withdrawals)

**Quality**: ⭐⭐⭐⭐ (4/5) - High quality, limited free tier

---

### E. Glassnode Free Tier
```
Endpoint: https://api.glassnode.com/v1/
Free Tier: 20 requests/day (limited metrics)
Data: On-chain metrics
Cost: $0
```

**What we get (FREE)**:
- Active addresses
- Exchange netflow
- Supply metrics
- SOPR (Spent Output Profit Ratio)

**Quality**: ⭐⭐⭐⭐⭐ (5/5) - Excellent, but very limited free tier

---

### F. Google Trends (FREE) ✅ ALREADY PLANNED
```
Library: pytrends
Free Tier: ~400 queries/hour (use carefully)
Data: Search interest over time
Cost: $0
```

**Quality**: ⭐⭐⭐⭐ (4/5) - Good retail sentiment proxy

---

## 3. Paid Data Sources (Worth Considering)

### A. LunarCrush (BEST Twitter Alternative) 💰
```
Cost: $99/month (Basic) - $999/month (Pro)
Data: Twitter/X sentiment, influencer scores, social metrics
Quality: ⭐⭐⭐⭐⭐ (5/5)
```

**What we get**:
- Real-time Twitter sentiment (cleaned, de-noised)
- Influencer impact scores
- Social dominance metrics
- Correlation with price (validated)
- Historical data (3+ years)

**Recommendation**: ✅ **Worth it if budget allows ($99/month)**
- Already filtered and de-noised
- 10x less work than doing Twitter ourselves
- Better quality than raw Twitter API

---

### B. CryptoMood 💰
```
Cost: $199/month (Starter) - $2,499/month (Pro)
Data: Multi-source sentiment (Twitter, Reddit, News)
Quality: ⭐⭐⭐⭐⭐ (5/5)
```

**What we get**:
- AI-powered sentiment from 1M+ sources
- Real-time sentiment scores
- Emotion detection (fear, greed, joy, etc.)
- Correlation metrics

**Recommendation**: ⚠️ **Too expensive for now**, consider after profitable

---

### C. Messari Pro 💰
```
Cost: $24.99/month (Essentials) - $149.99/month (Pro)
Data: Market intelligence, on-chain metrics, research
Quality: ⭐⭐⭐⭐⭐ (5/5)
```

**What we get**:
- Institutional-grade research
- On-chain metrics
- Protocol revenue data
- DeFi TVL and flows

**Recommendation**: ⚠️ **Good for research, not real-time trading**

---

### D. CryptoCompare Pro 💰
```
Cost: $29/month - $799/month
Data: Enhanced API limits, historical data, order book
Quality: ⭐⭐⭐⭐⭐ (5/5)
```

**Recommendation**: ❌ **Free tier sufficient for now**

---

## 4. Recommended Data Stack

### Free Tier Stack (RECOMMENDED TO START) - $0/month
```
1. Reddit API (FREE) - Community sentiment
   + Multi-layer filtering (our implementation)
   + Quality score: 3.5/5 (after filtering)

2. CryptoCompare API (FREE) - Social metrics
   + 250k calls/month
   + Quality score: 4/5

3. CoinGecko API (FREE) - Community & dev stats
   + Unlimited with rate limits
   + Quality score: 5/5

4. Fear & Greed Index (FREE) - Market sentiment
   + Daily granularity
   + Quality score: 5/5

5. Google Trends (FREE) - Retail interest
   + Weekly/daily data
   + Quality score: 4/5

6. Santiment Free (FREE) - On-chain metrics
   + 1,000 queries/month
   + Quality score: 4/5

TOTAL: $0/month
FEATURES ADDED: +15 sentiment features
```

### Premium Tier Stack (IF BUDGET ALLOWS) - $99/month
```
All Free Tier sources +

7. LunarCrush Basic ($99/month) - Twitter sentiment
   + Real-time, pre-filtered
   + Quality score: 5/5
   + ROI: High (saves 20+ hours/month dev time)

TOTAL: $99/month
FEATURES ADDED: +20 sentiment features (includes Twitter)
```

### Professional Tier Stack (AFTER PROFITABLE) - $300+/month
```
All Premium Tier sources +

8. Messari Pro ($150/month) - Research & intelligence
9. Glassnode Starter ($29/month) - Advanced on-chain
10. CryptoMood ($199/month) - Multi-source AI sentiment

TOTAL: $378/month
FEATURES ADDED: +30 sentiment features
```

---

## 5. Feature Engineering from Sentiment Data

### Raw Features (Per Symbol, Per Timeframe)
```python
sentiment_features = {
    # Reddit (after filtering)
    "reddit_sentiment_1h": float,  # -1 to +1
    "reddit_volume_1h": int,  # Post count
    "reddit_engagement_1h": int,  # Score + comments
    "reddit_quality_score_1h": float,  # Weighted average

    # CryptoCompare
    "social_twitter_followers_24h": int,
    "social_reddit_subscribers_24h": int,
    "social_sentiment_score": float,  # 0-1

    # CoinGecko
    "community_score": float,  # 0-100
    "developer_score": float,  # 0-100
    "github_commits_4w": int,

    # Fear & Greed
    "fear_greed_index": int,  # 0-100
    "fear_greed_classification": str,  # "Extreme Fear" etc.

    # Google Trends
    "search_interest_btc": float,  # 0-100
    "search_interest_normalized": float,  # -1 to +1

    # Derived features
    "sentiment_momentum_24h": float,  # Change in sentiment
    "sentiment_divergence": float,  # Sentiment vs price divergence
    "social_volume_spike": bool,  # Abnormal volume
}
```

### Advanced Features (14 total)
```python
advanced_features = {
    # Multi-source consensus
    "sentiment_consensus": float,  # Weighted average across all sources
    "sentiment_disagreement": float,  # Standard deviation (conflict = uncertainty)

    # Momentum features
    "sentiment_momentum_1h": float,
    "sentiment_momentum_4h": float,
    "sentiment_momentum_24h": float,

    # Divergence features
    "sentiment_price_divergence": float,  # Sentiment up, price down = potential reversal
    "sentiment_volume_divergence": float,

    # Quality features
    "data_quality_score": float,  # How reliable is current sentiment data
    "signal_strength": float,  # How strong is the sentiment signal

    # Volatility features
    "sentiment_volatility_24h": float,  # How stable is sentiment
    "sentiment_regime": str,  # "stable_bullish", "volatile_bearish", etc.

    # Anomaly detection
    "sentiment_anomaly_score": float,  # Unusual sentiment patterns
    "social_volume_anomaly": float,  # Unusual social activity
    "coordinated_activity_flag": bool,  # Potential manipulation
}
```

**Total Sentiment Features**: 15 (free tier) to 30 (premium tier)

---

## 6. Implementation Checklist for Option 2

### Phase 1: Data Infrastructure (Week 1)
```
✅ S3 storage setup
✅ AWS credentials configured
⏳ Reddit API setup
⏳ CryptoCompare API setup
⏳ CoinGecko API setup
⏳ Fear & Greed API integration
⏳ Google Trends setup
⏳ Santiment API setup (optional)
```

### Phase 2: Historical Data Collection (Week 1-2)
```
⏳ Fetch 7 years BTC data (Coinbase)
⏳ Fetch 7 years ETH data (Coinbase)
⏳ Fetch 4.4 years SOL data (Coinbase)
⏳ Fetch Reddit historical (3-6 months)
⏳ Fetch Fear & Greed historical (2 years)
⏳ Fetch Google Trends historical (5 years)
⏳ Fetch CryptoCompare social (2 years)
⏳ Fetch CoinGecko community (2 years)
```

### Phase 3: Sentiment Processing Pipeline (Week 2)
```
⏳ Implement Reddit filtering (6 layers)
⏳ Implement sentiment NLP models
⏳ Implement feature engineering
⏳ Implement quality scoring
⏳ Implement data validation
⏳ Test end-to-end pipeline
```

### Phase 4: Feature Engineering (Week 2-3)
```
⏳ Engineer multi-TF features (existing)
⏳ Engineer sentiment features (new)
⏳ Engineer derived features
⏳ Validate feature quality
⏳ Upload to S3
```

### Phase 5: Model Training (Week 3)
```
⏳ Setup GPU instance (p3.8xlarge)
⏳ Train with 7yr + sentiment data
⏳ Evaluate models (68% accuracy gate)
⏳ Hyperparameter tuning
⏳ Final training run
⏳ Promote to production
```

### Phase 6: Deployment (Week 4)
```
⏳ Deploy Kafka pipeline
⏳ Integrate sentiment real-time
⏳ Backtest with historical data
⏳ Paper trading (1-2 weeks)
⏳ Live trading preparation
```

---

## 7. Cost Breakdown

### Development Cost (One-Time)
```
Setup time: 40-60 hours
Cost: $0 (your time)

AWS GPU training: $0.61
S3 storage setup: $0

TOTAL ONE-TIME: $0.61
```

### Monthly Operating Cost

**Free Tier** (Recommended to start):
```
S3 storage (3GB):           $2.50
Reddit API:                 $0 (FREE)
CryptoCompare:             $0 (FREE, 250k/mo)
CoinGecko:                 $0 (FREE)
Fear & Greed:              $0 (FREE)
Google Trends:             $0 (FREE)
Santiment:                 $0 (FREE, 1k/mo)
GPU training (4x/month):   $2.44 ($0.61 × 4)
─────────────────────────────────
TOTAL MONTHLY:            $4.94/month ✅
```

**Premium Tier** (Better quality):
```
All Free Tier:             $4.94
LunarCrush Basic:         $99.00 (Twitter sentiment)
─────────────────────────────────
TOTAL MONTHLY:           $103.94/month
```

**Professional Tier** (After profitable):
```
All Premium Tier:         $103.94
Messari Pro:              $149.99
Glassnode Starter:        $29.00
CryptoMood:               $199.00
─────────────────────────────────
TOTAL MONTHLY:           $481.93/month
```

---

## 8. Expected ROI

### Free Tier ROI
```
Monthly Cost: $4.94
Features Added: +15 sentiment features
Expected Accuracy Gain: +3-5%
Expected Win Rate: 55-60% → 58-65%
Expected Sharpe Ratio: 1.5 → 1.8-2.0

ROI: If 1% win rate improvement = $100/month profit
     → $300-500/month profit potential
     → ROI: 60-100x
```

### Premium Tier ROI
```
Monthly Cost: $103.94
Features Added: +20 sentiment features (includes Twitter)
Expected Accuracy Gain: +5-8%
Expected Win Rate: 55-60% → 60-68%
Expected Sharpe Ratio: 1.5 → 2.0-2.3

ROI: If 1% win rate improvement = $100/month profit
     → $500-800/month profit potential
     → ROI: 5-8x
```

---

## 9. Recommendation

### Start with Free Tier ✅
**Why**:
1. Zero risk ($4.94/month is negligible)
2. Validate approach before spending
3. 15 sentiment features is already substantial
4. Can upgrade to Premium later if needed

### Upgrade to Premium when:
1. Free tier models are profitable (positive Sharpe > 1.5)
2. Need more features for edge case improvements
3. Twitter data shows clear alpha in backtest
4. Monthly profit > $1,000 (10x the cost)

### Upgrade to Professional when:
1. Monthly profit > $10,000 (20x the cost)
2. Running multiple strategies
3. Need institutional-grade data
4. Expanding to more assets

---

## 10. Next Immediate Steps

### Today
1. ✅ S3 setup complete
2. ⏳ Setup Reddit API (30 min)
3. ⏳ Setup CryptoCompare API (10 min)
4. ⏳ Setup CoinGecko API (10 min)
5. ⏳ Test all APIs (30 min)

### This Week
6. ⏳ Implement Reddit filtering pipeline (4-6 hours)
7. ⏳ Fetch 7 years historical data (4 hours)
8. ⏳ Fetch sentiment historical data (2 hours)
9. ⏳ Engineer all features (6-8 hours)
10. ⏳ Train on GPU with full data (10 min, $0.61)

---

**Ready to start? Let's begin with API setup!**
