# Training Optimization & Data Strategy

**Created**: 2025-11-10
**Status**: Planning Phase

## Current Situation

### Data Status
- **Current**: 2 years of data (Nov 2023 - Nov 2025)
  - BTC-USD: 1,030,512 1-minute candles (~35MB raw, 228MB features)
  - ETH-USD: 1,030,512 1-minute candles (~32MB raw)
  - SOL-USD: 1,030,513 1-minute candles (~23MB raw)
- **Target**: 7 years of data (Nov 2018 - Nov 2025)
- **Missing**: 5 years of historical data

### Training Performance (Current)
- **Hardware**: CPU-only (local machine)
- **Current Speed**: 1-5 iterations/second (extremely slow)
- **Time per Epoch**: ~5 hours (with resource contention)
- **Total Training Time**: 75+ hours per model (3+ days)
- **Issue**: 3 models running in parallel causing severe CPU contention

---

## 1. Training Acceleration Strategy

### Option A: AWS GPU Instances (RECOMMENDED)

#### GPU Options & Costs

| Instance Type | GPU | vCPU | RAM | GPU Memory | Cost/Hour | Training Time | Total Cost (3 models) |
|--------------|-----|------|-----|------------|-----------|---------------|----------------------|
| **g4dn.xlarge** | T4 (16GB) | 4 | 16GB | 16GB | $0.526 | ~30min/model | **$0.79** ✅ |
| **g4dn.2xlarge** | T4 (16GB) | 8 | 32GB | 16GB | $0.752 | ~20min/model | **$0.75** ✅ |
| **g5.xlarge** | A10G (24GB) | 4 | 16GB | 24GB | $1.006 | ~15min/model | **$0.75** ✅ |
| **p3.2xlarge** | V100 (16GB) | 8 | 61GB | 16GB | $3.06 | ~10min/model | **$1.53** |

**Recommendation**: **g4dn.xlarge** or **g4dn.2xlarge**
- Best cost/performance ratio
- $0.75-0.79 total for all 3 models
- 20-30 minutes vs 3+ days on CPU
- **200x faster** than current CPU training

#### Setup Process
```bash
# 1. Launch AWS EC2 instance (g4dn.xlarge)
# 2. Install CUDA, PyTorch
# 3. Clone repo, upload feature files (228MB + 200MB + 150MB = ~578MB)
# 4. Train all 3 models sequentially
# 5. Download trained models (~20MB)
# 6. Terminate instance

# Total time: ~2 hours (including setup)
# Total cost: ~$1.50 (instance cost + data transfer)
```

### Option B: Local GPU (If Available)
- Check if your machine has NVIDIA GPU: `nvidia-smi`
- If yes, training time: ~1 hour per model on GTX 1080 Ti or better
- Free, but requires CUDA setup

### Option C: Sequential CPU Training (Fallback)
- Kill parallel training, train one model at a time
- Estimated time: 6-8 hours per model = 18-24 hours total
- Free, but slow

---

## 2. Historical Data (7 Years)

### Data Requirements

For **7 years** (Nov 2018 - Nov 2025):
- 1-minute candles: ~3.68M candles per symbol
- Storage estimate:
  - Raw data (1m): ~120MB per symbol
  - Multi-TF features: ~800MB per symbol
  - **Total**: ~3GB for 3 symbols

### Data Availability by Symbol

| Symbol | Coinbase Listing Date | Available History | Notes |
|--------|----------------------|-------------------|-------|
| **BTC-USD** | 2015 | ✅ Full 7 years | Available |
| **ETH-USD** | 2016 | ✅ Full 7 years | Available |
| **SOL-USD** | Jun 2021 | ⚠️ Only 4.4 years | Listed later |

**Issue**: SOL-USD only has 4.4 years of data on Coinbase.

**Options**:
1. Use 4.4 years for SOL, 7 years for BTC/ETH
2. Use 4.4 years for all symbols (consistent training periods)
3. Add another symbol with longer history (LTC-USD, LINK-USD)

### Fetching Historical Data

**Estimated Time**: 2-3 hours per symbol (Coinbase rate limits)

```bash
# Fetch 7 years of data
uv run python scripts/fetch_data.py --symbol BTC-USD --start 2018-11-10 --interval 1m --output data/raw
uv run python scripts/fetch_data.py --symbol ETH-USD --start 2018-11-10 --interval 1m --output data/raw
uv run python scripts/fetch_data.py --symbol SOL-USD --start 2021-06-01 --interval 1m --output data/raw

# Then engineer features (multi-TF)
uv run python scripts/engineer_features.py --symbol BTC-USD --input data/raw --output data/features
# ... repeat for ETH and SOL
```

**Cost**: Free (Coinbase public API)

---

## 3. Alternative Data Sources

### A. Sentiment Data (Twitter Alternative)

**Twitter API Issues**:
- X (Twitter) API v2 costs $100/month (Basic tier) or $5,000/month (Pro tier)
- Rate limits are very restrictive
- Historical data not available in cheap tiers

**FREE Alternatives**:

#### Option 1: Reddit Sentiment (RECOMMENDED)
**Source**: Reddit r/cryptocurrency, r/bitcoin, r/ethereum
**API**: PRAW (Python Reddit API Wrapper) - FREE
**Data Available**:
- Post titles, comments, scores, timestamps
- Subreddit activity metrics
- FREE with Reddit API (1,000 requests/minute)

```python
# Example: Fetch Bitcoin sentiment from Reddit
import praw

reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    user_agent="CryptoBot/1.0"
)

# Get posts from r/bitcoin
for post in reddit.subreddit("bitcoin").hot(limit=100):
    print(post.title, post.score, post.created_utc)
```

**Pros**:
- Completely FREE
- Good crypto community discussion
- Easy to get historical data
- Higher signal quality than Twitter (less bots)

**Cons**:
- Less real-time than Twitter
- Smaller volume

#### Option 2: Fear & Greed Index
**Source**: Alternative.me Crypto Fear & Greed Index
**API**: Free, no authentication required
**Data**: Daily sentiment score (0-100)

```bash
# Free API endpoint
curl https://api.alternative.me/fng/?limit=365
```

**Pros**:
- Completely FREE
- Pre-aggregated sentiment (0-100 scale)
- Historical data available
- Widely used in crypto trading

**Cons**:
- Daily granularity only (not minute-level)
- Single metric (less nuanced)

#### Option 3: CryptoPanic News Sentiment
**Source**: CryptoPanic.com
**API**: Free tier: 500 requests/day
**Data**: Crypto news with sentiment labels

```bash
# Free API
curl "https://cryptopanic.com/api/v1/posts/?auth_token=YOUR_TOKEN&currencies=BTC"
```

**Pros**:
- FREE tier available
- News + sentiment labels
- Good for event detection

**Cons**:
- Limited to 500 requests/day (free tier)
- Paid tier: $30/month for 10,000 requests/day

#### Option 4: Google Trends
**Source**: Google Trends (pytrends library)
**API**: FREE (unofficial)
**Data**: Search interest over time for crypto keywords

```python
from pytrends.request import TrendReq

pytrends = TrendReq()
pytrends.build_payload(['Bitcoin', 'BTC'], timeframe='today 5-y')
data = pytrends.interest_over_time()
```

**Pros**:
- Completely FREE
- Good proxy for retail interest
- Historical data available

**Cons**:
- Not real sentiment, just search volume
- Weekly/daily granularity
- Rate limits (use carefully)

### B. On-Chain Data

#### Glassnode API
**Cost**: $29/month (Starter) - $799/month (Professional)
**Data**: On-chain metrics (addresses, transactions, exchange flows)

**Recommendation**: Skip for now (expensive)

#### Free On-Chain Data
- **Blockchain.com API** (Free for Bitcoin)
- **Etherscan API** (Free tier: 5 calls/second)

### C. Order Book Data

**Coinbase WebSocket** (FREE)
- Real-time order book depth
- Trade executions
- Can add as additional features

---

## 4. Recommended Data Integration Plan

### Phase 1: Reddit Sentiment (FREE) ✅
1. Setup Reddit API (FREE, 5 minutes)
2. Fetch historical posts from r/bitcoin, r/ethereum, r/solana
3. Calculate sentiment scores using VADER or TextBlob
4. Aggregate to 1-hour windows (align with our 1h multi-TF features)
5. Add as 3 new features:
   - `reddit_sentiment_1h` (-1 to +1)
   - `reddit_volume_1h` (post count)
   - `reddit_engagement_1h` (score sum)

**Estimated Time**: 4-6 hours (implementation + historical fetch)
**Cost**: $0 (FREE)

### Phase 2: Fear & Greed Index (FREE) ✅
1. Fetch historical F&G index (1 API call, 30 seconds)
2. Interpolate daily values to 1-hour windows
3. Add as 1 new feature:
   - `fear_greed_index` (0-100)

**Estimated Time**: 1 hour
**Cost**: $0 (FREE)

### Phase 3: Google Trends (FREE) ✅
1. Fetch search interest for "Bitcoin", "Ethereum", "Solana"
2. Normalize and align to 1-hour windows
3. Add as 3 new features:
   - `search_interest_btc`
   - `search_interest_eth`
   - `search_interest_sol`

**Estimated Time**: 2-3 hours
**Cost**: $0 (FREE)

### Total Additional Features: +7 sentiment features
- Current: 50 features
- After integration: **57 features**

---

## 5. Cost Breakdown & Spending Plan

### One-Time Costs

| Item | Cost | Notes |
|------|------|-------|
| **AWS GPU Training (g4dn.xlarge)** | $1.50 | 3 models, ~2 hours total |
| **Reddit API Setup** | $0 | FREE |
| **Data Storage (S3)** | $0.10 | 3GB @ $0.023/GB/month |
| **Total One-Time** | **$1.60** | ✅ Very affordable |

### Monthly Recurring Costs

| Item | Cost/Month | Notes |
|------|------------|-------|
| **Kafka Infrastructure** | $0 | Local Docker (free) |
| **Reddit API** | $0 | FREE tier (1,000 req/min) |
| **Fear & Greed API** | $0 | FREE |
| **Google Trends** | $0 | FREE (use sparingly) |
| **Coinbase API** | $0 | FREE (public data) |
| **AWS EC2 (if running Kafka)** | $30-50 | t3.medium (~$30/month) |
| **AWS S3 Storage** | $1-2 | 30GB @ $0.023/GB/month |
| **Total Monthly (Local)** | **$0** | If running locally |
| **Total Monthly (AWS)** | **$31-52** | If deploying to AWS |

### Optional Paid Services (Not Recommended Yet)

| Service | Cost | Value Proposition |
|---------|------|-------------------|
| **Twitter API (Basic)** | $100/month | ❌ Too expensive for value |
| **CryptoPanic Pro** | $30/month | ⚠️ Consider after profitability |
| **Glassnode Starter** | $29/month | ⚠️ Consider after profitability |
| **AWS SageMaker Training** | $50-100/month | ❌ Overkill for our needs |

---

## 6. Recommended Action Plan

### Immediate (This Week)

1. **✅ Stop parallel training** (killing ETH/SOL for now)
2. **✅ Let BTC finish on CPU** (current progress: ~50 hours remaining)
3. **⏳ Setup Reddit API** (30 minutes, FREE)
4. **⏳ Fetch Fear & Greed historical data** (30 minutes, FREE)
5. **⏳ Fetch 7 years of BTC/ETH data** (3-4 hours, FREE)

### Next Week

6. **⏳ Integrate Reddit sentiment** (4-6 hours implementation)
7. **⏳ Engineer features with 7 years + sentiment data** (6-8 hours)
8. **⏳ Setup AWS GPU instance** (1 hour)
9. **⏳ Train all 3 models on AWS GPU** (~30 min per model = 1.5 hours)
10. **⏳ Evaluate models** (1 hour)

### Month 1

11. **⏳ Deploy Kafka pipeline locally** (Docker, FREE)
12. **⏳ Backtest with historical data** (2-3 days)
13. **⏳ Paper trading (dry-run)** (1-2 weeks)

### Month 2+

14. **⏳ Deploy to AWS (if needed)** ($30-50/month)
15. **⏳ Live trading (FTMO challenge)** ($0 upfront, profit sharing later)

---

## 7. Training Optimization Implementation

### Current vs Optimized

| Metric | Current (CPU) | Optimized (GPU) | Improvement |
|--------|--------------|-----------------|-------------|
| **Hardware** | Local CPU | AWS g4dn.xlarge | N/A |
| **Time/Epoch** | ~5 hours | ~2 minutes | **150x faster** |
| **Time/Model** | ~75 hours | ~30 minutes | **150x faster** |
| **Total Time (3 models)** | ~225 hours (9 days) | ~1.5 hours | **150x faster** |
| **Cost** | $0 (electricity) | $1.50 | Minimal |
| **Resource Contention** | Severe | None | N/A |

### GPU Setup Script

```bash
#!/bin/bash
# AWS GPU Instance Setup Script

# 1. Launch instance (AWS Console or CLI)
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \  # Deep Learning AMI
  --instance-type g4dn.xlarge \
  --key-name your-key \
  --security-group-ids sg-xxx \
  --subnet-id subnet-xxx

# 2. SSH into instance
ssh -i your-key.pem ubuntu@<instance-ip>

# 3. Clone repo and setup
git clone https://github.com/your-repo/crpbot.git
cd crpbot
pip install -r requirements.txt

# 4. Upload feature files (use scp or S3)
scp -i your-key.pem data/features/*.parquet ubuntu@<instance-ip>:~/crpbot/data/features/

# 5. Train models
uv run python apps/trainer/main.py --task lstm --coin BTC --epochs 15 --device cuda
uv run python apps/trainer/main.py --task lstm --coin ETH --epochs 15 --device cuda
uv run python apps/trainer/main.py --task lstm --coin SOL --epochs 15 --device cuda

# 6. Download trained models
scp -i your-key.pem ubuntu@<instance-ip>:~/crpbot/models/*.pt ./models/

# 7. Terminate instance
aws ec2 terminate-instances --instance-ids i-xxx
```

---

## 8. Sentiment Feature Engineering

### Reddit Sentiment Implementation

```python
# apps/data_sources/reddit_sentiment.py

import praw
import pandas as pd
from datetime import datetime, timedelta
from textblob import TextBlob  # or use VADER

class RedditSentimentFetcher:
    """Fetch and analyze Reddit sentiment for crypto symbols."""

    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )

    def fetch_historical_sentiment(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        subreddits: list[str] = None
    ) -> pd.DataFrame:
        """Fetch historical Reddit posts and calculate sentiment."""

        if subreddits is None:
            subreddits = {
                "BTC": ["bitcoin", "cryptocurrency"],
                "ETH": ["ethereum", "cryptocurrency"],
                "SOL": ["solana", "cryptocurrency"]
            }[symbol.split("-")[0]]

        posts = []
        for subreddit_name in subreddits:
            subreddit = self.reddit.subreddit(subreddit_name)

            # Fetch posts in date range
            for post in subreddit.search(
                symbol,
                sort="new",
                time_filter="all",
                limit=None
            ):
                if start_date <= datetime.fromtimestamp(post.created_utc) <= end_date:
                    posts.append({
                        "timestamp": datetime.fromtimestamp(post.created_utc),
                        "title": post.title,
                        "score": post.score,
                        "num_comments": post.num_comments,
                        "subreddit": subreddit_name
                    })

        df = pd.DataFrame(posts)

        # Calculate sentiment
        df["sentiment"] = df["title"].apply(
            lambda x: TextBlob(x).sentiment.polarity
        )

        # Aggregate to 1-hour windows
        df = df.set_index("timestamp")
        hourly = df.resample("1H").agg({
            "sentiment": "mean",
            "score": "sum",
            "num_comments": "sum",
            "title": "count"  # post volume
        })

        hourly.columns = [
            "reddit_sentiment_1h",
            "reddit_engagement_1h",
            "reddit_comments_1h",
            "reddit_volume_1h"
        ]

        return hourly
```

---

## 9. Questions & Decisions Needed

### Question 1: Data History
**Options**:
- A) Use 7 years for BTC/ETH, 4.4 years for SOL ✅ (RECOMMENDED)
- B) Use 4.4 years for all (consistent)
- C) Replace SOL with LINK-USD (has 7 years)

**Your preference?**

### Question 2: Training Location
**Options**:
- A) AWS GPU (g4dn.xlarge) - $1.50, 2 hours ✅ (RECOMMENDED)
- B) Sequential CPU - FREE, 24 hours
- C) Check if local GPU available - FREE, 3 hours

**Your preference?**

### Question 3: Sentiment Data Priority
**Options**:
- A) Reddit only (FREE, good quality) ✅ (RECOMMENDED)
- B) Reddit + Fear & Greed + Google Trends (all FREE)
- C) Include Twitter ($100/month) - NOT RECOMMENDED

**Your preference?**

### Question 4: Monthly Budget
**Options**:
- A) $0/month (run everything locally) ✅ (RECOMMENDED for now)
- B) $30-50/month (AWS EC2 + S3)
- C) $100+/month (Add paid APIs)

**Your preference?**

---

## 10. Summary

### Cost Analysis
- **One-time**: $1.50 (AWS GPU training)
- **Monthly**: $0 (if local) or $30-50 (if AWS deployment)
- **Total Year 1**: ~$1.50 (training) + $0-$600 (infrastructure)

### Time Investment
- **Data fetching**: 6-8 hours (one-time)
- **Sentiment integration**: 6-8 hours (one-time)
- **GPU training**: 2 hours (one-time)
- **Total**: ~16-18 hours of work

### Expected Improvements
- **Training speed**: 150x faster (225 hours → 1.5 hours)
- **Data quality**: 3.5x more data (2 years → 7 years)
- **Features**: +7 sentiment features (50 → 57 features)
- **Model accuracy**: Expected +5-10% improvement

### Next Immediate Steps
1. Decide on questions above
2. Stop ETH/SOL training (save CPU resources)
3. Setup Reddit API (30 min)
4. Fetch 7 years of data (4 hours)
5. Setup AWS GPU for training (1 hour)

**Ready to proceed?** Let me know your preferences on the 4 questions above!
