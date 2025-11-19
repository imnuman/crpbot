# V7 Project Status & Implementation Roadmap

**Date**: 2025-11-17
**Status**: Planning Complete → Ready to Implement
**Approach**: Manual Trading Signal System (Renaissance Technologies principles)
**Budget**: $229/month (CoinGecko $129 + DeepSeek API $100)

---

## 🎯 Executive Summary

We are building **V7 Ultimate**: A clean, manual trading signal generation system that combines mathematical analysis with AI synthesis to produce high-quality trading signals for cryptocurrency markets.

**Key Decisions Made**:
- ✅ **Manual trading only** (no automated execution)
- ✅ **Pure mathematical/statistical approach** (no ML model reliance)
- ✅ **Clean system design** (exclude V6 Fixed models - deemed faulty)
- ✅ **Step-based implementation** (not timeline-based)
- ✅ **Renaissance Technologies methodology** (quantitative, diversified, data-driven)

---

## 📊 Current Infrastructure (What We Have)

### 1. Data Pipeline ✅

**Raw Data** (`data/raw/`):
```
✅ BTC-USD: 35 MB (1m candles, 2023-11-10 to 2025-11-10)
✅ ETH-USD: 32 MB (1m candles, 2023-11-10 to 2025-11-10)
✅ SOL-USD: 23 MB (1m candles, 2023-11-10 to 2025-11-10)
✅ Multi-timeframe data: 1m, 5m, 15m, 1h for all symbols
```

**Feature Data** (`data/features/`):
```
✅ features_BTC-USD_1m_latest.parquet (293 MB, 39 columns)
✅ features_ETH-USD_1m_latest.parquet (281 MB, 39 columns)
✅ features_SOL-USD_1m_latest.parquet (262 MB, 39 columns)
```

**Feature Columns** (39 total):
- 5 OHLCV columns
- 31 engineered features (RSI, MACD, Bollinger Bands, ATR, volume, session indicators)
- 3 categorical columns (volatility regime classification)

### 2. Codebase Structure ✅

**Libraries** (`libs/`):
```
libs/
├── aws/                 # S3 client, AWS secrets management
├── confidence/          # Confidence scaling, calibration
├── config/              # Configuration management
├── constants/           # Trading constants (symbols, thresholds)
├── data/                # Data providers (Coinbase, CCXT, yfinance)
│   ├── coinbase.py      # ✅ Coinbase API integration
│   ├── yfinance_provider.py  # ✅ Yahoo Finance integration
│   └── provider.py
├── db/                  # Database models, auto-learning
│   ├── database.py      # ✅ PostgreSQL/SQLite abstraction
│   └── models.py        # ✅ Signal, trade, prediction models
├── features/            # Feature engineering utilities
└── utils/               # Timezone, logging utilities
```

**Applications** (`apps/`):
```
apps/
├── dashboard/           # ✅ Real-time signal dashboard (Flask + WebSocket)
│   ├── app.py
│   ├── templates/dashboard.html
│   └── static/css/dashboard.css
├── runtime/             # ✅ Signal generation engine
│   ├── main.py          # Entry point
│   ├── ensemble.py      # Model ensemble logic
│   ├── data_fetcher.py  # Real-time data fetching
│   ├── signal_gen.py    # Signal generation
│   ├── telegram_bot.py  # ✅ Telegram notifications
│   └── ftmo_rules.py    # Risk management rules
└── trainer/             # Model training (currently V5/V6 - will exclude for V7)
    ├── main.py
    └── data_pipeline.py
```

### 3. Data Sources (Currently Active) ✅

**Configured APIs**:
```bash
✅ COINBASE_API (JWT authentication)
   - OHLCV data (1m, 5m, 15m, 1h)
   - Real-time WebSocket feed
   - Order book data

✅ COINGECKO_API (Analyst Plan - $129/month)
   - API Key: CG-VQhq64e59sGxchtK8mRgdxXW
   - Access: Analyst insights, sentiment, whale activity
   - Rate limit: Higher tier (need to verify exact limits)

✅ YAHOO_FINANCE (Free)
   - Macro data: DXY, Gold (GC=F), Yields (^TNX), S&P500 (^GSPC)
   - Access via yfinance library
```

**Not Yet Integrated** (Need to Add):
```bash
⏹️ DERIBIT (Free for options data)
   - Put/Call ratio
   - Implied volatility
   - Options sentiment

⏹️ DEEPSEEK API ($100/month planned)
   - LLM for signal synthesis
   - Model: deepseek-chat
   - Need to subscribe and get API key
```

### 4. Existing Models (V5/V6) ⚠️

**Models Directory** (`models/`):
```
models/promoted/
├── lstm_BTC-USD_v6_enhanced.pt (242 KB) - Nov 16
├── lstm_ETH-USD_v6_enhanced.pt (242 KB) - Nov 16
├── lstm_SOL-USD_v6_enhanced.pt (MISSING - only 2/3 present)
├── lstm_BTC-USD_1m_v6_real.pt (231 KB)
├── lstm_ETH-USD_1m_v6_real.pt (231 KB)
├── lstm_SOL-USD_1m_v6_real.pt (231 KB)
└── (older v5 models...)
```

**User Decision**: ❌ **DO NOT USE** - Models deemed "faulty" and not good enough for production.

**Rationale for V7 Ultimate**:
- V6 Fixed models showed 100% confidence issue (fixed with StandardScaler + temperature scaling)
- Even with calibration (T=2.5), confidence still 78-92% (above 60-70% target)
- User wants clean system without relying on potentially flawed ML predictions
- Focus on pure mathematical/statistical analysis instead

### 5. Infrastructure & Deployment ✅

**Local Machine** (`/home/numan/crpbot`):
- User: numan
- Role: QC Review, Documentation, Local Testing
- Python: 3.12 with uv package manager
- Database: SQLite (dev)

**Cloud Server** (`root@178.156.136.185:~/crpbot`):
- User: root
- Role: Production runtime, Training
- Python: 3.10+ with uv
- Database: PostgreSQL RDS (configured)
- AWS: S3 for data/models (configured)
- IP: 178.156.136.185
- Dashboard: http://178.156.136.185:5000 (✅ operational)

**Git Sync**:
- Repository: `github.com/imnuman/crpbot`
- Branch: main
- Both environments sync via git push/pull

### 6. Database Schema ✅

**PostgreSQL/SQLite** (dual-support):
```sql
-- Existing tables (from libs/db/models.py):
✅ signals         # Trading signals with confidence, tier, reasoning
✅ predictions     # Model predictions with features
✅ trades          # Trade execution records (for future)
✅ market_data     # Real-time market data cache
```

**Key Fields** (signals table):
- timestamp, symbol, direction (BUY/SELL/HOLD)
- confidence (0-100), tier (HIGH/MEDIUM/LOW)
- entry_price, stop_loss, take_profit
- reasoning (text explanation)
- model_version, features_used

---

## 🚀 What We're Building (V7 Ultimate)

### Vision

A **manual trading signal generation system** that:
1. Collects data from multiple high-quality sources
2. Applies 6 mathematical/statistical theories for analysis
3. Uses DeepSeek LLM to synthesize insights into actionable signals
4. Presents signals via dashboard + Telegram for manual review
5. Tracks signal performance for continuous learning

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA COLLECTION LAYER                     │
├─────────────────────────────────────────────────────────────┤
│  CoinGecko Analyst  │  Coinbase OHLCV  │  Coinbase WebSocket│
│  Yahoo Finance      │  Deribit Options │  (Future: More)    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│               MATHEMATICAL ANALYSIS LAYER                    │
├─────────────────────────────────────────────────────────────┤
│  1. Shannon Entropy      → Market predictability             │
│  2. Hurst Exponent       → Trend vs mean-reversion          │
│  3. Markov Chain         → 6-state regime detection         │
│  4. Kalman Filter        → Price denoising + momentum       │
│  5. Bayesian Inference   → Belief updating                  │
│  6. Monte Carlo          → Risk simulation (10k scenarios)  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   TECHNICAL ANALYSIS LAYER                   │
├─────────────────────────────────────────────────────────────┤
│  RSI, MACD, Bollinger Bands, ATR, Volume Analysis           │
│  Support/Resistance, Session Indicators                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    DEEPSEEK LLM SYNTHESIS                    │
├─────────────────────────────────────────────────────────────┤
│  Prompt: Comprehensive context from all sources              │
│  Output: BUY/SELL/HOLD + Confidence + Reasoning             │
│          Entry/SL/TP + Risk Level + Key Factors             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  SIGNAL GENERATION & DELIVERY                │
├─────────────────────────────────────────────────────────────┤
│  Dashboard (Real-time)  │  Telegram Bot  │  Database Log    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              MANUAL TRADING DECISION BY USER                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│           SIGNAL TRACKING & PERFORMANCE LEARNING             │
├─────────────────────────────────────────────────────────────┤
│  Track signal outcomes → Update Bayesian priors             │
│  Identify best-performing combinations                       │
└─────────────────────────────────────────────────────────────┘
```

### Expected Performance

**Initial Target** (Weeks 1-4):
- Win Rate: 58-65% (up from baseline 50%)
- Signal Volume: 5-10 signals/day (filtered for quality)
- Confidence Range: 60-75% (realistic, calibrated)

**Long-term Goal** (Months 3-6):
- Win Rate: 70-75% (with learning and refinement)
- Sharpe Ratio: 1.9-2.2
- Drawdown: <15%
- Renaissance Technologies style performance

### Budget Breakdown

```
Monthly Costs:
├── CoinGecko Analyst Plan:  $129/month (CONFIRMED - already subscribed)
├── DeepSeek API:            $100/month (PLANNED - need to subscribe)
├── AWS (S3 + RDS):           ~$15/month (current usage)
└── Server Hosting:           ~$10/month (existing)
─────────────────────────────────────────
Total:                        $254/month
```

---

## 🛠️ Implementation Roadmap (Step-by-Step)

### STEP 1: Data Infrastructure Setup

**Goal**: Connect all 5 data sources and build unified data fetcher

**Sub-steps**:
1. **CoinGecko Analyst API Integration**
   - File: `libs/data/coingecko_pro_client.py` (create)
   - Methods: `get_analyst_insights()`, `get_whale_activity()`, `get_sentiment()`
   - Test with API key: `CG-VQhq64e59sGxchtK8mRgdxXW`

2. **Coinbase WebSocket Integration**
   - File: `libs/data/coinbase_websocket.py` (create)
   - Subscribe to ticker, L2 orderbook, matches
   - Calculate: spread, order imbalance, whale trades

3. **Yahoo Finance Macro Data**
   - File: `libs/data/yahoo_finance_client.py` (enhance existing)
   - Fetch: DXY, GC=F (Gold), ^TNX (10Y Yield), ^GSPC (S&P500)
   - Cache data (update every 1 hour)

4. **Deribit Options Integration**
   - File: `libs/data/deribit_client.py` (create)
   - Get Put/Call ratio, IV rank, options sentiment
   - Focus on BTC/ETH only (Deribit main markets)

5. **Unified Data Fetcher**
   - File: `apps/runtime/v7_data_fetcher.py` (create)
   - Orchestrate all 5 sources
   - Return unified data structure for analysis

**Deliverable**: `v7_data_fetcher.py` that returns complete market context every 5 minutes

---

### STEP 2: Mathematical Analysis Framework

**Goal**: Implement 6 mathematical theories for signal generation

**Sub-steps**:
1. **Shannon Entropy (Information Theory)**
   - File: `libs/analytics/entropy.py` (create)
   - Calculate market entropy: H(X) = -Σ p(x) log₂ p(x)
   - Output: Predictability score (0-3, lower = more predictable)

2. **Hurst Exponent (Chaos Theory)**
   - File: `libs/analytics/hurst.py` (create)
   - R/S Analysis method
   - Output: H value (0-1, <0.5 mean-reverting, >0.5 trending)

3. **Markov Chain (State Detection)**
   - File: `libs/analytics/markov_chain.py` (create)
   - 6 states: TRENDING_BULLISH, TRENDING_BEARISH, RANGING_CALM, RANGING_VOLATILE, BREAKOUT_FORMING, REVERSAL_LIKELY
   - Transition matrix updated based on features
   - Output: Current state + confidence

4. **Kalman Filter (Signal Processing)**
   - File: `libs/analytics/kalman_filter.py` (create)
   - Denoise price data
   - Output: smoothed_price, velocity (momentum), acceleration (trend strength)

5. **Bayesian Inference (Belief Updating)**
   - File: `libs/analytics/bayesian.py` (create)
   - P(Signal|Data) = P(Data|Signal) × P(Signal) / P(Data)
   - Online learning: Update win probability based on outcomes
   - Output: Posterior probability of signal success

6. **Monte Carlo Simulation (Risk Assessment)**
   - File: `libs/analytics/monte_carlo.py` (create)
   - Run 10,000 scenarios using Geometric Brownian Motion
   - Output: expected_return, win_probability, 90%/95%/99% confidence intervals, max_drawdown

**Deliverable**: 6 analytics modules that take price/feature data and return mathematical insights

---

### STEP 3: Technical Analysis Layer

**Goal**: Leverage existing feature engineering, add missing indicators

**Sub-steps**:
1. **Enhance Feature Engineering** (`libs/features/v6_enhanced_features.py`)
   - Add: Support/Resistance levels (recent highs/lows)
   - Add: Fibonacci retracement levels
   - Add: Volume profile (VWAP, VPOC)
   - Keep existing: RSI, MACD, BB, ATR, session indicators

2. **Create Technical Signal Generator**
   - File: `libs/analytics/technical_signals.py` (create)
   - RSI divergence detection
   - MACD crossover analysis
   - Bollinger Band squeeze detection
   - Volume spike detection

3. **Session & Time Analysis**
   - Tokyo/London/NY session strength
   - Best trading hours based on historical volatility

4. **Combine Technical + Mathematical**
   - File: `apps/runtime/v7_analyzer.py` (create)
   - Orchestrate: technical signals + 6 mathematical analyses
   - Return: comprehensive analysis object

**Deliverable**: `v7_analyzer.py` that provides full technical + mathematical context

---

### STEP 4: DeepSeek LLM Integration

**Goal**: Subscribe to DeepSeek API and build signal synthesis engine

**Sub-steps**:
1. **Subscribe to DeepSeek API**
   - Visit: https://platform.deepseek.com
   - Subscribe: $100/month plan
   - Get API key and store in `.env`

2. **Create DeepSeek Client**
   - File: `libs/deepseek/deepseek_client.py` (create)
   - Initialize with API key
   - Method: `analyze(context: Dict) -> Dict`

3. **Build Comprehensive Prompt Template**
   - File: `libs/deepseek/market_prompts.py` (create)
   - Template includes:
     - Current price, trend, volatility
     - All 6 mathematical analyses
     - Technical indicators (RSI, MACD, BB, ATR)
     - CoinGecko analyst insights
     - Order flow metrics
     - Macro context (DXY, Gold, Yields, S&P500)
     - Options sentiment (Put/Call ratio, IV)
   - Request structured JSON output

4. **Structured Output Parser**
   - File: `libs/deepseek/output_parser.py` (create)
   - Parse DeepSeek JSON response
   - Validate: recommendation, confidence, reasoning, prices, risk_level

5. **Test & Calibrate**
   - Run on historical data (past 7 days)
   - Verify output quality
   - Adjust prompt if needed for better reasoning

**Deliverable**: `deepseek_client.py` that synthesizes all data into trading signal

---

### STEP 5: Signal Generation System

**Goal**: Build main signal generation loop and database logging

**Sub-steps**:
1. **Create V7 Main Loop**
   - File: `apps/runtime/v7_main.py` (create)
   - Every 5 minutes:
     1. Fetch all data sources
     2. Run mathematical analyses
     3. Calculate technical indicators
     4. Send to DeepSeek
     5. Parse signal
     6. Log to database
     7. Send to dashboard + Telegram

2. **Signal Validator**
   - File: `apps/runtime/v7_signal_validator.py` (create)
   - Check: confidence >= 60%
   - Check: reasoning is present and logical
   - Check: entry/SL/TP are valid prices
   - Filter out low-quality signals

3. **Database Schema Extension**
   - Update `libs/db/models.py`
   - Add fields to `signals` table:
     - entropy, hurst, markov_state
     - kalman_smoothed, bayesian_posterior
     - monte_carlo_confidence
     - deepseek_reasoning (text)
     - data_sources_used (JSON)

4. **Signal Logging**
   - File: `apps/runtime/v7_signal_logger.py` (create)
   - Log complete signal with all context
   - Store for future analysis and learning

5. **Rate Limiting**
   - Max 10 signals/hour across all symbols
   - Max 5 high-confidence (>=75%) signals/hour
   - Prevent signal spam

**Deliverable**: `v7_main.py` running on cloud server, generating signals every 5 minutes

---

### STEP 6: User Interface - Dashboard

**Goal**: Enhance existing dashboard to show V7 signals with all context

**Sub-steps**:
1. **Update Dashboard Template** (`apps/dashboard/templates/dashboard.html`)
   - Show: Symbol, Direction, Confidence, Tier
   - Show: Entry Price, Stop Loss, Take Profit
   - Show: Risk Level (LOW/MEDIUM/HIGH)
   - Show: DeepSeek Reasoning (expandable text)
   - Show: Mathematical Analyses (entropy, hurst, markov state, etc.)

2. **Add Signal History View**
   - Last 24 hours of signals
   - Win/Loss tracking (manual input by user)
   - Performance metrics (win rate, avg confidence)

3. **Real-time Updates** (already implemented via WebSocket)
   - Keep 1-second refresh rate
   - Highlight new signals

4. **Mobile-Friendly Design**
   - Responsive CSS for phone/tablet viewing
   - Critical info at top

**Deliverable**: Enhanced dashboard at http://178.156.136.185:5000

---

### STEP 7: User Interface - Telegram Bot

**Goal**: Send signals to Telegram for mobile notifications

**Sub-steps**:
1. **Enhance Telegram Bot** (`apps/runtime/telegram_bot.py` already exists)
   - Format signal message:
     ```
     🚨 V7 SIGNAL - BTC-USD

     Direction: BUY
     Confidence: 72% (HIGH)
     Entry: $96,420
     Stop Loss: $95,800
     Take Profit: $97,500
     Risk: MEDIUM

     🧠 Reasoning:
     [DeepSeek reasoning here]

     📊 Analysis:
     • Entropy: 1.2 (predictable)
     • Hurst: 0.62 (trending)
     • Markov: BREAKOUT_FORMING
     • Monte Carlo: 68% win probability
     ```

2. **Add Signal Confirmation**
   - User can reply: ✅ Took trade, ❌ Skipped, 📊 Monitoring
   - Track user decisions for learning

3. **Daily Summary**
   - End of day: Send summary of signals, win rate, performance

4. **Alert Customization**
   - User can set: min confidence threshold
   - User can mute: certain symbols, time ranges

**Deliverable**: Telegram bot sending V7 signals with full context

---

### STEP 8: Signal Tracking & Learning

**Goal**: Track signal outcomes and improve system over time

**Sub-steps**:
1. **Manual Outcome Entry**
   - Dashboard button: "Mark as Win/Loss/Breakeven"
   - User inputs: actual entry, exit, P/L
   - Store in database

2. **Performance Analytics**
   - File: `apps/runtime/v7_performance.py` (create)
   - Calculate: win rate by symbol, time, market state
   - Identify: best-performing Markov states, entropy ranges, Hurst values

3. **Bayesian Learning Loop**
   - Update Bayesian priors based on outcomes
   - If TRENDING_BULLISH signals win 75% of time → increase prior for that state
   - Continuous learning without retraining models

4. **Performance Dashboard**
   - Show: cumulative P/L (paper trading)
   - Show: win rate trend over time
   - Show: best/worst signal characteristics

**Deliverable**: Self-improving signal system via Bayesian learning

---

### STEP 9: Validation & Testing

**Goal**: Validate system on paper trading before real money

**Sub-steps**:
1. **Paper Trading Mode**
   - Track all signals but don't execute
   - Simulate trades at entry/SL/TP prices
   - Calculate virtual P/L

2. **7-Day Validation Period**
   - Run system 24/7 for 7 days
   - Collect at least 50 signals
   - Target: 60%+ win rate

3. **Statistical Validation**
   - Hypothesis test: Is win rate > 50%? (p < 0.05)
   - Check: Sharpe ratio > 1.0
   - Check: Max drawdown < 20%

4. **Review & Iterate**
   - If performance < target: adjust prompt, thresholds
   - If performance >= target: proceed to production

**Deliverable**: 7-day paper trading report with win rate, Sharpe, drawdown

---

### STEP 10: Production Deployment

**Goal**: Deploy to cloud server for 24/7 operation

**Sub-steps**:
1. **Cloud Server Setup**
   - Already configured: root@178.156.136.185
   - Install dependencies: `uv sync`
   - Set environment variables in `.env`

2. **Systemd Service** (for auto-restart)
   - File: `/etc/systemd/system/crpbot-v7.service`
   - Auto-start on boot
   - Auto-restart on crash

3. **Monitoring & Logging**
   - Logs to: `/var/log/crpbot/v7_runtime.log`
   - Monitor: Signal volume, API errors, database writes
   - Alert: If no signals for 30 minutes (possible system failure)

4. **Go Live**
   - Start systemd service
   - Monitor dashboard for first 24 hours
   - User manually trades signals

**Deliverable**: Production V7 system running 24/7 on cloud server

---

## 📍 Where to Start (Your Next Actions)

### Immediate Prerequisites (Before STEP 1)

**1. Subscribe to DeepSeek API**
- Visit: https://platform.deepseek.com
- Sign up and subscribe to API plan ($100/month)
- Get API key
- Add to `.env`:
  ```bash
  DEEPSEEK_API_KEY=sk-...
  ```

**2. Verify CoinGecko Analyst Access**
- Test current API key: `CG-VQhq64e59sGxchtK8mRgdxXW`
- Confirm analyst endpoints are accessible
- Check rate limits for plan

**3. Create New Git Branch**
```bash
git checkout -b feature/v7-ultimate
```

**4. Update PROJECT_MEMORY.md**
```markdown
**Last Session**: V7 Ultimate Implementation - STEP 1 Started
**Current Phase**: V7 Ultimate - Data Infrastructure Setup
**Status**: In Progress - CoinGecko API Integration
```

---

### STEP 1.1: CoinGecko Analyst API Integration (START HERE)

**File to Create**: `libs/data/coingecko_pro_client.py`

**Starter Code**:
```python
"""
CoinGecko Analyst Plan API Client
Provides analyst insights, sentiment, whale activity for crypto markets
"""

import requests
from typing import Dict, List
from libs.config.config import config


class CoinGeckoProClient:
    """Client for CoinGecko Analyst Plan (Professional Tier)"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.COINGECKO_API_KEY
        self.base_url = "https://pro-api.coingecko.com/api/v3"
        self.headers = {
            "accept": "application/json",
            "x-cg-pro-api-key": self.api_key
        }

    def get_coin_data(self, coin_id: str) -> Dict:
        """
        Fetch comprehensive coin data including analyst insights

        Args:
            coin_id: CoinGecko coin ID (bitcoin, ethereum, solana)

        Returns:
            Dict with price, market cap, sentiment, developer activity, community
        """
        endpoint = f"{self.base_url}/coins/{coin_id}"
        params = {
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "true",
            "developer_data": "true",
            "sparkline": "false"
        }

        response = requests.get(endpoint, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

    def get_market_sentiment(self, coin_id: str) -> Dict:
        """
        Extract sentiment metrics from coin data

        Returns:
            {
                'sentiment_up': float (0-100),
                'sentiment_down': float (0-100),
                'community_score': float (0-100),
                'developer_score': float (0-100)
            }
        """
        data = self.get_coin_data(coin_id)

        return {
            'sentiment_up': data.get('sentiment_votes_up_percentage', 50),
            'sentiment_down': data.get('sentiment_votes_down_percentage', 50),
            'community_score': data.get('community_score', 0),
            'developer_score': data.get('developer_score', 0),
            'market_cap_rank': data.get('market_cap_rank', 999)
        }

    # TODO: Add methods for:
    # - get_whale_activity() - large transactions
    # - get_trending_coins() - what's hot right now
    # - get_fear_greed_index() - market-wide sentiment


# Symbol mapping: Our format → CoinGecko ID
SYMBOL_TO_COINGECKO_ID = {
    'BTC-USD': 'bitcoin',
    'ETH-USD': 'ethereum',
    'SOL-USD': 'solana'
}


if __name__ == "__main__":
    # Test the client
    client = CoinGeckoProClient()

    for symbol, coin_id in SYMBOL_TO_COINGECKO_ID.items():
        print(f"\n{symbol} ({coin_id}):")
        sentiment = client.get_market_sentiment(coin_id)
        print(f"  Sentiment: {sentiment['sentiment_up']:.1f}% UP, {sentiment['sentiment_down']:.1f}% DOWN")
        print(f"  Community Score: {sentiment['community_score']:.1f}")
        print(f"  Developer Score: {sentiment['developer_score']:.1f}")
```

**Tasks for STEP 1.1**:
1. Create the file above
2. Test with your API key
3. Add additional methods for whale activity, trending coins
4. Create unit test: `tests/unit/test_coingecko_pro_client.py`
5. Document API endpoints used in comments

**Expected Time**: 2-3 hours

**Ready**: Once this works, move to STEP 1.2 (Coinbase WebSocket Integration)

---

## 📊 Success Metrics

### Short-term (Week 1-2)
- ✅ All 5 data sources integrated and tested
- ✅ 6 mathematical modules implemented and validated
- ✅ DeepSeek API integrated with structured output
- ✅ Signal generation running every 5 minutes
- ✅ Dashboard showing signals in real-time

### Mid-term (Week 3-4)
- ✅ 50+ signals generated in paper trading
- ✅ Win rate >= 60%
- ✅ Signal quality validated by user
- ✅ Telegram notifications working
- ✅ Performance tracking implemented

### Long-term (Month 2-3)
- ✅ Win rate >= 70%
- ✅ Sharpe ratio > 1.5
- ✅ Max drawdown < 15%
- ✅ Bayesian learning showing improvement over time
- ✅ User confident in signals for real trading

---

## 🎯 Key Principles (Renaissance Technologies)

1. **Data-Driven**: Every decision backed by multiple data sources
2. **Quantitative**: Mathematical rigor over intuition
3. **Diversified**: Multiple signals, theories, timeframes
4. **Transparent**: Every signal has clear reasoning
5. **Adaptive**: System learns from outcomes via Bayesian updating
6. **Risk-Managed**: Monte Carlo simulation for every signal
7. **Clean Code**: Well-tested, maintainable, documented

---

## 📁 File Summary

**New Files to Create** (27 total):
```
libs/data/
├── coingecko_pro_client.py      # STEP 1.1
├── coinbase_websocket.py        # STEP 1.2
├── yahoo_finance_client.py      # STEP 1.3 (enhance existing)
└── deribit_client.py            # STEP 1.4

libs/analytics/
├── entropy.py                   # STEP 2.1
├── hurst.py                     # STEP 2.2
├── markov_chain.py              # STEP 2.3
├── kalman_filter.py             # STEP 2.4
├── bayesian.py                  # STEP 2.5
├── monte_carlo.py               # STEP 2.6
└── technical_signals.py         # STEP 3.2

libs/deepseek/
├── deepseek_client.py           # STEP 4.2
├── market_prompts.py            # STEP 4.3
└── output_parser.py             # STEP 4.4

apps/runtime/
├── v7_data_fetcher.py           # STEP 1.5
├── v7_analyzer.py               # STEP 3.4
├── v7_main.py                   # STEP 5.1
├── v7_signal_validator.py       # STEP 5.2
├── v7_signal_logger.py          # STEP 5.4
└── v7_performance.py            # STEP 8.2

tests/unit/
├── test_coingecko_pro_client.py
├── test_entropy.py
├── test_hurst.py
├── test_markov_chain.py
├── test_kalman_filter.py
└── (etc. for each new module)
```

**Files to Modify**:
```
libs/db/models.py                # Add V7 signal fields
apps/dashboard/templates/dashboard.html  # V7 signal display
apps/runtime/telegram_bot.py     # V7 signal formatting
libs/config/config.py            # Add DEEPSEEK_API_KEY
.env                             # Add DeepSeek API key
PROJECT_MEMORY.md                # Track V7 progress
```

---

**Status**: 📋 **Documentation Complete** - Ready to begin STEP 1.1

**Next Action**: Create `libs/data/coingecko_pro_client.py` and test CoinGecko Analyst API

**Questions?** Ask before starting if anything is unclear. This is your complete roadmap for V7 Ultimate.
