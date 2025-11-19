# V7 Trading System - Active Tasks

**Last Updated**: 2025-11-19 14:58
**Status**: V7 Ultimate with CoinGecko integration complete

---

## DOCUMENTATION CLEANUP COMPLETE ✅

**Achievement**: Reduced from 172 docs to 7 essential files (165 archived)

**Essential Files Kept**:
1. `README.md` - Project overview
2. `CLAUDE.md` - Instructions for Claude (MASTER FILE)
3. `PROJECT_MEMORY.md` - Session continuity
4. `MASTER_TRAINING_WORKFLOW.md` - Authoritative training guide
5. `V7_CLOUD_DEPLOYMENT.md` - V7 deployment guide
6. `V7_MATHEMATICAL_THEORIES.md` - 6 theories documentation
7. `V7_MONITORING.md` - V7 monitoring guide
8. `TODO.md` - **THIS FILE** - Master task list (single source of truth)

**Archived**: 165 obsolete files to `.archive/old_docs/`

---

## IN PROGRESS (Now)

- [x] CoinGecko integration complete
- [x] Market Context Theory (7th theory) implemented
- [x] V7 Runtime with CoinGecko data running
- [x] Documentation cleanup complete (172 → 7 files)
- [ ] **Pass CoinGecko market context to DeepSeek LLM** ← Next step

---

## NEXT UP (This Week)

### 1. DeepSeek Integration Enhancement
- [ ] Update `libs/llm/signal_generator.py` to accept `market_context` parameter
- [ ] Incorporate CoinGecko data (MCap, Vol, ATH dist, sentiment) into DeepSeek prompts
- [ ] Test that DeepSeek analysis includes market context in reasoning

### 2. Dashboard Enhancements
- [ ] Verify live price ticker showing current prices
- [ ] Verify DeepSeek analysis box displaying LLM reasoning
- [ ] Add CoinGecko metrics display (MCap, Vol, ATH distance, sentiment)

### 3. Telegram Notifications
- [ ] Implement BUY/SELL signal notifications
- [ ] Include DeepSeek reasoning in notifications
- [ ] Include CoinGecko market context

### 4. Backtesting
- [ ] Run V7 backtest on 30-90 days historical data
- [ ] Compare V7 vs V6 performance
- [ ] Analyze win rate with CoinGecko context

---

## DONE (Completed)

### 2025-11-19
- [x] **CoinGecko Integration** ✅
  - Created `libs/data/coingecko_client.py` (138 lines)
  - Created `libs/theories/market_context.py` (226 lines) - 7th theory
  - Integrated into V7 runtime
  - Successfully fetching: MCap $1770B, Vol $76B, ATH -29.7%, sentiment data
- [x] **Documentation Cleanup** ✅
  - Reduced 172 files → 7 essential files
  - Archived 165 obsolete docs to `.archive/old_docs/`
  - Created this TODO.md as master task list
- [x] **DeepSeek Budget Increase** ✅
  - Increased from $3/day to $5/day
  - Monthly budget: $150/month
- [x] **Dashboard Enhancements** ✅
  - Added live price ticker (BTC, ETH, SOL)
  - Added DeepSeek analysis display box
  - Updated JavaScript for real-time price fetching

### Previous Work
- [x] V7 Runtime with 6 mathematical theories
- [x] DeepSeek LLM integration
- [x] Bayesian learning framework
- [x] Rate limiting (30 signals/hour)
- [x] Budget controls ($5/day, $150/month)
- [x] Dashboard with live prices
- [x] FTMO rules integration

---

## BLOCKED (Waiting)

- None

---

## BACKLOG (Future)

### Performance Optimization
- [ ] Bayesian learning from actual trade outcomes
- [ ] Adaptive confidence calibration based on market conditions
- [ ] Multi-timeframe analysis (5m/15m/1h)

### Trading Features
- [ ] Paper trading mode for testing
- [ ] Position sizing based on Bayesian confidence
- [ ] Stop-loss and take-profit automation (optional)

### Infrastructure
- [ ] AWS Lambda deployment for cost reduction
- [ ] Scheduled model retraining pipeline
- [ ] Automated backtest reports

---

## CURRENT SYSTEM STATUS

### V7 Runtime
- **Status**: ✅ Running (PID in /tmp/v7_with_coingecko.log)
- **Mode**: Aggressive (30 signals/hour max)
- **Symbols**: BTC-USD, ETH-USD, SOL-USD
- **Scan Interval**: 120 seconds
- **Budget**: $5/day, $150/month

### CoinGecko Integration
- **Status**: ✅ Active
- **API Key**: CG-VQhq64e59sGxchtK8mRgdxXW
- **Cost**: $129/month (Analyst Plan)
- **Data**: Market cap, volume, ATH distance, sentiment
- **Theory**: 7th theory - Market Context Analysis
- **Next**: Pass data to DeepSeek LLM prompts

### Mathematical Theories (All Working ✅)
1. Shannon Entropy - Market predictability (0.947 = high randomness)
2. Hurst Exponent - Trend persistence (0.528 = mean-reverting)
3. Kolmogorov Complexity - Pattern complexity
4. Market Regime - Bull/bear/sideways classification
5. Risk Metrics - VaR, CVaR, Sharpe ratio
6. Fractal Dimension - Market structure analysis
7. **Market Context** - CoinGecko macro analysis ⭐ NEW

### Dashboard
- **URL**: http://localhost:5000
- **Status**: Running
- **Features**: Live prices, DeepSeek analysis, V7 signals

### Database
- **Path**: /root/crpbot/tradingai.db
- **Type**: SQLite
- **Tables**: signals, candles, performance

---

## WEEKLY CLEANUP SCHEDULE

**Every Friday**:
- [ ] Review and archive temporary docs
- [ ] Update TODO.md with completed tasks
- [ ] Verify only 7-10 essential docs remain
- [ ] Commit archive changes to git

---

## SUCCESS CRITERIA

### CoinGecko Integration ✅
- [x] CoinGecko client created
- [x] Market context theory implemented
- [x] V7 runtime using CoinGecko data
- [ ] Dashboard showing CoinGecko metrics (optional)
- [ ] DeepSeek analysis includes market cap/volume (pending)

### Documentation Cleanup ✅
- [x] 172 files → 7 essential files
- [x] TODO.md created as master task list
- [x] .archive/ folder with old docs
- [ ] CLAUDE.md updated with CoinGecko integration (pending)
- [x] No more "which doc is correct?" confusion

### Process Fix ✅
- [x] Weekly doc cleanup scheduled
- [x] TODO.md updated after every task
- [x] No new docs without archiving old ones
- [x] Single source of truth maintained

---

## IMPORTANT NOTES

### Why Documentation Cleanup Was Critical
- **Problem**: 172 files caused important steps (like CoinGecko) to be missed
- **Root Cause**: Every session created new docs without archiving old ones
- **Solution**: Reduced to 7 essential files + regular weekly cleanup
- **Result**: Clear single source of truth, no more missed steps

### Current Priority
**Pass CoinGecko market context to DeepSeek LLM**:
- CoinGecko data is being fetched and logged
- Not yet incorporated into DeepSeek prompts
- Need to update `libs/llm/signal_generator.py` to accept `market_context` parameter
- This will give DeepSeek macro market intelligence for better signal quality

### Budget Tracking
- **DeepSeek**: $5/day ($150/month) - ~$0.0004 per signal
- **CoinGecko**: $129/month (fixed, paid)
- **AWS (training)**: ~$5-8/month (spot instances)
- **Total**: ~$285/month

---

**This is the SINGLE SOURCE OF TRUTH for all active tasks. Update this file after every completed task.**
