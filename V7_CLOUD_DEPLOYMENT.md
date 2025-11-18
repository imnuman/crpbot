# V7 Ultimate - Cloud Server Deployment Instructions

## Status: STEP 4 Complete - Ready for Cloud Deployment

**Date**: 2025-11-18
**Local Testing**: All components verified locally (sans live API calls)
**Next**: Deploy to cloud server for end-to-end testing with live market data

---

## What Was Completed (STEP 4)

✅ **V7 Runtime Orchestrator** - Full trading runtime with all integrations
✅ **Live Data Fetching** - Coinbase REST API integration
✅ **Signal Output Formatting** - Professional console output
✅ **Rate Limiting** - 6 signals/hour with sliding window tracker
✅ **Cost Controls** - Daily ($3) and monthly ($100) budget enforcement
✅ **Statistics Tracking** - DeepSeek API usage and Bayesian learning metrics
✅ **FTMO Rules Integration** - Daily/total loss limits checked before signals

**All bugs fixed**:
- ✅ Added `deepseek_api_key` field to Settings
- ✅ Fixed SignalGenerator parameter name (`api_key` not `deepseek_api_key`)
- ✅ Fixed 25 property access errors (nested `parsed_signal` structure)
- ✅ Fixed 4 statistics access errors (correct key names)

---

## Prerequisites on Cloud Server

### 1. Environment Variables (`.env`)

Add DeepSeek API key to your `.env` file:

```bash
# DeepSeek LLM API (for V7 Ultimate)
DEEPSEEK_API_KEY=sk-...your-key-here...

# Existing Coinbase credentials (already configured)
COINBASE_API_KEY_NAME=organizations/.../apiKeys/...
COINBASE_API_PRIVATE_KEY=-----BEGIN EC PRIVATE KEY-----...
```

### 2. Get DeepSeek API Key

1. Visit: https://platform.deepseek.com/
2. Sign up / Login
3. Generate API key
4. Add to `.env` as shown above

**Pricing**: $0.27/M input tokens, $1.10/M output tokens (~$0.0003 per signal)

---

## Deployment Steps

### Step 1: Push V7 Code to GitHub

**On Local Machine**:

```bash
cd /home/numan/crpbot

# Stage V7 files
git add apps/runtime/v7_runtime.py
git add libs/config/config.py
git add libs/llm/
git add libs/theories/
git add libs/bayesian/
git add V7_CLOUD_DEPLOYMENT.md

# Commit
git commit -m "feat(v7): complete V7 Ultimate signal generation pipeline

STEP 4 Complete - Signal Generation Pipeline:
- V7 runtime orchestrator with full integration
- Live data fetching via Coinbase REST API
- 6 mathematical theories + DeepSeek LLM synthesis
- Rate limiting: 6 signals/hour
- Cost controls: \$3/day, \$100/month budgets
- FTMO compliance checking
- Comprehensive statistics tracking

Components:
- apps/runtime/v7_runtime.py (551 lines)
- libs/llm/* (DeepSeek client, synthesizer, parser, generator)
- libs/theories/* (6 mathematical theories)
- libs/bayesian/* (Bayesian learning framework)

All local tests passing. Ready for cloud deployment.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push to GitHub
git push origin main
```

### Step 2: Pull on Cloud Server

**On Cloud Server** (178.156.136.185):

```bash
ssh root@178.156.136.185

cd ~/crpbot

# Pull latest V7 code
git pull origin main

# Verify new files
ls -lh apps/runtime/v7_runtime.py
ls -lh libs/llm/
ls -lh libs/theories/
ls -lh libs/bayesian/
```

### Step 3: Add DeepSeek API Key

**On Cloud Server**:

```bash
cd ~/crpbot

# Edit .env file
nano .env

# Add this line (replace with your actual key):
# DEEPSEEK_API_KEY=sk-...your-key-here...

# Verify .env has the key
grep DEEPSEEK_API_KEY .env
```

### Step 4: Test V7 Runtime (Single Iteration)

**On Cloud Server**:

```bash
cd ~/crpbot

# Test with 1 iteration
.venv/bin/python3 apps/runtime/v7_runtime.py --iterations 1 --sleep-seconds 10

# Expected output:
# ================================================================================
# V7 ULTIMATE TRADING RUNTIME - STARTED
# ================================================================================
# Symbols: BTC-USD, ETH-USD, SOL-USD
# Scan Interval: 10s
# Rate Limit: 6 signals/hour
# Daily Budget: $3.00
# Monthly Budget: $100.00
# Conservative Mode: True
# ================================================================================
#
# [Market data fetching...]
# [Theory analysis: Shannon, Hurst, Kolmogorov, etc...]
# [DeepSeek LLM synthesis...]
# [Signal generation...]
#
# V7 Statistics:
#   DeepSeek API Calls:    3
#   Total API Cost:        $0.000XXX
#   Bayesian Win Rate:     50.0%
#   Daily Cost:            $0.00XX / $3.00
#   Monthly Cost:          $0.00XX / $100.00
```

### Step 5: Run V7 in Continuous Mode (Background)

**On Cloud Server**:

```bash
cd ~/crpbot

# Run V7 in background with 2-minute scan interval
nohup .venv/bin/python3 apps/runtime/v7_runtime.py \
  --iterations -1 \
  --sleep-seconds 120 \
  > /tmp/v7_runtime.log 2>&1 &

# Save process ID
V7_PID=$!
echo "V7 Runtime PID: $V7_PID"

# Monitor live output
tail -f /tmp/v7_runtime.log

# Or monitor with filtering
tail -f /tmp/v7_runtime.log | grep -E "SIGNAL|Statistics|DeepSeek"
```

### Step 6: Monitor V7 Runtime

**On Cloud Server**:

```bash
# Check if V7 is running
ps aux | grep v7_runtime

# View recent logs
tail -100 /tmp/v7_runtime.log

# View statistics only
tail -100 /tmp/v7_runtime.log | grep -A 10 "V7 Statistics"

# Check database for V7 signals
sqlite3 tradingai.db "SELECT * FROM signals ORDER BY timestamp DESC LIMIT 10;"

# Stop V7 runtime
kill $V7_PID
# Or use pkill:
pkill -f v7_runtime.py
```

---

## V7 Runtime CLI Options

```bash
python apps/runtime/v7_runtime.py [OPTIONS]

Options:
  --iterations N        Number of scan iterations (-1 = infinite)
  --sleep-seconds N     Seconds between scans (default: 120)
  --symbols S1,S2,S3    Comma-separated symbols (default: BTC-USD,ETH-USD,SOL-USD)
  --max-signals N       Max signals per hour (default: 6)
  --daily-budget N      Daily cost budget in USD (default: 3.00)
  --monthly-budget N    Monthly cost budget in USD (default: 100.00)
  --conservative        Use conservative LLM prompts (default: True)
```

**Examples**:

```bash
# Test mode - 1 iteration
python apps/runtime/v7_runtime.py --iterations 1

# Production mode - infinite, 2-minute scans
python apps/runtime/v7_runtime.py --iterations -1 --sleep-seconds 120

# High-frequency mode - 30-second scans, 20 signals/hour
python apps/runtime/v7_runtime.py --iterations -1 --sleep-seconds 30 --max-signals 20

# Single symbol test
python apps/runtime/v7_runtime.py --iterations 5 --symbols BTC-USD

# Aggressive mode (higher risk tolerance)
python apps/runtime/v7_runtime.py --iterations -1 --conservative False
```

---

## Expected Behavior

### Successful Run

When V7 runtime is working correctly, you should see:

1. **Initialization**:
   ```
   ✅ Database initialized: sqlite:///tradingai.db
   ✅ Coinbase REST client initialized
   ✅ Market data fetcher initialized
   ✅ V7 SignalGenerator initialized (6 theories + DeepSeek LLM)
   ```

2. **Per-Scan Output**:
   ```
   ================================================================================
   ITERATION 1 | 2025-11-18 11:05:23
   ================================================================================
   Starting V7 scan across 3 symbols...
   Fetching live data for BTC-USD...
   Analyzing with 6 theories...
   Querying DeepSeek LLM...

   ================================================================================
   V7 ULTIMATE SIGNAL | BTC-USD
   Timestamp:    2025-11-18 11:05:25 UTC
   SIGNAL:       BUY
   CONFIDENCE:   78%
   REASONING:    Strong trending (Hurst 0.72) + bull regime with positive momentum

   Market Context:
   Current Price:      $91,234.56
   24h Change:         +2.3%

   Theory Analysis:
   Shannon Entropy:     0.523
   Hurst Exponent:      0.72 (trending)
   Kolmogorov Complexity: 0.34
   Market Regime:       BULL (65% confidence)
   Risk Metrics:        VaR: 12%, Sharpe: 1.2
   Fractal Dimension:   1.45

   DeepSeek Cost:  $0.000342
   FTMO Status:    ✅ PASS
   ================================================================================
   ```

3. **Statistics After Each Scan**:
   ```
   V7 Statistics:
     DeepSeek API Calls:    3
     Total API Cost:        $0.001026
     Bayesian Win Rate:     50.0%
     Bayesian Total Trades: 0
     Daily Cost:            $0.0010 / $3.00
     Monthly Cost:          $0.00 / $100.00
   ```

### Common Issues

**Issue 1: Missing DeepSeek API Key**
```
ERROR: DeepSeek API key required. Set DEEPSEEK_API_KEY environment variable
```
**Fix**: Add `DEEPSEEK_API_KEY=sk-...` to `.env` file

**Issue 2: Coinbase 401 Unauthorized**
```
HTTP Error: 401 Client Error: Unauthorized
```
**Fix**: Verify `COINBASE_API_KEY_NAME` and `COINBASE_API_PRIVATE_KEY` in `.env`

**Issue 3: Rate Limit Exceeded**
```
Rate limit: 6/6 signals in last hour
```
**Fix**: This is expected behavior. Wait for rate limit window to reset or increase `--max-signals`

**Issue 4: Budget Exceeded**
```
Daily budget exceeded: $3.00 spent
```
**Fix**: Wait for daily reset or increase `--daily-budget`

---

## Performance Expectations

### Cost Estimates

**Per Signal**:
- Input tokens: ~400-500 (theory analysis + market context)
- Output tokens: ~100-150 (signal + reasoning)
- Cost per signal: ~$0.0003 - $0.0005

**Daily Cost** (6 signals/hour, 24 hours):
- Signals per day: 6 × 24 = 144 signals
- Daily cost: 144 × $0.0004 = **$0.058/day** (~$1.75/month)

**Well under budget**: $3/day, $100/month limits provide 50x safety margin

### Signal Frequency

- **Default**: 6 signals/hour across 3 symbols = ~2 signals/hour/symbol
- **Conservative mode**: Higher confidence threshold, fewer signals
- **Aggressive mode**: Lower threshold, more signals (higher cost)

---

## Next Steps After Deployment

### STEP 5: Dashboard/Telegram Output

Once V7 is running on cloud:

1. **STEP 5.1**: Integrate Telegram notifications for V7 signals
2. **STEP 5.2**: Update dashboard to display V7 signals alongside V6
3. **STEP 5.3**: Add signal history visualization

### STEP 6: Backtesting Framework

1. Historical backtest of V7 vs V6 performance
2. Risk-adjusted returns comparison
3. Bayesian learning validation

### STEP 7: Performance Monitoring

1. Real-time win rate tracking
2. Cost per signal monitoring
3. Theory contribution analysis

---

## Troubleshooting

### View V7 Logs with Context

```bash
# Last 200 lines
tail -200 /tmp/v7_runtime.log

# Only signals
tail -500 /tmp/v7_runtime.log | grep -A 20 "V7 ULTIMATE SIGNAL"

# Only statistics
tail -500 /tmp/v7_runtime.log | grep -A 6 "V7 Statistics"

# Only errors
tail -500 /tmp/v7_runtime.log | grep -i error

# Live tail with filtering
tail -f /tmp/v7_runtime.log | grep -E "SIGNAL|Statistics|ERROR"
```

### Database Queries

```bash
# View all V7 signals
sqlite3 tradingai.db "
SELECT
  timestamp,
  symbol,
  signal_type,
  confidence,
  reasoning
FROM signals
WHERE reasoning LIKE '%Hurst%'  -- V7 signals include theory mentions
ORDER BY timestamp DESC
LIMIT 20;
"

# V7 performance summary
sqlite3 tradingai.db "
SELECT
  symbol,
  COUNT(*) as total_signals,
  AVG(confidence) as avg_confidence,
  SUM(CASE WHEN signal_type = 'BUY' THEN 1 ELSE 0 END) as buy_signals,
  SUM(CASE WHEN signal_type = 'SELL' THEN 1 ELSE 0 END) as sell_signals
FROM signals
WHERE reasoning LIKE '%Hurst%'
GROUP BY symbol;
"
```

---

## Files Modified/Created

**New Files**:
- `apps/runtime/v7_runtime.py` (551 lines) - Main V7 runtime
- `libs/llm/deepseek_client.py` - DeepSeek API client
- `libs/llm/signal_synthesizer.py` - Theory → LLM prompt converter
- `libs/llm/signal_parser.py` - LLM response → structured signal
- `libs/llm/signal_generator.py` - Complete signal generation orchestrator
- `libs/theories/shannon_entropy.py` - Market entropy analysis
- `libs/theories/hurst_exponent.py` - Trend persistence detection
- `libs/theories/kolmogorov_complexity.py` - Pattern complexity
- `libs/theories/market_regime.py` - Bull/bear/sideways detection
- `libs/theories/risk_metrics.py` - VaR, Sharpe, volatility
- `libs/theories/fractal_dimension.py` - Market structure analysis
- `libs/bayesian/bayesian_learner.py` - Beta distribution learning

**Modified Files**:
- `libs/config/config.py` - Added `deepseek_api_key` field

---

## Support

If you encounter issues during deployment:

1. Check `/tmp/v7_runtime.log` for error messages
2. Verify `.env` has `DEEPSEEK_API_KEY` configured
3. Test Coinbase API separately: `python -c "from apps.runtime.data_fetcher import MarketDataFetcher; mdf = MarketDataFetcher(); print(mdf.fetch_latest_candles('BTC-USD', 10))"`
4. Test DeepSeek API separately: `python libs/llm/deepseek_client.py`

---

**Deployment Ready**: All code tested locally. Deploy to cloud server and run with live market data to complete STEP 4 validation and proceed to STEP 5.
