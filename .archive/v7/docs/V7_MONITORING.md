# V7 Ultimate - Monitoring Guide

V7 Ultimate is now running in production! This guide shows you how to monitor and manage the V7 runtime.

## 📊 Quick Status Check

```bash
# Check if V7 is running
ps aux | grep "v7_runtime.py" | grep -v grep
# Expected output: root 1868284 ... .venv/bin/python3 apps/runtime/v7_runtime.py --iterations -1 --sleep-seconds 120

# View latest logs
tail -f /tmp/v7_runtime.log

# Check database for recent V7 signals
sqlite3 tradingai.db "SELECT timestamp, symbol, direction, confidence, entry_price, sl_price, tp_price FROM signals WHERE model_version='v7_ultimate' AND timestamp > datetime('now', '-1 hour') ORDER BY timestamp DESC LIMIT 10"

# Count signals by type (24 hours)
sqlite3 tradingai.db "SELECT direction, COUNT(*) as count FROM signals WHERE model_version='v7_ultimate' AND timestamp > datetime('now', '-24 hours') GROUP BY direction"
```

## 📈 Dashboard Access

The V7 dashboard is accessible at:
- **Local**: http://localhost:5000
- **External**: http://178.156.136.185:5000 (if firewall allows)
- **Features**:
  - V7 Ultimate Signals with price predictions (Entry, Stop Loss, Take Profit)
  - Live statistics (BUY/SELL/HOLD counts, confidence, API costs)
  - Signal breakdown by direction, symbol, and confidence tier
  - Clear explanations of how V7 works (6 mathematical theories)

To check dashboard status:
```bash
ps aux | grep "app.py" | grep python | grep -v grep
# Expected: root ... .venv/bin/python3 -m apps.dashboard.app

lsof -i :5000
curl http://localhost:5000/api/v7/signals/recent/1
```

**Restart Dashboard:**
```bash
pkill -9 -f "app.py"
.venv/bin/python3 -m apps.dashboard.app > /tmp/dashboard.log 2>&1 &
```

## 🔔 Telegram Notifications & Commands

V7 sends real-time signals to Telegram with:
- Signal type (BUY/SELL/HOLD)
- Confidence level
- Mathematical theory analysis (Shannon Entropy, Hurst Exponent, etc.)
- LLM reasoning
- Risk metrics (VaR, Sharpe Ratio, Profit Probability)

Chat ID: `8302332448`

### Telegram Bot Commands (STEP 5)

The V7 Telegram bot command listener is now available! Control V7 directly from Telegram:

**V7 Control Commands:**
- `/v7_status` - Show V7 runtime status (running/stopped, latest signal, 24h stats)
- `/v7_start` - Start V7 runtime in background
- `/v7_stop` - Stop V7 runtime
- `/v7_stats` - Detailed V7 statistics (24h/7d signals, costs, projections)
- `/v7_config` - Show/adjust V7 parameters (rate limit, confidence threshold)

**Legacy Commands:**
- `/check` - System status
- `/stats` - Performance metrics
- `/ftmo_status` - FTMO account status
- `/help` - Show all commands

**Start Telegram Bot Listener:**
```bash
# Run in foreground (testing)
.venv/bin/python3 apps/runtime/v7_telegram_bot_runner.py

# Run in background (production)
nohup .venv/bin/python3 apps/runtime/v7_telegram_bot_runner.py > /tmp/v7_telegram_bot.log 2>&1 &

# Check if running
ps aux | grep v7_telegram_bot_runner | grep -v grep

# View logs
tail -f /tmp/v7_telegram_bot.log
```

## 💰 Cost Tracking

```bash
# View cost statistics
# Check the BashOutput of background process 99fd09
# Or query database:
sqlite3 tradingai.db "SELECT
    DATE(timestamp) as date,
    COUNT(*) as signals,
    SUM(CASE WHEN direction='long' THEN 1 ELSE 0 END) as buys,
    SUM(CASE WHEN direction='short' THEN 1 ELSE 0 END) as sells,
    SUM(CASE WHEN direction='neutral' THEN 1 ELSE 0 END) as holds,
    AVG(confidence) as avg_conf
FROM signals
WHERE model_version='v7_ultimate' AND timestamp > datetime('now', '-7 days')
GROUP BY DATE(timestamp)
ORDER BY date DESC"
```

**Expected Costs**:
- ~$0.0003 per signal
- ~6 signals/hour (rate limited)
- ~$0.04/day (~$1.20/month at current rate)
- Daily budget: $3.00
- Monthly budget: $100.00

## 🛠️ Management Commands

### Start V7 (if not running)

```bash
# Start in background (continuous mode)
.venv/bin/python3 apps/runtime/v7_runtime.py --iterations -1 --sleep-seconds 300 &

# Check output via BashOutput tool or redirect to log file:
nohup .venv/bin/python3 apps/runtime/v7_runtime.py --iterations -1 --sleep-seconds 300 > /tmp/v7_runtime.log 2>&1 &

# Monitor log file
tail -f /tmp/v7_runtime.log
```

### Stop V7

```bash
# Find V7 process
ps aux | grep "v7_runtime.py" | grep -v grep

# Kill gracefully
pkill -f "v7_runtime.py"

# Force kill if needed
pkill -9 -f "v7_runtime.py"
```

### Restart V7

```bash
# Stop and start
pkill -f "v7_runtime.py"
sleep 2
.venv/bin/python3 apps/runtime/v7_runtime.py --iterations -1 --sleep-seconds 300 &
```

## 📊 Performance Metrics

### Signal Quality Metrics

```bash
# Average confidence by signal type
sqlite3 tradingai.db "SELECT
    signal,
    AVG(confidence) as avg_conf,
    MIN(confidence) as min_conf,
    MAX(confidence) as max_conf,
    COUNT(*) as count
FROM signals
WHERE timestamp > datetime('now', '-24 hours')
GROUP BY signal"

# Signals per symbol
sqlite3 tradingai.db "SELECT
    symbol,
    COUNT(*) as total,
    SUM(CASE WHEN valid=1 THEN 1 ELSE 0 END) as valid,
    SUM(CASE WHEN valid=0 THEN 1 ELSE 0 END) as invalid
FROM signals
WHERE timestamp > datetime('now', '-24 hours')
GROUP BY symbol"
```

### Rate Limiting Stats

V7 enforces 6 signals/hour (60 seconds minimum between signals) to:
- Prevent spam
- Control costs
- Ensure quality analysis

Check rate limiting in action:
```bash
# View signal timestamps to verify spacing
sqlite3 tradingai.db "SELECT timestamp, symbol, signal FROM signals ORDER BY timestamp DESC LIMIT 20"
```

## 🔍 Troubleshooting

### V7 Not Generating Signals

1. **Check if running**: `ps aux | grep v7_runtime`
2. **Check DeepSeek API key**: `grep DEEPSEEK_API_KEY .env`
3. **Check database connection**: `sqlite3 tradingai.db "SELECT COUNT(*) FROM signals"`
4. **Check Coinbase API**: Test data fetching separately

### High HOLD Rate (>90%)

This is **expected behavior** during ranging/choppy markets. V7 is conservative by design.
- Shannon Entropy > 0.7 = unpredictable market → HOLD
- VaR > 5% = high risk → HOLD
- Low profit probability < 20% → HOLD

### Telegram Not Sending

1. **Check Telegram token**: `grep TELEGRAM_TOKEN .env`
2. **Check chat ID**: `grep TELEGRAM_CHAT_ID .env`
3. **Test manually**: `.venv/bin/python3 test_telegram_v7.py`

### Database Errors

```bash
# Check database integrity
sqlite3 tradingai.db "PRAGMA integrity_check"

# Check table schema
sqlite3 tradingai.db ".schema signals"

# Rebuild indexes if needed
sqlite3 tradingai.db "REINDEX"
```

## 📁 Important Files

- **Runtime**: `apps/runtime/v7_runtime.py`
- **Config**: `.env`
- **Database**: `tradingai.db`
- **Dashboard**: `apps/dashboard/app.py`
- **Logs**: Check background process output via BashOutput tool

## 📊 Performance Monitoring & Cost Tracking (STEP 7)

### Dashboard Performance Section

**NEW**: V7 dashboard now includes comprehensive performance monitoring!

Access at: http://178.156.136.185:5000 (scroll to "Performance Monitoring & Cost Tracking" section)

**Features**:
1. **Cost Tracking Dashboard** (6 real-time metrics)
2. **Win/Loss Performance Tracking** (manual trading)
3. **Budget Alerts** (visual green/yellow/red indicators)
4. **Auto-refresh** (every 5 seconds)

### API Endpoints for Monitoring

**1. Cost Tracking API**:
```bash
# Get V7 cost breakdown
curl http://localhost:5000/api/v7/costs | python3 -m json.tool

# Response includes:
# - today.cost, today.remaining (vs $3/day budget)
# - month.cost, month.remaining (vs $100/month budget)
# - avg_cost_per_signal
# - by_symbol breakdown
# - daily/monthly trends
```

**2. Performance API**:
```bash
# Get win/loss statistics
curl http://localhost:5000/api/v7/performance | python3 -m json.tool

# Response includes:
# - total_trades, wins, losses, win_rate
# - total_pnl, avg_pnl_per_trade
# - breakdowns by symbol/tier/direction
```

**3. Theory Contribution API**:
```bash
# Analyze which theories correlate with winning trades
curl http://localhost:5000/api/v7/theories/contribution | python3 -m json.tool

# Shows which theories (Shannon, Hurst, etc.) predict wins
```

### Manual Trade Tracking

V7 is a **manual trading system**. To track performance:

**Log a trade result**:
```bash
# Method 1: Using curl
curl -X POST http://localhost:5000/api/v7/signals/774/result \
  -H "Content-Type: application/json" \
  -d '{
    "result": "win",
    "exit_price": 95500.0,
    "pnl": 250.00,
    "notes": "Exited at resistance level"
  }'

# Method 2: Using Python
python3 -c "
import requests
requests.post(
    'http://localhost:5000/api/v7/signals/774/result',
    json={'result': 'win', 'exit_price': 95500.0, 'pnl': 250.00}
)"
```

**Find signal ID**:
```bash
# Get recent signals
curl -s http://localhost:5000/api/v7/signals/recent/24 | python3 -m json.tool | grep -A 5 '"id"'

# Or from dashboard signals table
```

**Result options**: `"win"`, `"loss"`, `"pending"`, `"skipped"`

### Budget Alerts

The dashboard visually indicates budget usage:
- 🟢 Green border: < 80% used (healthy)
- 🟡 Yellow border: 80-95% used (warning)
- 🔴 Red border: ≥ 95% used (critical)

**Daily Budget**: $3.00 (currently at $0.019 = 0.64% ✅)
**Monthly Budget**: $100.00 (currently at $0.031 = 0.03% ✅)

### Cost Query Examples

```bash
# Total V7 cost all-time
sqlite3 tradingai.db "SELECT SUM(
    CASE
        WHEN notes LIKE '%llm_cost_usd%'
        THEN CAST(json_extract(notes, '$.llm_cost_usd') AS REAL)
        ELSE 0
    END
) as total_cost
FROM signals WHERE model_version='v7_ultimate'"

# Cost by day
sqlite3 tradingai.db "SELECT
    DATE(timestamp) as date,
    COUNT(*) as signals,
    SUM(CASE
        WHEN notes LIKE '%llm_cost_usd%'
        THEN CAST(json_extract(notes, '$.llm_cost_usd') AS REAL)
        ELSE 0
    END) as daily_cost
FROM signals
WHERE model_version='v7_ultimate'
GROUP BY DATE(timestamp)
ORDER BY date DESC
LIMIT 7"
```

### Performance Query Examples

```bash
# Win rate (when trades are tracked)
sqlite3 tradingai.db "SELECT
    COUNT(*) as total_trades,
    SUM(CASE WHEN result='win' THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN result='loss' THEN 1 ELSE 0 END) as losses,
    CAST(SUM(CASE WHEN result='win' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100 as win_rate_pct,
    SUM(pnl) as total_pnl
FROM signals
WHERE model_version='v7_ultimate'
AND result IN ('win', 'loss')"

# Performance by symbol
sqlite3 tradingai.db "SELECT
    symbol,
    COUNT(*) as trades,
    SUM(CASE WHEN result='win' THEN 1 ELSE 0 END) as wins,
    CAST(SUM(CASE WHEN result='win' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100 as win_rate,
    SUM(pnl) as total_pnl
FROM signals
WHERE model_version='v7_ultimate'
AND result IN ('win', 'loss')
GROUP BY symbol"
```

## 🚀 Next Steps (STEP 8)

### ✅ Completed Steps:
- ✅ **STEP 5**: Dashboard & Telegram Integration
- ✅ **STEP 6**: Backtesting Framework
- ✅ **STEP 7**: Performance Monitoring & Cost Tracking ⭐ **NEW**

### 🚧 Remaining:
**STEP 8: Documentation** (estimated: 1-2 hours)
1. Complete API documentation for all V7 endpoints
2. User guide for manual trading workflow
3. Theory module detailed documentation
4. Deployment/maintenance guide

---

**Current Status**: ✅ V7 Running in Production with Full Performance Monitoring

**Live Processes**:
- V7 Runtime: PID 1911821
- Dashboard: PID 1995084 (with performance monitoring)
- Telegram Bot: PID (check with `ps aux | grep v7_telegram`)

**Dashboard URL**: http://178.156.136.185:5000

---

## 🔧 Signal Generation Fixes (2025-11-19 17:55 EST)

### Problem: Missed +0.88% BTC Move (3-5 PM)

**What Happened**:
- BTC moved from $88,769 → $89,554 (+$785, +0.88%) from 3:01-3:58 PM
- System only output HOLD signals at 15-40% confidence
- DeepSeek reasoning: "High entropy (0.92) shows random conditions conflicting with consolidation regime"

**Root Cause**:
- Conservative mode prompt made DeepSeek default to HOLD when theories conflicted
- Equal theory weighting → conservative theories (entropy, Sharpe, Monte Carlo) dominated
- No fail-safe override for strong momentum in choppy markets

### Fixes Implemented ✅

#### Fix 1: Enhanced DeepSeek Prompt
**File**: `/root/crpbot/libs/llm/signal_synthesizer.py` (lines 186-208)

Added explicit rules:
1. Prioritize momentum signals (Kalman, Hurst) in choppy markets (entropy >0.85)
2. Strong momentum (>±15) with trending Hurst (>0.55) = ACTIONABLE SIGNAL
3. Confidence calibration: 35-45% acceptable in ranging markets
4. Don't let negative Sharpe ratios paralyze decision-making

#### Fix 2: Momentum Override Logic
**File**: `/root/crpbot/apps/runtime/v7_runtime.py` (lines 456-502)

Automatic override when:
- Bullish: momentum > +20, hurst > 0.55, entropy > 0.85 → BUY @ 40%
- Bearish: momentum < -20, hurst < 0.45, entropy > 0.85 → SELL @ 40%

### Monitoring Results (17:51-17:56 EST)

**Runtime Configuration**:
- Mode: Aggressive (conservative_mode=False)
- Rate Limit: 30 signals/hour
- Scan Interval: 120 seconds

**Signal Quality**:
| Time     | Symbol   | Signal | Confidence | Price      | Notes                        |
|----------|----------|--------|------------|------------|------------------------------|
| 17:51:47 | BTC-USD  | HOLD   | 45.0%      | $90,355.42 | ✅ Improved from 15-40%      |
| 17:53:54 | BTC-USD  | HOLD   | 40.0%      | $90,368.72 | Stable confidence            |
| 17:55:56 | BTC-USD  | HOLD   | 40.0%      | $90,391.21 | +$36 move (0.04% - minimal)  |

**Key Observations**:
- ✅ Confidence improved: 40-45% (up from 15-40% baseline)
- ✅ HOLD signals appropriate (market only +0.04%, not volatile)
- ⏳ No momentum override triggered yet (momentum below ±20 threshold)
- 🔍 Awaiting volatile market to validate full effectiveness

**Next Validation**: Need market scenario with >±0.5% move + entropy >0.85 + momentum >±20

**Documentation**: Full analysis in `V7_SIGNAL_FIXES.md`

---

**Last Updated**: 2025-11-19 17:56 EST
