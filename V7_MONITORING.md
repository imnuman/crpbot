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

## 🚀 Next Steps (STEP 5)

V7 STEP 4 is complete! Next enhancements:
1. **Dashboard V7 Integration**: Add V7 signal visualization
2. **Advanced Telegram Commands**: Stop/start V7, adjust parameters
3. **Performance Tracking**: Win rate, PnL tracking for executed trades
4. **Bayesian Learning**: Continuous improvement from trade outcomes

---

**Current Status**: ✅ V7 Running in Production (Background Process ID: 99fd09)

**Last Updated**: 2025-11-18
