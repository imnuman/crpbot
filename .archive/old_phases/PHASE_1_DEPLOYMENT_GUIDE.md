# Phase 1 Deployment Guide - V7 Ultimate Enhanced

**Date**: 2025-11-24
**Version**: Phase 1 Integration Complete
**Status**: Ready for A/B Testing

---

## 📋 Quick Start

### Deploy Phase 1 (A/B Test Mode)

```bash
# On cloud server (Builder Claude)
cd /root/crpbot

# Stop current V7 if needed (optional - can run both)
# pkill -f v7_runtime.py

# Start Phase 1 Enhanced Runtime
nohup .venv/bin/python3 apps/runtime/v7_runtime_phase1.py \
  --iterations -1 \
  --sleep-seconds 300 \
  --max-signals-per-hour 3 \
  --variant "v7_phase1" \
  > /tmp/v7_phase1_$(date +%Y%m%d_%H%M).log 2>&1 &

# Get PID
ps aux | grep v7_runtime_phase1 | grep -v grep

# Monitor logs
tail -f /tmp/v7_phase1_*.log
```

---

## 🎯 Deployment Strategy

### Option 1: A/B Testing (Recommended)
Run both V7 current and V7 Phase 1 simultaneously for comparison.

**Advantages**:
- Direct performance comparison
- Can revert quickly if issues
- Statistically significant results

**Setup**:
```bash
# Current V7 (already running as PID 2620770)
# Keep running as-is

# Phase 1 (new)
nohup .venv/bin/python3 apps/runtime/v7_runtime_phase1.py \
  --iterations -1 \
  --sleep-seconds 300 \
  --max-signals-per-hour 3 \
  --variant "v7_phase1" \
  > /tmp/v7_phase1_$(date +%Y%m%d_%H%M).log 2>&1 &
```

**Duration**: 7 days (target: 30+ trades per variant)

### Option 2: Replace Current V7
Stop current V7, deploy Phase 1 exclusively.

**Advantages**:
- Simpler to manage
- More signals for Phase 1

**Disadvantages**:
- No baseline comparison
- Can't easily revert

**Setup**:
```bash
# Stop current V7
pkill -f v7_runtime.py

# Start Phase 1
nohup .venv/bin/python3 apps/runtime/v7_runtime_phase1.py \
  --iterations -1 \
  --sleep-seconds 300 \
  --max-signals-per-hour 6 \
  --variant "v7_phase1_exclusive" \
  > /tmp/v7_phase1_$(date +%Y%m%d_%H%M).log 2>&1 &
```

---

## 🔧 Configuration Options

### Command-Line Arguments

```bash
python3 apps/runtime/v7_runtime_phase1.py \
  --iterations N            # Number of iterations (-1 = infinite, default: -1)
  --sleep-seconds N         # Seconds between iterations (default: 300 = 5 min)
  --max-signals-per-hour N  # Max signals per hour (default: 3)
  --variant NAME            # Strategy variant name for tracking (default: v7_phase1)
```

### Environment Variables (.env)

All V7 environment variables apply, plus Phase 1 uses:
- `DB_URL` - Database connection (for Kelly calculation, paper trading)
- `DEEPSEEK_API_KEY` - LLM API key
- `COINBASE_API_KEY_NAME` - Market data
- `COINBASE_API_PRIVATE_KEY` - Market data

### Code Configuration (V7Phase1Config)

Edit `apps/runtime/v7_runtime_phase1.py` to adjust:

```python
phase1_config = V7Phase1Config(
    # Base V7 config
    symbols=["BTC-USD", "ETH-USD", "SOL-USD", ...],
    max_signals_per_hour=3,
    enable_paper_trading=True,

    # Phase 1 specific
    fractional_kelly=0.5,              # 50% Kelly (conservative)
    trailing_stop_activation=0.005,    # 0.5% profit to activate
    trailing_stop_distance=0.002,      # 0.2% trailing distance
    max_hold_hours=24,                 # 24h max hold
    breakeven_profit_threshold=0.0025, # 0.25% to move SL to breakeven
    correlation_threshold=0.7,         # Block if corr > 0.7
    enable_regime_filtering=True       # Enable regime-based filtering
)
```

---

## 📊 Monitoring

### Check Status

```bash
# Check if running
ps aux | grep v7_runtime_phase1 | grep -v grep

# Check logs
tail -100 /tmp/v7_phase1_*.log

# Check recent signals
sqlite3 tradingai.db "
SELECT timestamp, symbol, direction, confidence, strategy
FROM signals
WHERE strategy = 'v7_phase1'
ORDER BY timestamp DESC
LIMIT 10;"
```

### Monitor Performance

```bash
# Phase 1 paper trading results
sqlite3 tradingai.db "
SELECT
    COUNT(*) as total,
    SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
    ROUND(100.0 * SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) / COUNT(*), 2) as win_rate,
    ROUND(AVG(pnl_percent), 2) as avg_pnl
FROM signal_results sr
JOIN signals s ON sr.signal_id = s.id
WHERE s.strategy = 'v7_phase1'
AND sr.outcome IN ('win', 'loss');"

# Compare with current V7
sqlite3 tradingai.db "
SELECT
    s.strategy,
    COUNT(*) as total,
    SUM(CASE WHEN sr.outcome = 'win' THEN 1 ELSE 0 END) as wins,
    ROUND(100.0 * SUM(CASE WHEN sr.outcome = 'win' THEN 1 ELSE 0 END) / COUNT(*), 2) as win_rate,
    ROUND(AVG(sr.pnl_percent), 2) as avg_pnl
FROM signal_results sr
JOIN signals s ON sr.signal_id = s.id
WHERE sr.outcome IN ('win', 'loss')
GROUP BY s.strategy
ORDER BY win_rate DESC;"
```

### Daily Monitoring Script

Save as `scripts/monitor_phase1.sh`:

```bash
#!/bin/bash
# Daily Phase 1 monitoring

echo "=== V7 Phase 1 Monitoring Report ==="
echo "Date: $(date)"
echo ""

echo "=== Runtime Status ==="
if ps aux | grep v7_runtime_phase1 | grep -v grep > /dev/null; then
    echo "✅ Phase 1 Runtime: RUNNING"
    ps aux | grep v7_runtime_phase1 | grep -v grep | awk '{print "   PID: "$2", Uptime: "$10}'
else
    echo "❌ Phase 1 Runtime: NOT RUNNING"
fi

if ps aux | grep "v7_runtime.py" | grep -v grep > /dev/null; then
    echo "✅ Current V7 Runtime: RUNNING"
    ps aux | grep "v7_runtime.py" | grep -v grep | awk '{print "   PID: "$2", Uptime: "$10}'
else
    echo "⚠️  Current V7 Runtime: NOT RUNNING"
fi

echo ""
echo "=== Performance Comparison ==="
sqlite3 /root/crpbot/tradingai.db <<EOF
.mode column
.headers on
SELECT
    COALESCE(s.strategy, 'v7_current') as variant,
    COUNT(*) as trades,
    SUM(CASE WHEN sr.outcome = 'win' THEN 1 ELSE 0 END) as wins,
    ROUND(100.0 * SUM(CASE WHEN sr.outcome = 'win' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate,
    ROUND(AVG(sr.pnl_percent), 2) as avg_pnl,
    ROUND(SUM(sr.pnl_percent), 2) as total_pnl
FROM signal_results sr
JOIN signals s ON sr.signal_id = s.id
WHERE sr.outcome IN ('win', 'loss')
AND sr.created_at > datetime('now', '-7 days')
GROUP BY s.strategy
ORDER BY win_rate DESC;
EOF

echo ""
echo "=== Recent Phase 1 Signals ==="
sqlite3 /root/crpbot/tradingai.db <<EOF
.mode column
.headers on
SELECT
    datetime(timestamp, 'localtime') as time,
    symbol,
    direction,
    ROUND(confidence, 2) as conf
FROM signals
WHERE strategy = 'v7_phase1'
ORDER BY timestamp DESC
LIMIT 5;
EOF

echo ""
echo "=== Log Tail (Last 20 lines) ==="
tail -20 /tmp/v7_phase1_*.log | tail -20
```

Run daily:
```bash
chmod +x scripts/monitor_phase1.sh
./scripts/monitor_phase1.sh
```

---

## 🧪 Testing Before Deployment

### 1. Dry Run (No Database Writes)

```bash
# Test Phase 1 initialization only
.venv/bin/python3 -c "
from apps.runtime.v7_runtime_phase1 import V7Phase1Runtime, V7Phase1Config
config = V7Phase1Config(symbols=['BTC-USD'])
runtime = V7Phase1Runtime(runtime_config=config)
print('✅ Phase 1 runtime initialized successfully')
"
```

### 2. Single Iteration Test

```bash
# Run one iteration and exit
.venv/bin/python3 apps/runtime/v7_runtime_phase1.py \
  --iterations 1 \
  --variant "v7_phase1_test"
```

### 3. Monitor Logs

```bash
# Check for errors
grep -i "error\|exception\|failed" /tmp/v7_phase1_*.log | head -20

# Check Phase 1 specific logs
grep -E "Kelly|Exit Strategy|Correlation|Regime" /tmp/v7_phase1_*.log | head -20
```

---

## ⚠️ Troubleshooting

### Issue: Phase 1 not generating signals

**Check**:
1. Kelly fraction might be 0% (negative EV)
   ```bash
   grep "Kelly Updated" /tmp/v7_phase1_*.log
   ```
2. Regime filtering might be blocking all signals
   ```bash
   grep "Signal filtered by regime" /tmp/v7_phase1_*.log
   ```

**Solution**:
- If Kelly = 0%, wait for more baseline trades or temporarily use default 10%
- If regime blocking all, check market regime detection
- Can disable regime filtering: `enable_regime_filtering=False` in config

### Issue: Import errors

**Check**:
```bash
.venv/bin/python3 -c "
from libs.risk.kelly_criterion import KellyCriterion
from libs.risk.exit_strategy import ExitStrategy
from libs.risk.correlation_analyzer import CorrelationAnalyzer
from libs.risk.regime_strategy import RegimeStrategyManager
print('✅ All Phase 1 components import successfully')
"
```

**Solution**:
- Ensure all Phase 1 files are present in `libs/risk/`
- Check git pull completed successfully

### Issue: Database connection errors

**Check**:
```bash
sqlite3 /root/crpbot/tradingai.db "SELECT COUNT(*) FROM signals;"
```

**Solution**:
- Verify `DB_URL=sqlite:///tradingai.db` in `.env`
- Check database file exists and has correct permissions

---

## 📈 Success Criteria

### Week 1 (Data Collection)
- [ ] Phase 1 runtime running stable for 7 days
- [ ] Both variants generating signals (if A/B testing)
- [ ] No crashes or errors in logs
- [ ] At least 20 Phase 1 paper trades completed

### Week 2 (Evaluation)
- [ ] Calculate Sharpe ratio for Phase 1
- [ ] Compare with baseline V7
- [ ] Win rate improved by 10+ points
- [ ] P&L positive

### Decision Criteria

**Deploy to Production IF**:
- Phase 1 Sharpe > 1.0
- Phase 1 Win Rate > 45%
- Phase 1 Total P&L > 0%
- No critical bugs or crashes

**Iterate to Phase 2 IF**:
- Phase 1 Sharpe < 1.0
- Improvements marginal (< 5 point win rate increase)
- Need additional enhancements

---

## 🔄 Rollback Procedure

If Phase 1 has issues:

### Option 1: Stop Phase 1, Keep Current V7

```bash
# Stop Phase 1
pkill -f v7_runtime_phase1

# Verify current V7 still running
ps aux | grep "v7_runtime.py" | grep -v grep
```

### Option 2: Revert to Baseline

```bash
# Stop all runtimes
pkill -f v7_runtime

# Start baseline V7
nohup .venv/bin/python3 apps/runtime/v7_runtime.py \
  --iterations -1 \
  --sleep-seconds 300 \
  --max-signals-per-hour 10 \
  > /tmp/v7_runtime_$(date +%Y%m%d_%H%M).log 2>&1 &
```

---

## 📝 Change Log

### Phase 1 Integration (2025-11-24)

**Added**:
- `apps/runtime/v7_runtime_phase1.py` - Phase 1 enhanced runtime
- Kelly Criterion position sizing integration
- Exit strategy management integration
- Correlation analysis integration
- Market regime filtering integration

**Modified**:
- None (Phase 1 extends V7, doesn't modify it)

**Configuration**:
- New config class: `V7Phase1Config`
- Extends `V7RuntimeConfig` with Phase 1 parameters

---

## 📞 Support

### For QC Claude (Local)
- Review deployment logs
- Analyze performance metrics
- Suggest optimization if needed

### For Builder Claude (Cloud)
- Execute deployment
- Monitor daily performance
- Collect data for evaluation

### Emergency Contact
- Check logs first: `/tmp/v7_phase1_*.log`
- Review recent commits: `git log -5`
- Rollback if critical: See Rollback Procedure above

---

**Deployment Guide Version**: 1.0
**Last Updated**: 2025-11-24
**Status**: ✅ Ready for A/B Testing
