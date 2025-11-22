# Current Status & Next Actions for Builder Claude
## V7 Monitoring Period (2025-11-22 to 2025-11-25)

**Date**: 2025-11-22
**Status**: ✅ V7 OPERATIONAL - Collecting Data
**Current Phase**: Monitoring & Data Collection
**Next Review**: 2025-11-25 (Monday)

---

## 📊 CURRENT STATUS (Verified 2025-11-22 14:52)

### ✅ What's Working

**V7 Runtime**:
- PID: 2620770
- Uptime: 6 hours stable
- Scan frequency: 5 minutes
- Symbols: 10 (BTC, ETH, SOL, XRP, DOGE, ADA, AVAX, LINK, MATIC, LTC)
- Status: ✅ RUNNING SMOOTHLY

**Theories**:
- All 11 theories operational (100%)
- Shannon Entropy, Hurst, Markov, Kalman, Bayesian, Monte Carlo
- Random Forest, Autocorrelation, Stationarity, Variance
- Plus: Market context integration

**Database**:
- Total signals: 4,075
- Signals (24h): 545
- Paper trades: 13
- Win rate: 53.8%
- Total P&L: +5.48%

**A/B Testing**:
- v7_deepseek_only: 69.2% avg confidence
- v7_full_math: 47.2% avg confidence
- Distribution: Active and working

**APIs**:
- DeepSeek: $0.19/$150 monthly budget (0.13% used)
- Coinbase: Working
- CoinGecko: Working

**Dashboard**:
- URL: http://178.156.136.185:3000
- Status: Running

---

## 🎯 CURRENT MISSION: Data Collection

**Goal**: Accumulate 20+ paper trades for statistical significance

**Current Progress**:
- Paper trades: 13/20 (65%)
- Estimated completion: 3-5 days (at ~3 trades/day)
- Target date: 2025-11-25

**Why Wait**:
- 13 trades insufficient for statistical analysis
- Need 20+ trades to calculate reliable Sharpe ratio
- Premature optimization wastes effort

---

## 📋 BUILDER CLAUDE: DAILY TASKS (During Waiting Period)

### Daily Monitoring Checklist (5-10 minutes/day)

**Run once per day** (any time):

```bash
cd /root/crpbot

# 1. Check V7 is still running
ps aux | grep v7_runtime | grep -v grep

# If NOT running, restart:
# nohup .venv/bin/python3 apps/runtime/v7_runtime.py \
#   --iterations -1 \
#   --sleep-seconds 300 \
#   --max-signals-per-hour 3 \
#   > /tmp/v7_runtime_$(date +%Y%m%d_%H%M).log 2>&1 &

# 2. Check paper trade count
sqlite3 tradingai.db "SELECT COUNT(*) as paper_trades FROM signal_results;"

# 3. Quick health check
sqlite3 tradingai.db "
SELECT
  'Signals (24h): ' || COUNT(*) as status
FROM signals
WHERE timestamp > datetime('now', '-24 hours');
"

# 4. Check for errors in logs (last 50 lines)
tail -50 /tmp/v7_runtime_*.log | grep -i error || echo "No errors"

# 5. Check disk space
df -h | grep -E "Filesystem|/$"

# 6. Check system resources
free -h
```

**Expected Results**:
- V7 running: 1 process found
- Paper trades increasing daily
- Signals: 400-600 per day
- No critical errors
- Disk space >20% free
- RAM usage <80%

**If Issues Found**:
- V7 crashed → Restart (see command above)
- Disk full → Clean old logs (`rm /tmp/v7_runtime_2025111*.log`)
- High RAM → Investigate with `top`, may need restart
- API errors → Check API keys in `.env`

---

### Weekly Deep Check (Once on 2025-11-25)

**On Monday 2025-11-25**, run full analysis:

```bash
cd /root/crpbot

# 1. Pull latest code (in case QC Claude updated anything)
git pull origin feature/v7-ultimate

# 2. Run comprehensive performance analysis
python3 analyze_v7_performance.py > V7_PERFORMANCE_20251125.txt

# 3. Check paper trade count
TRADE_COUNT=$(sqlite3 tradingai.db "SELECT COUNT(*) FROM signal_results;")
echo "Paper trades: $TRADE_COUNT"

# 4. Decision time
if [ $TRADE_COUNT -ge 20 ]; then
    echo "✅ SUFFICIENT DATA - Ready for Phase 1 decision"
    echo "Action: Analyze results and decide on enhancements"
else
    echo "⏳ NEED MORE DATA - Continue monitoring"
    echo "Action: Wait until $TRADE_COUNT >= 20"
fi

# 5. Create weekly status report
cat > WEEKLY_STATUS_20251125.md <<EOF
# V7 Weekly Status Report

**Date**: $(date)
**Review Period**: 2025-11-22 to 2025-11-25

## Paper Trading Results
- Total Trades: $TRADE_COUNT
- Win Rate: $(sqlite3 tradingai.db "SELECT ROUND(SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) FROM signal_results WHERE outcome IS NOT NULL;")%
- Total P&L: $(sqlite3 tradingai.db "SELECT ROUND(SUM(pnl_percent), 2) FROM signal_results;")%

## Signal Statistics
$(sqlite3 tradingai.db "SELECT direction, COUNT(*) as count FROM signals WHERE timestamp > datetime('now', '-7 days') GROUP BY direction;")

## Recommendation
$(if [ $TRADE_COUNT -ge 20 ]; then echo "Proceed to Phase 1 analysis"; else echo "Continue monitoring"; fi)
EOF

cat WEEKLY_STATUS_20251125.md

# 6. Push report to GitHub
git add WEEKLY_STATUS_20251125.md V7_PERFORMANCE_20251125.txt
git commit -m "docs: weekly V7 status report - 2025-11-25"
git push origin feature/v7-ultimate
```

---

## 🚫 WHAT NOT TO DO (During Waiting Period)

**DO NOT start these yet**:

- ❌ Phase 1 (10-hour quant plan) - Wait for 20+ trades
- ❌ Phase 2 (advanced components) - Wait for Phase 1 completion
- ❌ New feature development - V7 is working, let it collect data
- ❌ Major configuration changes - Don't disturb running system
- ❌ Aggressive tuning - Premature optimization

**Why Wait**:
- V7 is working correctly (53.8% win rate, +5.48% P&L)
- Need statistical sample size (20+ trades minimum)
- Data-driven decisions better than assumptions
- "If it ain't broke, don't fix it"

---

## ✅ WHAT TO DO (Optional Improvements)

**Low-priority fixes** (only if you have spare time):

### Fix 1: MATIC-USD Symbol Issue (30 minutes)

**Observed**: MATIC-USD had 0 signals in last 24h (might be data issue)

```bash
# Test MATIC data fetch
python3 -c "
from libs.data.coinbase_client import CoinbaseClient
client = CoinbaseClient()
candles = client.get_candles('MATIC-USD', granularity=3600, limit=10)
print(f'MATIC candles: {len(candles)}')
print(candles[0] if candles else 'No data')
"

# If data available, check why no signals generated
# May just be HOLD signals (which is fine)
```

**Only fix if**: MATIC data fetch fails completely

**Otherwise**: Ignore, might just be all HOLD signals

---

### Fix 2: Documentation Cleanup (1 hour)

**Optional**: Update CLAUDE.md with latest V7 status

```bash
# Current theory count discrepancy
# Some docs say 6, some 7, some 8, actually 11 theories

# Update CLAUDE.md (if time permits)
# Change: "8 theories" → "11 theories"
# Add: List of all 11 theory names
```

**Priority**: LOW - Documentation can wait

---

## 📅 TIMELINE & MILESTONES

| Date | Milestone | Action |
|------|-----------|--------|
| **2025-11-22** | ✅ Verification complete | V7 operational, 13 trades |
| **2025-11-23** | Daily check | Monitor, expect ~16 trades |
| **2025-11-24** | Daily check | Monitor, expect ~19 trades |
| **2025-11-25** | **DECISION POINT** | Run weekly analysis, decide next steps |

---

## 🎯 DECISION TREE (2025-11-25)

**When you run weekly analysis on Monday**:

```
IF paper_trades >= 20:
    RUN: Sharpe ratio calculation

    IF Sharpe < 1.0:
        ACTION: Start Phase 1 (10-hour quant plan)
        REASON: Performance needs improvement
        FILE: QUANT_FINANCE_10_HOUR_PLAN.md

    ELSE IF Sharpe >= 1.0 AND Sharpe < 1.5:
        ACTION: Monitor 1 more week
        REASON: Decent performance, need more data
        NEXT_REVIEW: 2025-12-02

    ELSE IF Sharpe >= 1.5:
        ACTION: Continue as-is, Phase 1 optional
        REASON: Excellent performance already
        NEXT_REVIEW: 2025-12-09

ELSE:  # paper_trades < 20
    ACTION: Continue monitoring
    REASON: Insufficient data for decision
    NEXT_REVIEW: 2025-11-27 (check again Wednesday)
```

---

## 📊 CALCULATE SHARPE RATIO (On 2025-11-25)

**When you have 20+ trades**, run this:

```bash
cat > calculate_sharpe.py <<'EOF'
"""Calculate Sharpe Ratio from paper trading results"""
import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect('tradingai.db')

# Get all paper trading results
df = pd.read_sql("""
    SELECT
        timestamp,
        pnl_percent / 100.0 as returns
    FROM signal_results
    WHERE pnl_percent IS NOT NULL
    ORDER BY timestamp
""", conn, parse_dates=['timestamp'])

conn.close()

if len(df) < 20:
    print(f"❌ Insufficient data: {len(df)} trades (need 20+)")
    exit(1)

# Calculate Sharpe ratio
returns = df['returns']
mean_return = returns.mean()
std_return = returns.std()
risk_free_rate = 0.02 / 252  # 2% annual risk-free rate, daily

# Sharpe = (mean - rf) / std
sharpe_ratio = (mean_return - risk_free_rate) / std_return if std_return > 0 else 0

# Annualized
sharpe_annual = sharpe_ratio * np.sqrt(252)  # Assuming daily trades

print(f"\n{'='*70}")
print("SHARPE RATIO CALCULATION")
print(f"{'='*70}")
print(f"Trades:              {len(df)}")
print(f"Mean Return:         {mean_return*100:.3f}%")
print(f"Std Deviation:       {std_return*100:.3f}%")
print(f"Sharpe Ratio (daily): {sharpe_ratio:.3f}")
print(f"Sharpe Ratio (annual): {sharpe_annual:.3f}")
print(f"{'='*70}")

if sharpe_annual >= 1.5:
    print("✅ EXCELLENT - V7 is performing very well")
    print("Recommendation: Continue as-is, Phase 1 optional")
elif sharpe_annual >= 1.0:
    print("✅ GOOD - V7 is performing decently")
    print("Recommendation: Monitor 1 more week")
elif sharpe_annual >= 0.5:
    print("⚠️  ACCEPTABLE - V7 has positive edge")
    print("Recommendation: Consider Phase 1 enhancements")
else:
    print("❌ NEEDS IMPROVEMENT - V7 underperforming")
    print("Recommendation: Start Phase 1 immediately")
EOF

python3 calculate_sharpe.py
```

---

## 📁 FILES TO MONITOR

**Check these logs periodically**:

- `/tmp/v7_runtime_*.log` - Latest V7 runtime log
- `tradingai.db` - Database (growing daily)
- `V7_PERFORMANCE_*.txt` - Performance reports
- `WEEKLY_STATUS_*.md` - Weekly status reports

**Clean up old logs** (if disk space low):
```bash
# Keep last 7 days only
find /tmp -name "v7_runtime_*.log" -mtime +7 -delete
```

---

## 🔄 SUMMARY: What Builder Claude Should Do Now

### **This Week (2025-11-22 to 2025-11-25)**:

1. **Daily (5 min/day)**:
   - Check V7 is running (`ps aux | grep v7_runtime`)
   - Verify paper trade count increasing
   - Check for errors in logs

2. **Monday 2025-11-25 (30 min)**:
   - Run weekly analysis (`analyze_v7_performance.py`)
   - Calculate Sharpe ratio (`calculate_sharpe.py`)
   - Decide on Phase 1 based on Sharpe
   - Create weekly status report
   - Push to GitHub

3. **Communication**:
   - Report to QC Claude on Monday with findings
   - Include Sharpe ratio and recommendation

### **DO NOT Do**:
- ❌ Start Phase 1 before 2025-11-25
- ❌ Make major changes to V7
- ❌ Disturb running system

### **IF Emergency** (V7 crashes, APIs fail):
- Restart V7 (see daily checklist)
- Check API keys in `.env`
- Report issue to QC Claude immediately

---

**Status**: ✅ Instructions Clear
**Next Major Action**: 2025-11-25 (Monday weekly review)
**Current Focus**: Let V7 run and collect data
**Builder Claude Role**: Monitor and maintain (not develop)
