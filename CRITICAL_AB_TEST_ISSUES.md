# CRITICAL A/B Testing Issues - Immediate Action Required

**Date**: 2025-11-21
**Status**: 🚨 SEVERE BUGS DETECTED - System NOT providing valid data

---

## 🔴 Critical Problems Identified

### 1. **A/B Test Strategy Imbalance** (SEVERE)

**Expected**: 50/50 split between v7_full_math and v7_deepseek_only
**Actual**: 1576 v7_full_math vs 54 v7_deepseek_only (97% vs 3%)

**Impact**: A/B test is invalid - not comparing strategies equally

**Root Cause**: Strategy alternation logic in `apps/runtime/v7_runtime.py` is not working correctly

---

### 2. **Paper Trading Win/Loss Inversion** (CRITICAL)

**Observed**: Logs show WINNING trades (+2.08%, +2.13%, +2.17%) being marked as LOSSES in database

**Example from logs**:
```
✅ PAPER EXIT: SOL-USD HOLD @ $122.86 | P&L: +2.17% | Reason: take_profit
```

**Example from database**:
```
SOL-USD | v7_full_math | loss | P&L: -2.17%
```

**Impact**:
- Dashboard shows 7.5% win rate when actual may be 92.5%
- All performance metrics are inverted
- Cannot trust any results

**Root Cause**: Sign inversion bug in `libs/tracking/paper_trader.py` when recording outcomes

---

### 3. **HOLD Signals Being Paper Traded** (MEDIUM)

**Expected**: HOLD signals should NOT create paper trades
**Actual**: System is paper trading HOLD signals

**Example from logs**:
```
Signal 3456 is HOLD - skipping paper trade
[But then immediately:]
✅ PAPER EXIT: SOL-USD HOLD @ $122.86 | P&L: +2.17%
```

**Impact**: Inflating trade count with meaningless HOLD positions

---

### 4. **Aggressive Mode = Terrible Results**

**Current Performance**:
- Win Rate: 7.5% (likely inverted, should be 92.5%)
- 804 total trades (way too many)
- All trades showing losses

**Aggressive Mode Side Effects**:
- Generating 30 signals/hour (too frequent)
- Low-quality signals being traded
- Overwhelming the paper trading system

---

## 📊 Current Dashboard Data (UNRELIABLE)

```
v7_full_math (WITH mathematical theories):
  Total Trades: 804
  Win Rate: 7.5%
  Wins / Losses: 60 / 724
  Avg P&L: -1.81%
  Profit Factor: 0.08

v7_deepseek_only (WITHOUT math theories):
  Total Trades: 0
  (No data - strategy not running)
```

**⚠️ THIS DATA IS COMPLETELY UNRELIABLE** - Do NOT make decisions based on it

---

## 🔧 Required Fixes (Priority Order)

### Fix 1: Paper Trading Sign Inversion (CRITICAL)

**File**: `libs/tracking/paper_trader.py`
**Issue**: P&L calculation has wrong sign

**Need to check**:
- Line where `pnl_percent` is calculated
- Line where `outcome` is determined (win/loss)
- Ensure positive P&L = win, negative P&L = loss

### Fix 2: Strategy Alternation Logic (CRITICAL)

**File**: `apps/runtime/v7_runtime.py`
**Issue**: Not alternating strategies properly (97% vs 3%)

**Current behavior**: Nearly always selecting v7_full_math
**Expected behavior**: Should alternate EVENLY between strategies

**Possible causes**:
- Strategy counter not persisting between runs
- Random selection not working (should be deterministic alternation)
- A/B test flag not being set correctly

### Fix 3: HOLD Signal Paper Trading (MEDIUM)

**File**: `libs/tracking/paper_trader.py`
**Issue**: HOLD signals creating paper trades despite being skipped

**Need to verify**:
- HOLD signals should never call `open_position()`
- Existing HOLD positions from before should be closed
- But NEW HOLD signals should not create trades

### Fix 4: Disable Aggressive Mode (IMMEDIATE)

**Current**: `--aggressive --max-signals-per-hour 30`
**Should be**: Remove `--aggressive`, reduce to `--max-signals-per-hour 6`

**Why**:
- 30 signals/hour is way too many for quality trading
- Aggressive mode lowers quality standards
- V7 is designed for 2-5 high-quality signals per day, not 30/hour

---

## 🛠️ Immediate Actions Required

### Step 1: STOP Current Runtime (URGENT)

```bash
ps aux | grep v7_runtime | grep -v grep | awk '{print $2}' | xargs -r kill -9
```

**Reason**: Current data is polluting the database with inverted results

### Step 2: Clear Bad Data

```bash
.venv/bin/python3 << 'EOF'
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta

engine = create_engine('sqlite:///tradingai.db')
with engine.connect() as conn:
    # Get count before deletion
    result = conn.execute(text("SELECT COUNT(*) FROM signal_results WHERE entry_timestamp >= datetime('now', '-24 hours')"))
    before_count = result.scalar()

    # Delete last 24 hours of paper trading results
    conn.execute(text("DELETE FROM signal_results WHERE entry_timestamp >= datetime('now', '-24 hours')"))
    conn.commit()

    print(f"❌ Deleted {before_count} unreliable paper trading results from last 24 hours")
    print("Database ready for fresh start after bugs are fixed")
EOF
```

### Step 3: Fix Code Issues

1. Review and fix `libs/tracking/paper_trader.py` for sign inversion
2. Review and fix `apps/runtime/v7_runtime.py` for strategy alternation
3. Add proper logging to show which strategy is selected each iteration

### Step 4: Restart with Correct Settings

```bash
# After fixes applied:
rm -rf /root/crpbot/**/__pycache__ 2>/dev/null

# Run WITHOUT aggressive mode, REDUCED frequency
nohup .venv/bin/python3 apps/runtime/v7_runtime.py \
  --iterations -1 \
  --sleep-seconds 600 \
  --max-signals-per-hour 6 \
  > /tmp/v7_ab_testing_FIXED.log 2>&1 &

# Monitor logs
tail -f /tmp/v7_ab_testing_FIXED.log | grep -E "(Strategy selected|Paper trade|WIN_RATE)"
```

---

## 📝 Verification Checklist

After fixes applied, verify:

- [ ] Strategy alternation is 50/50 (check every 10 signals)
- [ ] Winning trades show positive P&L in database
- [ ] Losing trades show negative P&L in database
- [ ] HOLD signals do NOT create new paper trades
- [ ] Signal frequency is 6/hour maximum (not 30)
- [ ] Dashboard shows balanced data for both strategies

---

## 🎯 Expected Behavior After Fixes

**Signal Generation**:
- 2-5 high-quality signals per day (not 30/hour)
- Even split: v7_full_math and v7_deepseek_only
- Clear logging: "Strategy selected: v7_full_math" or "Strategy selected: v7_deepseek_only"

**Paper Trading**:
- Positive P&L = Win (green ✅ in dashboard)
- Negative P&L = Loss (red ❌ in dashboard)
- HOLD signals skipped completely
- Realistic win rates (55-65%, not 7.5%)

**Dashboard**:
- Both strategies show data
- Metrics are comparable
- Results make sense (not all losses)

---

## 📚 Related Files

- `/root/crpbot/apps/runtime/v7_runtime.py` - Main runtime (strategy alternation)
- `/root/crpbot/libs/tracking/paper_trader.py` - Paper trading logic (P&L calculation)
- `/root/crpbot/libs/tracking/performance_tracker.py` - Performance metrics
- `/root/crpbot/apps/dashboard_reflex/dashboard_reflex/dashboard_reflex.py` - Dashboard
- `/root/crpbot/REFLEX_DASHBOARD_BACKEND_GUIDE.md` - Dashboard documentation
- `/root/crpbot/DASHBOARD_QUICK_REF.md` - Quick reference

---

**Last Updated**: 2025-11-21 11:45 EST
**Status**: Awaiting code fixes before restart
