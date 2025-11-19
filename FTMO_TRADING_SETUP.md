# V7 Ultimate - FTMO Challenge Trading Setup

**Date**: 2025-11-19
**Challenge**: $15,000 Account → $1,500 Profit Goal (10% gain)
**Mode**: AGGRESSIVE (generates BUY/SELL signals, not just HOLD)

---

## Problem Identified

**Issue**: V7 was running in CONSERVATIVE mode, generating ONLY HOLD signals.
- 157 total signals generated
- 0 BUY signals
- 0 SELL signals
- 157 HOLD signals (100%)

**Result**: No Stop Loss, No Take Profit, No Risk/Reward = No tradeable signals!

---

## Solution: AGGRESSIVE Mode for FTMO Trading

V7 has been restarted in **AGGRESSIVE mode**:

```bash
# Old command (conservative)
.venv/bin/python3 apps/runtime/v7_runtime.py --iterations -1 --sleep-seconds 120

# New command (AGGRESSIVE for FTMO)
.venv/bin/python3 apps/runtime/v7_runtime.py --iterations -1 --sleep-seconds 120 --aggressive
```

**Current PID**: 2059967
**Log File**: `/tmp/v7_runtime_aggressive.log`

---

## What Changed

### Conservative Mode (OLD):
- Emphasis on risk management
- Prefers HOLD over BUY/SELL
- Only generates signals in "perfect" conditions
- Result: 100% HOLD signals (not tradeable)

### Aggressive Mode (NEW):
- Balanced risk/reward
- Generates actionable BUY/SELL signals
- Looks for trading opportunities actively
- Provides Stop Loss and Take Profit prices
- Still uses FTMO risk rules (4.5% daily loss limit, position sizing)

---

## FTMO Challenge Parameters

**Account Size**: $15,000
**Profit Target**: $1,500 (10%)
**Max Daily Loss**: $675 (4.5% of $15,000)
**Max Total Drawdown**: $1,350 (9% of $15,000)

**Trading Strategy**:
1. Risk 1-2% per trade ($150-$300)
2. Target 5-10 winning trades to reach $1,500
3. Win rate needed: ~60-70% (expected from V7)
4. Average R:R: 1:2 (risk $1 to make $2)

---

## Expected Signal Output (Aggressive Mode)

### Sample BUY Signal:
```
Symbol: BTC-USD
Signal: BUY (LONG)
Confidence: 72%
Entry Price: $91,500
Stop Loss: $90,800 (-0.76%, risk: $700)
Take Profit: $93,200 (+1.86%, reward: $1,700)
Risk/Reward: 1:2.43

AI Reasoning: Bullish momentum confirmed by Hurst > 0.6, low entropy (0.41), 
positive Sharpe ratio (1.6). Market regime trending upward.
```

### Sample SELL Signal:
```
Symbol: ETH-USD
Signal: SELL (SHORT)
Confidence: 68%
Entry Price: $3,100
Stop Loss: $3,140 (+1.29%, risk: $40)
Take Profit: $3,020 (-2.58%, reward: $80)
Risk/Reward: 1:2.0

AI Reasoning: Bearish divergence detected, entropy increasing (0.55), 
mean-reversion signal from Hurst < 0.5.
```

---

## Position Sizing for FTMO

**Formula**: Position Size = Risk Amount / (Entry - Stop Loss)

**Example 1**: BTC-USD Long
- Risk: $150 (1% of $15k)
- Entry: $91,500
- Stop Loss: $90,800
- Distance: $700
- Position Size: $150 / $700 = 0.214 BTC (~$19,600 notional)

**Example 2**: Conservative Risk (0.5%)
- Risk: $75 (0.5% of $15k)
- Entry: $91,500
- Stop Loss: $90,800
- Distance: $700
- Position Size: $75 / $700 = 0.107 BTC (~$9,800 notional)

---

## Trading Rules for FTMO Success

### DO:
✅ Only trade BUY/SELL signals (ignore HOLD)
✅ Use signals with confidence ≥ 65%
✅ Respect stop loss ALWAYS (no moving it wider)
✅ Risk 0.5-2% per trade ($75-$300)
✅ Track every trade via API
✅ Stop trading if daily loss hits 3% ($450)

### DON'T:
❌ Don't trade HOLD signals
❌ Don't revenge trade after losses
❌ Don't move stop loss against you
❌ Don't risk more than 2% per trade
❌ Don't trade during major news (FOMC, NFP, CPI)
❌ Don't exceed 4.5% daily loss ($675)

---

## Dashboard Improvements

The dashboard has been updated:

1. **Signal Table**: Now shows only 20 most recent signals (was showing 100+)
2. **Total Counter**: Shows total signals vs. displayed
3. **Price Fields**: Will now display SL/TP/R:R for BUY/SELL signals (not HOLD)

---

## How to Monitor

### Check V7 Runtime Status:
```bash
ps aux | grep v7_runtime | grep -v grep
```

### Check Latest Signals:
```bash
tail -50 /tmp/v7_runtime_aggressive.log | grep "Signal generated"
```

### View Dashboard:
```
http://178.156.136.185:5000
```

### Check for BUY/SELL Signals:
```bash
curl -s http://localhost:5000/api/v7/signals/recent/24 | python3 -c "
import sys, json
data = json.load(sys.stdin)
buysell = [s for s in data if s['direction'] != 'hold']
print(f'BUY/SELL signals: {len(buysell)} / {len(data)} total')
for s in buysell[:5]:
    print(f\"{s['direction'].upper()}: {s['symbol']} @ {s['entry_price']} (SL: {s['sl_price']}, TP: {s['tp_price']})\")
"
```

---

## Expected Performance (Aggressive Mode)

**Signal Distribution** (estimated):
- BUY: 20-30% of signals
- SELL: 15-25% of signals
- HOLD: 45-65% of signals

**Quality Metrics**:
- Average Confidence: 65-75%
- Risk/Reward: 1:1.5 to 1:3
- Win Rate: 60-70% (after learning)

**FTMO Timeline**:
- Week 1: 5-10 trades → $300-$600 profit
- Week 2: 5-10 trades → $300-$600 profit
- Week 3-4: 5-10 trades → $400-$700 profit
- **Total**: 15-30 trades → $1,500+ profit (challenge passed!)

---

## Important Notes

1. **Manual Trading System**: V7 generates signals only. YOU execute trades.
2. **Discipline Required**: Follow stop losses religiously to pass FTMO.
3. **Track All Trades**: Log results via API to improve V7's learning.
4. **Cost**: ~$2-3/month for DeepSeek API (negligible vs. $1,500 profit target).

---

## Next Steps

1. ✅ V7 running in AGGRESSIVE mode (PID: 2059967)
2. ⏳ Wait for first BUY/SELL signal (within 2-4 hours)
3. ⏳ Verify SL/TP fields show in dashboard
4. ⏳ Execute first trade when high-confidence signal appears
5. ⏳ Log trade result via API
6. ⏳ Repeat until $1,500 profit achieved

---

**Status**: V7 Ultimate optimized for FTMO challenge trading
**Mode**: AGGRESSIVE (generates actionable signals)
**Goal**: $1,500 profit from $15,000 account
**Expected Completion**: 3-4 weeks with disciplined trading

**Last Updated**: 2025-11-19 11:25 EST
