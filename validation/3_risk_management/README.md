# Risk Management

**File**: `hydra_guardian.py`

**Status**: IMPLEMENTED (Monitoring Only)

---

## What This Module Does

HYDRA Guardian is a **system monitoring tool**, NOT a trading risk manager.

### Guardian Functions:

1. **Process Health Monitoring**
   - Checks if HYDRA runtime is running
   - Auto-restarts on failure (max 3 attempts)
   - Alerts via Telegram on crashes

2. **API Credit Monitoring**
   - Scans logs for 402/429 errors (credit exhausted)
   - Alerts when DeepSeek, Claude, Grok, or Gemini credits run out
   - Prevents silent failures

3. **Disk Usage Monitoring**
   - Checks root partition usage
   - Auto-rotates logs if usage > 95%
   - Prevents disk-full crashes

4. **Paper Trading Activity**
   - Monitors time since last trade
   - Alerts if no trades in 2+ hours
   - Detects stuck processes

5. **Directional Bias Detection**
   - Analyzes BUY/SELL distribution after 20+ trades
   - Alerts if > 90% in one direction (regime bias bug)

---

## What This Module Does NOT Do

**HYDRA Guardian does NOT manage trading risk:**

- ❌ No position sizing
- ❌ No stop-loss enforcement
- ❌ No daily/max drawdown limits
- ❌ No spread filtering
- ❌ No account balance tracking

**Why?** HYDRA operates in **paper trading mode only**. All trades are simulated in JSONL files, not executed on real exchanges.

---

## Future Trading Risk Management

When HYDRA moves to live trading, a separate risk module will be needed:

### Planned Features:
1. **Position Sizing**
   - Risk per trade (1-2% of capital)
   - Reduced size on exotic pairs (50%)

2. **Drawdown Limits**
   - Daily loss cap (2%)
   - Max drawdown (6%)
   - Kill switch on breach

3. **Spread Filtering**
   - Reject trades with excessive spread
   - Protect against slippage

4. **Exchange Integration**
   - FTMO, Binance, Bybit connectors
   - Limit orders with spread checks

---

## Guardian Configuration

**Check Interval**: 300 seconds (5 minutes)

**Auto-Actions**:
- Process restart (up to 3 times per session)
- Log rotation (keeps last 10,000 lines)

**Alerts** (via Telegram):
- Process down
- API credits exhausted
- Disk usage > 95%
- No trades in 2+ hours
- Directional bias > 90%

**Cooldown**: 1 hour (prevents alert spam)

---

## Current Production Status

**Guardian PID**: Running
**Check Interval**: 5 minutes
**Telegram**: Enabled
**Last Check**: All systems healthy
**HYDRA Status**: UP (251+ trades, 64% BUY / 36% SELL)

---

## Files

**Guardian**: `apps/runtime/hydra_guardian.py`
**Startup Script**: `/root/crpbot/start_guardian.sh`
**Log**: `/tmp/hydra_guardian_*.log`

---

**Date**: 2025-11-30
