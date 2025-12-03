# HYDRA 3.0 Dashboard - Usage Instructions

**Dashboard URL**: http://178.156.136.185:3000

---

## 🚨 IMPORTANT: First Time Loading

When you first open the dashboard, you will see "Last Update: Never" with all zeros.

**You MUST click the "Refresh" button** (top right, next to "Live" badge) to load data from Mother AI.

After clicking Refresh, you should see:
- Last Update: Current time (e.g., "20:45:32")
- Gladiator action counts
- Performance stats (trades, P&L, win rate)
- Recent trades (if any)

---

## 📊 Dashboard Layout

### Header
- **Live** badge: Dashboard is connected
- **Last Update**: Timestamp of last data refresh
- **Refresh** button: Click to manually load latest Mother AI data

### Gladiators Section
Shows 4 independent trading agents:
- **Gladiator A (DeepSeek)**: Structural edge analysis
- **Gladiator B (Claude)**: Logic validation
- **Gladiator C (Grok)**: Fast backtesting
- **Gladiator D (Gemini)**: Strategy synthesis

Each card shows:
- Provider (AI model)
- Role (what they analyze)
- **Actions**: Total number of trades this gladiator has executed

### Performance Section
Aggregated stats across all 4 gladiators:
- **Total Trades**: Combined trades from all gladiators
- **Open Positions**: Currently active trades
- **Win Rate**: Percentage of closed trades that were profitable
- **Total P&L**: Cumulative profit/loss across all gladiators

### Recent Trades Table
Shows last 10 trades with:
- Asset (BTC-USD, ETH-USD, SOL-USD)
- Direction (LONG/SHORT)
- Entry Price
- Status (OPEN/CLOSED)
- P&L (profit/loss percentage)

---

## 🔄 How Data Updates

### Mother AI → Dashboard Flow

1. **Mother AI runs every 5 minutes** (300 seconds)
2. **After each cycle**, Mother AI:
   - Makes trading decisions (4 gladiators analyze market)
   - Updates portfolios
   - Saves state to disk: `/root/crpbot/data/hydra/mother_ai_state.json`
3. **Dashboard reads this file** when you click "Refresh"

### Auto-Refresh

The dashboard does NOT auto-refresh by default. You must:
- Click the **"Refresh"** button to manually update
- OR reload the page (but still need to click Refresh after reload)

---

## 📁 Data Source

**Primary**: `/root/crpbot/data/hydra/mother_ai_state.json`

This JSON file contains:
```json
{
  "timestamp": "2025-12-02T01:20:17Z",
  "cycle_count": 5,
  "gladiators": {
    "A": {"total_trades": 0, "wins": 0, ...},
    "B": {"total_trades": 0, "wins": 0, ...},
    "C": {"total_trades": 0, "wins": 0, ...},
    "D": {"total_trades": 0, "wins": 0, ...}
  },
  "rankings": [...],
  "recent_cycles": [...]
}
```

**Fallback**: If Mother AI state file doesn't exist, dashboard falls back to old HYDRA data:
- `/root/crpbot/data/hydra/paper_trades.jsonl`
- `/root/crpbot/data/hydra/hydra.db`

---

## 🐛 Troubleshooting

### "Last Update: Never" - No Data Showing

**Problem**: Dashboard hasn't loaded data yet

**Solution**: Click the **"Refresh"** button (top right)

---

### "All zeros" After Clicking Refresh

**Problem**: No trades have been executed yet (all gladiators on HOLD)

**Why**: Markets are CHOPPY (ranging, no clear trend). Mother AI is conservative and waits for better conditions.

**Normal Behavior**: This is INTENTIONAL - the system avoids low-probability trades.

**When will trades appear?**:
- When market regime shifts from CHOPPY to TRENDING or VOLATILE
- Mother AI identifies structural edges (liquidations, funding divergences, orderbook imbalances)

**Check Mother AI Activity**:
```bash
tail -20 /tmp/mother_ai_production.log
```

Look for lines like:
- "Cycle #X complete: 0 trades opened" (no opportunities found)
- "Gladiator [A/B/C/D] Decision: HOLD" (conservative mode)

---

### Refresh Button Not Working

**Check**:
1. Mother AI is running:
   ```bash
   ps aux | grep mother_ai_runtime | grep -v grep
   ```
   Should show PID

2. State file exists and is recent:
   ```bash
   ls -lh /root/crpbot/data/hydra/mother_ai_state.json
   ```
   Should show file with recent timestamp

3. Dashboard backend is running:
   ```bash
   ps aux | grep "reflex run" | grep -v grep
   ```
   Should show PID

**Restart Dashboard** (if needed):
```bash
# Kill dashboard (use sudo if pkill doesn't work)
sudo lsof -ti:3000 -ti:8000 | xargs -r sudo kill -9

# Restart dashboard
cd /root/crpbot/apps/dashboard_reflex
nohup /root/crpbot/.venv/bin/reflex run --loglevel info --backend-host 0.0.0.0 > /tmp/dashboard.log 2>&1 &

# Wait 10 seconds for startup
sleep 10

# Check it's running
curl -s http://localhost:3000 | head -10
```

**Note**: Dashboard must be restarted when code changes are made to `dashboard_reflex.py`. The Reflex process loads code on startup and doesn't auto-reload.

---

### Dashboard Shows Old HYDRA Data

**Problem**: Dashboard reading from old HYDRA files instead of Mother AI

**Check State File**:
```bash
ls -lh /root/crpbot/data/hydra/mother_ai_state.json
```

**If file doesn't exist**:
```bash
# Check Mother AI is running and saving state
tail -50 /tmp/mother_ai_production.log | grep -i "save\|state\|warning"
```

**If warnings about state save**:
```bash
# Restart Mother AI to fix persistence
pkill -f mother_ai_runtime
cd /root/crpbot
nohup .venv/bin/python3 apps/runtime/mother_ai_runtime.py \
  --assets BTC-USD ETH-USD SOL-USD \
  --iterations -1 \
  --interval 300 \
  --paper > /tmp/mother_ai_production.log 2>&1 &
```

---

## 📊 Expected Initial State

### First 30 Minutes (Cycles 1-6)

**What you'll see**:
- All gladiators: 0 trades
- Total Trades: 0
- Win Rate: 0%
- Total P&L: 0%

**Why**: Markets are likely CHOPPY, Mother AI waits for structural edges

**This is NORMAL and GOOD** - system is protecting capital

### After First Trades

**When**: Market regime shifts to TRENDING or high-volatility

**What changes**:
- Gladiator action counts increase (e.g., A: 3, B: 1, C: 2, D: 4)
- Total Trades increments
- Win Rate calculates (e.g., 66.7% if 2 wins, 1 loss)
- Total P&L shows aggregate performance (e.g., +2.4%)
- Recent Trades table populates

---

## 🔍 Monitoring Cycle Activity

Even with 0 trades, Mother AI is ACTIVE. Check logs:

```bash
tail -50 /tmp/mother_ai_production.log
```

**You should see** (every 5 minutes):
```
✅ Cycle #X complete: 0 trades opened by 0 gladiators
📊 TOURNAMENT STANDINGS:
  #1 - Gladiator A | Weight: 25% | P&L: $+0.00 | WR: 0.0% | Trades: 0
  #2 - Gladiator B | Weight: 25% | P&L: $+0.00 | WR: 0.0% | Trades: 0
  #3 - Gladiator C | Weight: 25% | P&L: $+0.00 | WR: 0.0% | Trades: 0
  #4 - Gladiator D | Weight: 25% | P&L: $+0.00 | WR: 0.0% | Trades: 0
⏳ Sleeping for 300s until next cycle...
```

This confirms Mother AI is working, just being conservative.

---

## 📞 Support

### Check System Status

```bash
# Mother AI status
ps aux | grep mother_ai_runtime | grep -v grep

# Dashboard status
ps aux | grep "reflex run" | grep -v grep

# Latest Mother AI cycle
tail -30 /tmp/mother_ai_production.log

# State file freshness
stat /root/crpbot/data/hydra/mother_ai_state.json
```

### Quick Health Check

```bash
/root/crpbot/.venv/bin/python3 << 'EOF'
import json
from pathlib import Path
from datetime import datetime

state_file = Path("/root/crpbot/data/hydra/mother_ai_state.json")
if state_file.exists():
    with open(state_file, 'r') as f:
        state = json.load(f)

    timestamp = datetime.fromisoformat(state['timestamp'].replace('Z', '+00:00'))
    age_minutes = (datetime.now(timestamp.tzinfo) - timestamp).total_seconds() / 60

    print(f"✅ State file found")
    print(f"   Last update: {age_minutes:.1f} minutes ago")
    print(f"   Cycles: {state['cycle_count']}")
    print(f"   Total trades: {sum(g['total_trades'] for g in state['gladiators'].values())}")

    if age_minutes < 10:
        print(f"\n✅ Mother AI is active (recent update)")
    else:
        print(f"\n⚠️  Mother AI may be stuck (last update {age_minutes:.0f} min ago)")
else:
    print("❌ State file not found - Mother AI may not be running")
EOF
```

---

**Last Updated**: 2025-12-01 20:30 UTC
