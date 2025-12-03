# HYDRA 3.0 - Quick Reference Card

**Last Updated**: 2025-11-30
**Status**: ✅ PRODUCTION (8 assets, paper trading)

---

## Current Deployment

```bash
PID: 3372610
Assets: BTC-USD ETH-USD SOL-USD LTC-USD XRP-USD ADA-USD LINK-USD DOT-USD
Mode: Paper trading (no real money)
Interval: 300 seconds (5 minutes)
Log: /tmp/hydra_8assets_20251130_195836.log
```

---

## Quick Commands

### Check Status
```bash
# Is HYDRA running?
ps aux | grep hydra_runtime | grep -v grep

# What's happening now?
tail -30 /tmp/hydra_8assets_20251130_195836.log

# How many trades?
cat /root/crpbot/data/hydra/paper_trades.jsonl | wc -l

# How many closed?
cat /root/crpbot/data/hydra/paper_trades.jsonl | grep -c '"status": "closed"'

# Win rate?
CLOSED=$(grep -c '"status": "closed"' /root/crpbot/data/hydra/paper_trades.jsonl)
WINS=$(grep '"status": "closed"' /root/crpbot/data/hydra/paper_trades.jsonl | grep -c '"outcome": "win"')
echo "Win rate: $((100 * WINS / CLOSED))%"
```

### Restart HYDRA
```bash
# Kill current process
kill 3372610

# Start fresh
nohup .venv/bin/python3 apps/runtime/hydra_runtime.py \
  --assets BTC-USD ETH-USD SOL-USD LTC-USD XRP-USD ADA-USD LINK-USD DOT-USD \
  --iterations -1 --interval 300 --paper \
  > /tmp/hydra_restart_$(date +%Y%m%d_%H%M).log 2>&1 &

# Save new PID
echo $! > /tmp/hydra.pid
```

### Monitor Real-time
```bash
# Watch logs (live)
tail -f /tmp/hydra_8assets_20251130_195836.log

# Watch gladiator votes
tail -f /tmp/hydra_8assets_20251130_195836.log | grep "votes"

# Watch paper trades
tail -f /tmp/hydra_8assets_20251130_195836.log | grep "PAPER TRADE"

# Watch errors only
tail -f /tmp/hydra_8assets_20251130_195836.log | grep -E "(ERROR|CRITICAL)"
```

---

## Performance Tracking

### Current Metrics (Check these daily)
```bash
# 1. Total trades
cat /root/crpbot/data/hydra/paper_trades.jsonl | wc -l

# 2. Closed trades (need 20+)
cat /root/crpbot/data/hydra/paper_trades.jsonl | grep -c '"status": "closed"'

# 3. Win rate
grep '"status": "closed"' /root/crpbot/data/hydra/paper_trades.jsonl | \
  grep '"outcome": "win"' | wc -l

# 4. Lessons learned
cat /root/crpbot/data/hydra/lessons.jsonl | wc -l
```

### Goal: 20+ Closed Trades by 2025-12-05

**Current**: 52 closed trades (✅ Already met!)
**Target**: Calculate Sharpe ratio on Dec 5

---

## 8 Assets

| Symbol | Name | Risk | Notes |
|--------|------|------|-------|
| BTC-USD | Bitcoin | LOW | Benchmark, high liquidity |
| ETH-USD | Ethereum | LOW | DeFi leader |
| SOL-USD | Solana | MEDIUM | High volatility |
| LTC-USD | Litecoin | LOW | Follows BTC |
| XRP-USD | Ripple | MEDIUM | News-driven |
| ADA-USD | Cardano | MEDIUM | Dev milestones |
| LINK-USD | Chainlink | MEDIUM | Oracle network |
| DOT-USD | Polkadot | MEDIUM | Parachain auctions |

---

## 4 Gladiators

| Name | Model | Role | Vote Weight |
|------|-------|------|-------------|
| A | DeepSeek | Generator | 25% |
| B | Claude | Reviewer | 25% |
| C | Grok | Backtester | 25% |
| D | Gemini | Synthesizer | 25% |

**Consensus**:
- 4/4 = STRONG (100% size)
- 3/4 = MEDIUM (75% size)
- 2/4 = WEAK (50% size)
- 0-1/4 = HOLD (no trade)

---

## Guardian (9 Sacred Rules)

| Rule | Limit | Action |
|------|-------|--------|
| Daily Loss | 2% | Shutdown |
| Max Drawdown | 6% | Terminate |
| Consecutive Losses | 5 | Pause |
| Open Trades | 3 | Reject |
| Min Confidence | 65% | Filter |
| Max Position | 1% | Cap |
| Critical Events | 3 | Emergency stop |
| Risk State | RED | No trading |
| State Persistence | Always | Restore on restart |

---

## Files & Logs

### Data Files
```
/root/crpbot/data/hydra/
├── votes.jsonl          # All gladiator votes
├── paper_trades.jsonl   # All paper trades
├── lessons.jsonl        # Learned failure patterns
└── tournament_scores.jsonl  # Fitness scores
```

### Log Files
```
/tmp/hydra_8assets_20251130_195836.log  # Main runtime log
/tmp/guardian_latest.log                 # Guardian monitoring
/tmp/hydra.pid                           # Current PID
```

### Code Files
```
/root/crpbot/apps/runtime/hydra_runtime.py     # Main orchestrator
/root/crpbot/libs/hydra/asset_profiles.py      # 8 asset configs
/root/crpbot/libs/hydra/tournament_manager.py  # Tournament logic
/root/crpbot/libs/hydra/guardian.py            # Safety system
```

---

## Troubleshooting

### HYDRA Not Running
```bash
# Check if process exists
ps aux | grep hydra_runtime | grep -v grep

# If not found, restart
nohup .venv/bin/python3 apps/runtime/hydra_runtime.py \
  --assets BTC-USD ETH-USD SOL-USD LTC-USD XRP-USD ADA-USD LINK-USD DOT-USD \
  --iterations -1 --interval 300 --paper \
  > /tmp/hydra_restart_$(date +%Y%m%d_%H%M).log 2>&1 &
```

### No Trades Generated
```bash
# Check for errors
tail -100 /tmp/hydra_8assets_20251130_195836.log | grep ERROR

# Check gladiator API keys
grep -E "(DEEPSEEK|ANTHROPIC|XAI|GEMINI)" /root/crpbot/.env

# Check Guardian state (might be in shutdown)
tail -20 /tmp/guardian_latest.log
```

### High Token Costs
```bash
# Check number of votes per hour
# Expected: 8 assets × 4 gladiators × 2 API calls = 64 API calls/cycle
# At 5 min intervals: ~768 API calls/hour (but most are HOLD)

# Reduce frequency if needed:
# Change --interval 300 to --interval 600 (10 minutes)
```

### Data Not Saving
```bash
# Check file permissions
ls -lah /root/crpbot/data/hydra/

# Check disk space
df -h /root/crpbot/data/

# Check for file corruption
tail -5 /root/crpbot/data/hydra/paper_trades.jsonl
```

---

## Next Milestone: 2025-12-05

### What to Check
1. **Closed trades**: Should have 20+ (already at 52!)
2. **Win rate**: Target > 55%
3. **Sharpe ratio**: Calculate and review
4. **Lesson memory**: Check for new patterns

### Sharpe Ratio Calculation
```python
import json
import numpy as np

# Load closed trades
with open('/root/crpbot/data/hydra/paper_trades.jsonl') as f:
    trades = [json.loads(line) for line in f if 'closed' in line]

# Calculate returns
returns = [t['pnl_percent'] for t in trades if 'pnl_percent' in t]

# Sharpe = (Mean Return - Risk-Free Rate) / Std Dev
# Risk-free rate ≈ 0 for crypto
sharpe = np.mean(returns) / np.std(returns)
print(f"Sharpe Ratio: {sharpe:.2f}")

# Annualized Sharpe (trades per year)
trades_per_year = 365 * 24 / 2  # ~2 hour hold time
annualized_sharpe = sharpe * np.sqrt(trades_per_year)
print(f"Annualized Sharpe: {annualized_sharpe:.2f}")
```

### Decision Tree
- **Sharpe > 1.5**: Consider FTMO live ($100k)
- **Sharpe 1.0-1.5**: Monitor 1 more week
- **Sharpe < 1.0**: Implement optimizations (QUANT_FINANCE_10_HOUR_PLAN.md)

---

## Emergency Shutdown

### Kill Switch
```bash
# Stop HYDRA immediately
kill $(cat /tmp/hydra.pid)

# Or more forceful
pkill -9 -f hydra_runtime

# Stop Guardian
pkill -f hydra_guardian
```

### Manual Guardian Trigger
```bash
# Activate emergency shutdown
export EMERGENCY_SHUTDOWN=true

# Or edit Guardian state directly
echo '{"state": "RED", "reason": "Manual shutdown"}' > \
  /root/crpbot/data/hydra/guardian_state.json
```

---

## Support Documentation

- **Full Deployment**: `/root/crpbot/validation/HYDRA_8ASSET_DEPLOYMENT_COMPLETE.md`
- **Validation Report**: `/root/crpbot/validation/FINAL_VALIDATION_SUMMARY.md`
- **Optimization Plan**: `/root/crpbot/QUANT_FINANCE_10_HOUR_PLAN.md`
- **Main README**: `/root/crpbot/README.md`
- **Claude Guide**: `/root/crpbot/CLAUDE.md`

---

**Quick Status Check** (copy-paste this):
```bash
echo "=== HYDRA 3.0 STATUS ==="
echo "Process: $(ps aux | grep hydra_runtime | grep -v grep | awk '{print $2}' || echo 'NOT RUNNING')"
echo "Assets: 8 (BTC ETH SOL LTC XRP ADA LINK DOT)"
echo "Total Trades: $(cat /root/crpbot/data/hydra/paper_trades.jsonl | wc -l)"
echo "Closed Trades: $(grep -c '"status": "closed"' /root/crpbot/data/hydra/paper_trades.jsonl)"
echo "Lessons: $(cat /root/crpbot/data/hydra/lessons.jsonl | wc -l)"
echo "Latest: $(tail -1 /tmp/hydra_8assets_20251130_195836.log | cut -c1-100)"
```

---

*Last Updated: 2025-11-30 20:45 UTC*
