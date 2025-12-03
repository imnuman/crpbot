# HYDRA 3.0 - Deployment Guide

**Complete guide for deploying HYDRA 3.0 to production with real money.**

---

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Paper Trading Validation](#paper-trading-validation)
3. [Account Setup](#account-setup)
4. [Configuration](#configuration)
5. [Deployment Steps](#deployment-steps)
6. [Monitoring & Maintenance](#monitoring--maintenance)
7. [Emergency Procedures](#emergency-procedures)
8. [Gradual Expansion Plan](#gradual-expansion-plan)

---

## Pre-Deployment Checklist

### ✅ Phase 1: Paper Trading Validation (2 weeks minimum)

**Run HYDRA in paper trading mode for at least 2 weeks before deploying with real money.**

```bash
# Start paper trading
python apps/runtime/hydra_runtime.py \
  --assets BTC-USD ETH-USD SOL-USD \
  --paper \
  --iterations -1 \
  --interval 300
```

**Success Criteria:**

| Metric | Minimum Required | Recommended |
|--------|-----------------|-------------|
| Total Trades | 20+ | 50+ |
| Win Rate | 45%+ | 55%+ |
| Sharpe Ratio | 1.0+ | 1.5+ |
| Max Drawdown | < 15% | < 10% |
| Avg R:R | 1.2+ | 1.5+ |

**Decision Matrix:**

- **Sharpe ≥ 1.5**: ✅ Proceed to live deployment
- **Sharpe 1.0-1.5**: ⚠️ Monitor 1 more week, then proceed
- **Sharpe < 1.0**: ❌ Review strategies, DO NOT deploy

**Check paper trading stats:**

```bash
# View statistics in logs every 10 iterations
tail -f /tmp/hydra_runtime_*.log | grep "PAPER TRADING STATISTICS" -A 10

# Or check the paper trades file
cat /root/crpbot/data/hydra/paper_trades.jsonl | tail -20
```

### ✅ Phase 2: System Verification

**1. Verify all components initialized:**

```bash
python apps/runtime/hydra_runtime.py --iterations 1 --paper | grep -E "(initialized|success)"
```

Expected output:
```
✓ All layers initialized
✓ 4 Gladiators initialized
✓ Paper Trading System initialized
✓ Tournament Manager initialized
✓ Lesson Memory initialized (X lessons loaded)
✓ Guardian initialized
```

**2. Verify API keys set:**

```bash
# Check environment variables
echo "DeepSeek: ${DEEPSEEK_API_KEY:0:10}..."
echo "Claude: ${ANTHROPIC_API_KEY:0:10}..."
echo "Groq: ${GROQ_API_KEY:0:10}..."
echo "Gemini: ${GEMINI_API_KEY:0:10}..."
echo "Coinbase: ${COINBASE_API_KEY_NAME:0:20}..."
```

All should return truncated keys (not empty).

**3. Verify database:**

```bash
# Check HYDRA database exists
ls -lh /root/crpbot/data/hydra/*.db 2>/dev/null || echo "SQLite DB not created yet (OK)"

# Check data directories
ls -lh /root/crpbot/data/hydra/
```

Expected:
- `lessons.jsonl` (lesson memory)
- `paper_trades.jsonl` (paper trades)
- `explainability/` (trade logs)

---

## Paper Trading Validation

### Running the 2-Week Test

**Start HYDRA in paper mode:**

```bash
# Create a screen session (so it runs in background)
screen -S hydra_paper

# Inside screen, start HYDRA
python apps/runtime/hydra_runtime.py \
  --assets BTC-USD ETH-USD SOL-USD \
  --paper \
  --iterations -1 \
  --interval 300 \
  > /tmp/hydra_paper_$(date +%Y%m%d).log 2>&1

# Detach from screen: Ctrl+A, then D
# Reattach anytime: screen -r hydra_paper
```

**Daily Monitoring (5-10 min/day):**

```bash
# 1. Check HYDRA is still running
ps aux | grep hydra_runtime | grep -v grep

# 2. View last 50 lines of log
tail -50 /tmp/hydra_paper_*.log

# 3. Check paper trading stats
grep "PAPER TRADING STATISTICS" /tmp/hydra_paper_*.log | tail -1 -A 10

# 4. Check for errors
grep -i "error\|exception\|failed" /tmp/hydra_paper_*.log | tail -20
```

**Weekly Review (30 min/week):**

```python
# Analyze paper trading results
from libs.hydra.paper_trader import get_paper_trader

trader = get_paper_trader()

# Overall stats
stats = trader.get_overall_stats()
print(f"Win Rate: {stats['win_rate']:.1%}")
print(f"Sharpe: {stats['sharpe_ratio']:.2f}")
print(f"Total P&L: {stats['total_pnl_percent']:+.2%}")

# By asset
for asset in ["BTC-USD", "ETH-USD", "SOL-USD"]:
    asset_stats = trader.get_stats_by_asset(asset)
    print(f"\n{asset}: {asset_stats['win_rate']:.1%} WR, {asset_stats['total_trades']} trades")

# By regime
for regime in ["TRENDING", "RANGING", "VOLATILE", "BREAKOUT", "CHOPPY"]:
    regime_stats = trader.get_stats_by_regime(regime)
    print(f"{regime}: {regime_stats['win_rate']:.1%} WR, {regime_stats['total_trades']} trades")
```

**Decision Point (After 2 weeks):**

1. Calculate Sharpe ratio from paper trading stats
2. If Sharpe ≥ 1.0 → Proceed to live deployment
3. If Sharpe < 1.0 → Extend paper trading 1 more week OR review strategies

---

## Account Setup

### FTMO Account (Recommended)

**Why FTMO:**
- Funded account (trade with their money, not yours)
- Strict risk management (aligns with HYDRA's Guardian)
- $100k account, keep 80% of profits
- Protects you from catastrophic losses

**Account Requirements:**
- Challenge cost: $155 (one-time)
- Account size: $100k (funded by FTMO)
- Max daily loss: 5% ($5k)
- Max total loss: 10% ($10k)
- Profit target: 10% ($10k) in 30 days

**HYDRA Configuration for FTMO:**
```bash
# .env file
FTMO_LOGIN=your_login
FTMO_PASSWORD=your_password
FTMO_SERVER=FTMO-Demo  # or FTMO-Server for funded account

# Risk settings (in code)
MAX_RISK_PER_TRADE=0.003  # 0.3% ($300 on $100k)
MAX_DAILY_LOSS_PCT=0.045  # 4.5% ($4,500)
MAX_TOTAL_LOSS_PCT=0.09   # 9% ($9,000)
```

### Alternative: Micro Account ($100)

**If starting smaller:**

- Coinbase Advanced Trade or similar exchange
- Minimum: $100 deposit
- Max risk per trade: 0.3% ($0.30)
- Max daily loss: 4.5% ($4.50)
- Max total loss: 9% ($9.00)

**Risk Settings:**
```python
# In .env
ACCOUNT_SIZE_USD=100
MAX_RISK_PER_TRADE=0.003  # $0.30
```

---

## Configuration

### Environment Variables (.env)

**Create `.env` file with:**

```bash
# === DATA PROVIDER ===
DATA_PROVIDER=coinbase
COINBASE_API_KEY_NAME=organizations/.../apiKeys/...
COINBASE_API_PRIVATE_KEY=-----BEGIN EC PRIVATE KEY-----...

# === LLM API KEYS ===
DEEPSEEK_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AI...

# === PREMIUM DATA (Optional) ===
COINGECKO_API_KEY=CG-...

# === TRADING ACCOUNT ===
# For live trading (DO NOT set if paper trading)
# FTMO_LOGIN=your_login
# FTMO_PASSWORD=your_password
# FTMO_SERVER=FTMO-Server

# === RISK MANAGEMENT ===
ACCOUNT_SIZE_USD=100000  # FTMO: 100000, Micro: 100
MAX_RISK_PER_TRADE=0.003  # 0.3%
MAX_DAILY_LOSS_PCT=0.045  # 4.5%
MAX_TOTAL_LOSS_PCT=0.09   # 9%

# === SAFETY ===
KILL_SWITCH=false  # Set to true to emergency stop
CONFIDENCE_THRESHOLD=0.65  # Minimum consensus confidence
MAX_SIGNALS_PER_HOUR=3  # Rate limit

# === DATABASE ===
DB_URL=sqlite:///data/hydra/hydra.db

# === NOTIFICATIONS (Optional) ===
# TELEGRAM_TOKEN=...
# TELEGRAM_CHAT_ID=...
```

---

## Deployment Steps

### Step 1: Final Pre-Launch Checks

**1. Verify paper trading results pass criteria:**

```bash
# Check Sharpe ratio ≥ 1.0
python -c "
from libs.hydra.paper_trader import get_paper_trader
stats = get_paper_trader().get_overall_stats()
print(f'Sharpe: {stats[\"sharpe_ratio\"]:.2f}')
assert stats['sharpe_ratio'] >= 1.0, 'Sharpe too low for deployment'
print('✓ Sharpe ratio check passed')
"
```

**2. Backup current state:**

```bash
# Backup paper trades and lessons
cp /root/crpbot/data/hydra/paper_trades.jsonl \
   /root/crpbot/data/hydra/paper_trades_backup_$(date +%Y%m%d).jsonl

cp /root/crpbot/data/hydra/lessons.jsonl \
   /root/crpbot/data/hydra/lessons_backup_$(date +%Y%m%d).jsonl
```

**3. Final code check:**

```bash
# Ensure latest code
cd /root/crpbot
git status
git log -1 --oneline

# Should be on feature/v7-ultimate branch with latest commit
```

### Step 2: Start Micro Live Deployment

**IMPORTANT: Start with 1 asset only!**

```bash
# Create production screen session
screen -S hydra_live

# Start HYDRA in LIVE mode (no --paper flag)
# START WITH 1 ASSET ONLY!
python apps/runtime/hydra_runtime.py \
  --assets BTC-USD \
  --iterations -1 \
  --interval 300 \
  > /tmp/hydra_live_$(date +%Y%m%d).log 2>&1

# Detach: Ctrl+A, then D
```

**⚠️ CRITICAL: This is now trading with REAL MONEY!**

### Step 3: First 24 Hours - Intensive Monitoring

**Monitor every 1-2 hours:**

```bash
# 1. Check HYDRA is running
ps aux | grep hydra_runtime | grep -v grep

# 2. Check last 20 lines
tail -20 /tmp/hydra_live_*.log

# 3. Check for any trades executed
grep "LIVE TRADE" /tmp/hydra_live_*.log

# 4. Check Guardian rejections
grep "rejected by Guardian" /tmp/hydra_live_*.log | wc -l

# 5. Check lessons learned
grep "NEW LESSON LEARNED" /tmp/hydra_live_*.log

# 6. Check errors
grep -i "error" /tmp/hydra_live_*.log | tail -10
```

**First Trade Alert:**

When you see first `LIVE TRADE` log:
1. ✅ Verify entry price is reasonable
2. ✅ Verify SL/TP are set correctly
3. ✅ Verify position size ≤ max risk (0.3%)
4. ✅ Check consensus level (should be STRONG or UNANIMOUS)
5. ✅ Review gladiator votes (should be 3/4 or 4/4)

### Step 4: First Week - Daily Monitoring

**Daily checklist (10-15 min/day):**

```bash
# Morning check
screen -r hydra_live  # View HYDRA logs
# Ctrl+A, D to detach

# Check performance
tail -100 /tmp/hydra_live_*.log | grep -E "(LIVE TRADE|Guardian|Lesson)"

# Check account balance (if FTMO or exchange supports API)
# TODO: Add balance check command
```

**Weekly review:**

```python
# Analyze live trading results (same as paper trading)
from libs.hydra.paper_trader import get_paper_trader

trader = get_paper_trader()
stats = trader.get_overall_stats()

print(f"Week 1 Live Results:")
print(f"  Total Trades: {stats['total_trades']}")
print(f"  Win Rate: {stats['win_rate']:.1%}")
print(f"  Total P&L: ${stats['total_pnl_usd']:+.2f} ({stats['total_pnl_percent']:+.2%})")
print(f"  Sharpe: {stats['sharpe_ratio']:.2f}")
```

---

## Monitoring & Maintenance

### Daily Monitoring Script

**Create `monitor_hydra.sh`:**

```bash
#!/bin/bash

echo "=========================================="
echo "HYDRA 3.0 Live Monitoring"
echo "=========================================="
echo ""

# Check if running
if ps aux | grep hydra_runtime | grep -v grep > /dev/null; then
    echo "✓ HYDRA is running"
else
    echo "✗ HYDRA is NOT running!"
    exit 1
fi

# Recent activity
echo ""
echo "Last 10 log lines:"
tail -10 /tmp/hydra_live_*.log

# Trade count
echo ""
trades=$(grep -c "LIVE TRADE" /tmp/hydra_live_*.log)
echo "Total trades today: $trades"

# Error check
echo ""
errors=$(grep -c -i "error" /tmp/hydra_live_*.log)
if [ $errors -gt 0 ]; then
    echo "⚠️ $errors errors detected"
    grep -i "error" /tmp/hydra_live_*.log | tail -5
else
    echo "✓ No errors"
fi

echo ""
echo "=========================================="
```

**Run daily:**

```bash
chmod +x monitor_hydra.sh
./monitor_hydra.sh
```

### Performance Tracking

**Weekly Performance Report:**

```python
# Create weekly_report.py
from libs.hydra.paper_trader import get_paper_trader
from libs.hydra.lesson_memory import get_lesson_memory
from libs.hydra.tournament_manager import get_tournament_manager
from datetime import datetime

def generate_weekly_report():
    # Paper/Live trading stats
    trader = get_paper_trader()
    stats = trader.get_overall_stats()

    print("="*60)
    print(f"HYDRA 3.0 - Weekly Report ({datetime.now().strftime('%Y-%m-%d')})")
    print("="*60)

    print("\n📊 Trading Performance:")
    print(f"  Total Trades: {stats['total_trades']}")
    print(f"  Win Rate: {stats['win_rate']:.1%}")
    print(f"  Total P&L: ${stats['total_pnl_usd']:+.2f} ({stats['total_pnl_percent']:+.2%})")
    print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    print(f"  Avg R:R: {stats['avg_rr']:.2f}")

    # Lesson memory
    lessons = get_lesson_memory()
    lesson_stats = lessons.get_lesson_stats()

    print("\n📚 Lessons Learned:")
    print(f"  Total Lessons: {lesson_stats['total_lessons']}")
    print(f"  Avg Severity: {lesson_stats['avg_severity']:.1f}/10")
    print(f"  Total Occurrences: {lesson_stats['total_occurrences']}")

    # Top lessons
    print("\n🔝 Top 5 Lessons:")
    for i, lesson in enumerate(lessons.get_top_lessons(5), 1):
        print(f"  {i}. {lesson['pattern']} (severity: {lesson['severity']}, {lesson['occurrences']}x)")

    # Tournament stats
    tournament = get_tournament_manager()
    print("\n🏆 Tournament Status:")
    for pop_key in list(tournament.populations.keys())[:5]:
        stats = tournament.get_population_stats(*pop_key.split(":"))
        print(f"  {pop_key}: {stats['population_size']} strategies, "
              f"avg fitness: {stats.get('avg_fitness', 0):.2f}")

    print("\n" + "="*60)

if __name__ == "__main__":
    generate_weekly_report()
```

**Run weekly:**

```bash
python weekly_report.py
```

---

## Emergency Procedures

### KILL SWITCH - Emergency Stop

**If something goes wrong:**

```bash
# Method 1: Kill switch via environment variable
export KILL_SWITCH=true

# Method 2: Stop the process
screen -r hydra_live
# Ctrl+C to stop

# Method 3: Force kill
pkill -f hydra_runtime
```

**When to use:**

- ✅ Unexpected losses exceeding 2% in a day
- ✅ Multiple consecutive losses (5+)
- ✅ API errors preventing proper execution
- ✅ Guardian failing to reject bad trades
- ✅ Any behavior that seems abnormal

### Recovery Procedure

**After emergency stop:**

1. **Analyze what went wrong:**

```bash
# Check last 100 lines before stop
tail -100 /tmp/hydra_live_*.log > emergency_analysis.log

# Check lessons learned
grep "NEW LESSON LEARNED" emergency_analysis.log

# Check trades executed
grep "LIVE TRADE" emergency_analysis.log
```

2. **Review lessons:**

```python
from libs.hydra.lesson_memory import get_lesson_memory
lessons = get_lesson_memory()

# Check recent lessons
for lesson in lessons.get_top_lessons(10):
    print(f"{lesson['pattern']}: {lesson['prevention']}")
```

3. **Fix root cause** (code bug, configuration error, etc.)

4. **Return to paper trading** until confident

5. **Restart live deployment** (if appropriate)

---

## Gradual Expansion Plan

### Phase 1: Single Asset (Week 1-2)

**Start:**
- Asset: BTC-USD only
- Regime: All regimes
- Target: 20+ trades, Sharpe > 1.0

**Criteria to proceed:**
- ✅ 20+ trades completed
- ✅ Win rate ≥ 50%
- ✅ Sharpe ≥ 1.0
- ✅ No major losses (> 2% in a day)
- ✅ No emergency stops needed

### Phase 2: Three Assets (Week 3-4)

**Add:**
- ETH-USD
- SOL-USD

**New monitoring:**
```bash
python apps/runtime/hydra_runtime.py \
  --assets BTC-USD ETH-USD SOL-USD \
  --iterations -1 \
  --interval 300
```

**Criteria to proceed:**
- ✅ 50+ trades total
- ✅ Win rate ≥ 52%
- ✅ Sharpe ≥ 1.2
- ✅ Positive P&L across all assets

### Phase 3: Full Deployment (Week 5+)

**Add all 12 niche markets:**

Exotic Forex:
- USD/TRY (Turkish Lira)
- USD/ZAR (South African Rand)
- USD/MXN (Mexican Peso)
- USD/BRL (Brazilian Real)

Meme Perps:
- BONK-PERP
- WIF-PERP
- PEPE-PERP
- SHIB-PERP

Standard (continued):
- XRP-USD
- DOGE-USD
- ADA-USD
- AVAX-USD

**Full monitoring:**
```bash
python apps/runtime/hydra_runtime.py \
  --assets BTC-USD ETH-USD SOL-USD XRP-USD DOGE-USD ADA-USD AVAX-USD \
  --iterations -1 \
  --interval 300
```

---

## Success Metrics

### Monthly Targets

| Month | Target Trades | Target Win Rate | Target Sharpe | Target P&L |
|-------|--------------|----------------|---------------|------------|
| 1 | 50+ | 50%+ | 1.0+ | +5%+ |
| 2 | 100+ | 52%+ | 1.2+ | +10%+ |
| 3 | 150+ | 55%+ | 1.5+ | +15%+ |

### Red Flags (Stop Trading If)

- ❌ Win rate drops below 40%
- ❌ Sharpe ratio < 0.5
- ❌ Consecutive losses > 7
- ❌ Daily loss > 4.5%
- ❌ Total loss > 9%
- ❌ Guardian fails to reject obviously bad trades
- ❌ Lessons not being learned (same failures repeating)

---

## Final Checklist Before Going Live

- [ ] Paper trading results show Sharpe ≥ 1.0
- [ ] All 4 gladiator API keys are set
- [ ] Coinbase API credentials are configured
- [ ] Guardian rules are active (test with `--iterations 1 --paper`)
- [ ] Lesson memory loaded previous lessons
- [ ] Tournament manager initialized
- [ ] Database connection successful
- [ ] Monitoring scripts ready
- [ ] Emergency procedures documented
- [ ] Account funded (FTMO or micro account)
- [ ] Risk limits configured (0.3% per trade)
- [ ] Kill switch tested
- [ ] **You understand this is real money and you can lose it**

---

## Support & Debugging

### Common Issues

**1. HYDRA stops unexpectedly:**
```bash
# Check for crash
tail -100 /tmp/hydra_live_*.log | grep -i "exception\|error"

# Check system resources
free -h
df -h
```

**2. No trades being executed:**
```bash
# Check consensus results
grep "consensus" /tmp/hydra_live_*.log | tail -20

# Check Guardian rejections
grep "Guardian" /tmp/hydra_live_*.log | tail -20
```

**3. API rate limits:**
```bash
# Check for rate limit errors
grep "rate limit" /tmp/hydra_live_*.log
```

---

## Conclusion

**HYDRA 3.0 is a sophisticated system. Start small, monitor closely, and expand gradually.**

**Remember:**
- This is real money - you can lose it
- Start with 1 asset, not 12
- Monitor intensively for first week
- Use kill switch if anything seems wrong
- Paper trading results don't guarantee live results
- Market conditions change - HYDRA adapts via evolution, but it's not perfect

**Good luck! 🚀**

---

*Last Updated: 2025-11-29*
*HYDRA Version: 3.0*
*Deployment Guide Version: 1.0*
