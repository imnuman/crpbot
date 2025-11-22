# QC Review for Builder Claude - V7 Production Status

**Date**: 2025-11-21
**Reviewer**: QC Claude (Local Machine)
**Reviewee**: Builder Claude (Cloud Server: 178.156.136.185)
**Purpose**: Comprehensive production status review and issue identification

---

## 🚨 CONTEXT

QC Claude has pulled latest changes from GitHub and found:
- **286 files changed** with ~39,000 insertions
- Claims of "V7 Ultimate complete and running in production"
- **CRITICAL_AB_TEST_ISSUES.md** showing severe bugs
- Multiple documentation inconsistencies

**We need clarity on what is actually working vs broken.**

---

## SECTION 1: PRODUCTION RUNTIME STATUS

### 1.1 What is Currently Running?

**Please check and answer:**

```bash
# Run this on cloud server and paste output:
ps aux | grep -E "(python|v7|v6)" | grep -v grep
```

**Questions:**
- [x] Is V7 runtime currently running? **YES** (PID 2582246)
- [x] Is V6 runtime currently running? **YES** (PID 226398)
- [ ] If V7 is running, what is the exact command used?
- [ ] If V7 is NOT running, why not? What failed?
- [ ] When was the last time V7 successfully generated a signal?

**Paste runtime command here:**
```bash
.venv/bin/python3 apps/runtime/v7_runtime.py --iterations -1 --sleep-seconds 900 --max-signals-per-hour 3
```

### 1.2 Runtime Logs

**Check the most recent runtime logs:**

```bash
# Run on cloud server:
ls -lt /tmp/*.log | head -10
tail -50 /tmp/v7*.log  # If V7 is running
```

**Questions:**
- [ ] What are the most recent log files?
- [ ] Are there any ERROR messages in recent logs?
- [ ] Are there any WARNING messages that look suspicious?
- [ ] Is the runtime generating signals or stuck in a loop?

**Paste last 50 lines of most recent log:**
```
[BUILDER: PASTE LOG OUTPUT HERE]
```

### 1.3 Signal Generation Status

**Check recent signals in database:**

```bash
# Run on cloud server:
sqlite3 tradingai.db "SELECT signal_id, symbol, direction, confidence, timestamp FROM signals ORDER BY timestamp DESC LIMIT 20;"
```

**Questions:**
- [ ] How many signals were generated in the last 24 hours?
- [ ] Are signals showing variety (BUY/SELL/HOLD) or all one type?
- [ ] What is the confidence range of recent signals?
- [ ] Are all 3 symbols (BTC/ETH/SOL) being analyzed?

**Paste database query results:**
```sql
[BUILDER: PASTE RESULTS HERE]
```

---

## SECTION 2: CRITICAL BUGS STATUS

### 2.1 Paper Trading Win/Loss Inversion

**From CRITICAL_AB_TEST_ISSUES.md:** Logs show WINNING trades (+2.08%, +2.13%) being marked as LOSSES in database.

**Questions:**
- [ ] Has this bug been FIXED? (YES/NO)
- [ ] If YES, when was it fixed and what commit?
- [ ] If NO, is it still inverting wins/losses?
- [ ] Can you show evidence of a recent trade with correct P&L sign?

**Paste evidence (log + database entry for same trade):**
```
[BUILDER: PASTE EVIDENCE HERE]
```

### 2.2 A/B Test Strategy Imbalance

**From CRITICAL_AB_TEST_ISSUES.md:** 97% vs 3% split instead of 50/50.

**Questions:**
- [ ] Is A/B testing currently enabled? (YES/NO)
- [ ] If YES, what is the current split (check database)?
- [ ] If NO, was it disabled? Why?

```bash
# Run this to check:
sqlite3 tradingai.db "SELECT strategy, COUNT(*) FROM signals GROUP BY strategy;"
```

**Paste results:**
```sql
[BUILDER: PASTE RESULTS HERE]
```

### 2.3 HOLD Signals Being Paper Traded

**From CRITICAL_AB_TEST_ISSUES.md:** HOLD signals creating paper trades (should skip).

**Questions:**
- [ ] Has this been fixed? (YES/NO)
- [ ] Are HOLD signals currently being skipped from paper trading?
- [ ] What is the logic that prevents HOLD from being traded?

**Paste relevant code snippet from paper_trader.py:**
```python
[BUILDER: PASTE CODE SNIPPET HERE]
```

---

## SECTION 3: DASHBOARD STATUS

### 3.1 Dashboard Access

**Questions:**
- [ ] Is dashboard currently running? (YES/NO)
- [ ] Can you access it at http://178.156.136.185:5000? (YES/NO)
- [ ] If NO, what error do you see?
- [ ] Which dashboard implementation is running? (Flask/Reflex/Other)

```bash
# Check what's running on port 5000:
lsof -i :5000
```

**Paste output:**
```
[BUILDER: PASTE OUTPUT HERE]
```

### 3.2 Dashboard Data Quality

**Visit the dashboard and check:**

**Questions:**
- [ ] Performance Tracking page: Is it showing data or blank?
- [ ] A/B Test page: Is it showing comparison or blank?
- [ ] Live prices: Are they updating?
- [ ] Signal count: How many signals shown?
- [ ] DeepSeek analysis box: Is it showing LLM reasoning?

**Take a screenshot or describe what you see:**
```
[BUILDER: DESCRIBE DASHBOARD STATE HERE]
```

---

## SECTION 4: DATABASE SCHEMA

### 4.1 Required Tables

**Check if all required tables exist:**

```bash
sqlite3 tradingai.db "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
```

**Required tables:**
- [ ] `signals` - exists? (YES/NO)
- [ ] `signal_results` - exists? (YES/NO)
- [ ] `theory_performance` - exists? (YES/NO)
- [ ] `paper_trades` - exists? (YES/NO)

**Paste table list:**
```sql
[BUILDER: PASTE TABLE LIST HERE]
```

### 4.2 Missing Tables

**If any tables are missing:**

**Questions:**
- [ ] Did you run `scripts/create_missing_tables.py`? (YES/NO)
- [ ] If YES, did it succeed or error?
- [ ] If NO, why not?

**Paste output of table creation script:**
```
[BUILDER: PASTE OUTPUT HERE]
```

---

## SECTION 5: API INTEGRATIONS

### 5.1 DeepSeek API

**Questions:**
- [ ] Is DeepSeek API key configured? (YES/NO)
- [ ] Is DeepSeek responding to requests? (YES/NO)
- [ ] What is the daily/monthly API cost so far?
- [ ] Have you hit any rate limits or errors?

**Check recent DeepSeek costs:**
```bash
# Check logs for DeepSeek cost tracking
grep -i "deepseek\|cost" /tmp/v7*.log | tail -20
```

**Paste cost info:**
```
[BUILDER: PASTE COST INFO HERE]
```

### 5.2 CoinGecko API

**Questions:**
- [ ] Is CoinGecko API key configured? (YES/NO)
- [ ] Is CoinGecko returning market data? (YES/NO)
- [ ] Are you getting rate limited? (YES/NO)
- [ ] What data is CoinGecko providing (MCap, Volume, ATH, etc.)?

**Test CoinGecko integration:**
```bash
# Run on cloud server:
.venv/bin/python3 -c "
from libs.data.coingecko_client import CoinGeckoClient
client = CoinGeckoClient()
data = client.get_market_context('bitcoin')
print(f'MCap: {data.get(\"market_cap\")}')
print(f'Volume: {data.get(\"volume_24h\")}')
print(f'ATH: {data.get(\"ath_distance\")}%')
"
```

**Paste output:**
```
[BUILDER: PASTE OUTPUT HERE]
```

### 5.3 Coinbase API

**Questions:**
- [ ] Is Coinbase API returning price data? (YES/NO)
- [ ] Are you getting real-time 1m candles? (YES/NO)
- [ ] Any connection errors or rate limits?

**Test Coinbase connection:**
```bash
# Run on cloud server:
.venv/bin/python3 -c "
from apps.runtime.data_fetcher import get_data_fetcher
from libs.config.config import Settings
fetcher = get_data_fetcher(Settings())
df = fetcher.get_ohlcv('BTC-USD', limit=10)
print(f'Rows: {len(df)}')
print(f'Latest: {df.tail(1).to_dict()}')
"
```

**Paste output:**
```
[BUILDER: PASTE OUTPUT HERE]
```

---

## SECTION 6: MATHEMATICAL THEORIES

### 6.1 Theory Count Confusion

**Documentation shows conflicting numbers:**
- Some docs say "6 theories"
- Some docs say "7 theories" (with CoinGecko)
- Some docs say "11 theories"

**Questions:**
- [ ] How many mathematical theories are ACTUALLY implemented?
- [ ] List each theory module file and confirm it exists:

**Check theory files:**
```bash
ls -1 libs/theories/*.py | grep -v __
```

**Paste file list:**
```
[BUILDER: PASTE FILE LIST HERE]
```

**Questions:**
- [ ] Which theories are ACTUALLY being used in V7 runtime?
- [ ] Are all theories being calculated for every signal?
- [ ] Are any theories disabled or skipped?

---

## SECTION 7: TELEGRAM BOT

### 7.1 Telegram Status

**Questions:**
- [ ] Is Telegram bot running? (YES/NO)
- [ ] Is it sending signal notifications? (YES/NO)
- [ ] When was the last Telegram notification sent?
- [ ] Are notifications formatted correctly (Entry/SL/TP prices)?

**Check Telegram logs:**
```bash
grep -i "telegram" /tmp/v7*.log | tail -20
```

**Paste output:**
```
[BUILDER: PASTE OUTPUT HERE]
```

---

## SECTION 8: PERFORMANCE DATA

### 8.1 Current Performance Metrics

**Run this query:**

```bash
sqlite3 tradingai.db "
SELECT
  strategy,
  COUNT(*) as total_trades,
  SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) as wins,
  SUM(CASE WHEN outcome='loss' THEN 1 ELSE 0 END) as losses,
  ROUND(AVG(pnl_percent), 2) as avg_pnl
FROM signal_results
GROUP BY strategy;
"
```

**Questions:**
- [ ] What are the current performance numbers?
- [ ] Do the numbers make sense (win rate between 20-80%)?
- [ ] Is avg_pnl positive or negative?
- [ ] Do you trust this data or suspect it's inverted?

**Paste query results:**
```sql
[BUILDER: PASTE RESULTS HERE]
```

---

## SECTION 9: FILE INCONSISTENCIES

### 9.1 Multiple Runtime Versions

**Found these runtime files:**
- `apps/runtime/v7_runtime.py`
- `apps/runtime/v6_runtime.py`
- `apps/runtime/v6_fixed_runtime.py`
- `apps/runtime/v6_statistical_adapter.py`

**Questions:**
- [ ] Which runtime are you ACTUALLY using in production?
- [ ] Why do we have 4 different runtime files?
- [ ] Should we delete the unused ones?

### 9.2 Dashboard Confusion

**Found these dashboard implementations:**
- `apps/dashboard/app.py` (Flask)
- `apps/dashboard_flask_backup/` (Flask backup)
- `apps/dashboard_reflex/` (Reflex)

**Questions:**
- [ ] Which dashboard implementation is running in production?
- [ ] Why do we have 3 dashboard versions?
- [ ] Should we delete the unused ones?

---

## SECTION 10: DOCUMENTATION CLEANUP

### 10.1 Essential vs Obsolete

**According to TODO.md, you kept 7 "essential" files:**
1. README.md
2. CLAUDE.md
3. PROJECT_MEMORY.md
4. MASTER_TRAINING_WORKFLOW.md
5. V7_CLOUD_DEPLOYMENT.md
6. V7_MATHEMATICAL_THEORIES.md
7. V7_MONITORING.md

**But I see 20+ docs in root directory:**
```bash
ls -1 *.md | wc -l
# Shows 29 files
```

**Questions:**
- [ ] Why are there 29 .md files if you cleaned up to 7?
- [ ] Which docs were created AFTER the cleanup?
- [ ] Which of these 29 files are actually needed vs redundant?

**List files created after cleanup:**
```bash
ls -lt *.md | head -20
```

**Paste file list with dates:**
```
[BUILDER: PASTE LIST HERE]
```

---

## SECTION 11: ACTION ITEMS

### 11.1 Immediate Fixes Needed

**Based on your answers above, what needs to be fixed IMMEDIATELY?**

**Priority 1 (Must fix now):**
1. [BUILDER: LIST P1 ISSUES]
2.
3.

**Priority 2 (Fix this week):**
1. [BUILDER: LIST P2 ISSUES]
2.
3.

**Priority 3 (Nice to have):**
1. [BUILDER: LIST P3 ISSUES]
2.
3.

### 11.2 Files to Delete

**Based on cleanup analysis, which files can be safely deleted?**

**Runtime files to delete:**
- [ ] `apps/runtime/v6_runtime.py` (if not used)
- [ ] `apps/runtime/v6_fixed_runtime.py` (if not used)
- [ ] `apps/runtime/v6_statistical_adapter.py` (if not used)

**Dashboard files to delete:**
- [ ] `apps/dashboard_flask_backup/` (if not used)
- [ ] `apps/dashboard_reflex/` (if not used)

**Documentation to delete/archive:**
- [BUILDER: LIST OBSOLETE DOCS]

---

## SECTION 12: DEPLOYMENT PLAN

### 12.1 Current Deployment State

**Questions:**
- [ ] Is V7 stable enough for 24/7 production? (YES/NO)
- [ ] If NO, what blockers remain?
- [ ] Are you confident in the paper trading results?
- [ ] Would you trust this system with real money? (YES/NO)

### 12.2 Recommended Actions

**QC Claude will review your answers and recommend:**
1. What to fix immediately
2. What to delete/cleanup
3. What documentation to create
4. Whether V7 is ready for production or needs more work

---

## INSTRUCTIONS FOR BUILDER CLAUDE

**Please fill out ALL sections above:**

1. Run every command shown
2. Paste actual outputs (not "it works" - show real data)
3. Be honest about what's broken vs working
4. Don't skip sections
5. Provide evidence for claims (logs, database queries, code snippets)

**Time estimate**: 30-45 minutes to complete thoroughly

**Once complete:**
1. Save this file with your answers
2. Commit and push to GitHub
3. QC Claude will review and create action plan

---

**Status**: ⏳ AWAITING BUILDER CLAUDE RESPONSE
