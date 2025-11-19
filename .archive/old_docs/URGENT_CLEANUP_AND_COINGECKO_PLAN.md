# URGENT: Documentation Cleanup & CoinGecko Integration Plan

**Date**: 2025-11-19 14:45
**Severity**: CRITICAL
**Issue**: 266 documentation files causing important steps to be missed

---

## 🚨 ROOT CAUSE ANALYSIS

### Problem 1: Documentation Overload
- **266 total markdown files** in the project
- **171 files in root directory** alone
- Multiple conflicting/overlapping docs
- No single source of truth
- Critical steps (like CoinGecko) buried in outdated docs

### Problem 2: CoinGecko Integration Missing
- **Paid**: $129/month for CoinGecko Analyst API
- **API Key**: Available in `.env` (CG-VQhq64e59sGxchtK8mRgdxXW)
- **Status**: NOT INTEGRATED in V7 runtime
- **Why Missed**: Buried in 266 docs, no clear implementation checklist

### Problem 3: No Clear Execution Plan
- Too many planning docs, not enough action
- Steps documented but not tracked
- No single file showing "what needs to be done NOW"

---

## 📋 IMMEDIATE ACTION PLAN

### Phase 1: CoinGecko Integration (TODAY - 2 hours)

**Step 1**: Create CoinGecko client (30 min)
```python
# File: libs/data/coingecko_client.py
class CoinGeckoClient:
    """Fetch real-time market data from CoinGecko Analyst API"""

    def get_market_data(self, symbol: str) -> dict:
        """
        Fetch market cap, volume, ATH distance, sentiment
        Returns: {
            'market_cap': float,
            'total_volume': float,
            'ath_distance_pct': float,
            'price_change_24h_pct': float
        }
        """
```

**Step 2**: Integrate into V7 theory analysis (30 min)
```python
# File: libs/theories/market_context.py (NEW)
class MarketContextTheory:
    """7th Theory: CoinGecko market context"""

    def analyze(self, symbol: str, candles: pd.DataFrame) -> dict:
        """
        Add macro market context to signal generation
        """
```

**Step 3**: Update V7 runtime to use CoinGecko (30 min)
```python
# File: apps/runtime/v7_runtime.py
# Add CoinGecko data to theory analysis
coingecko_data = self.coingecko_client.get_market_data(symbol)
theories['market_context'] = MarketContextTheory().analyze(symbol, candles, coingecko_data)
```

**Step 4**: Test integration (30 min)
```bash
# Test CoinGecko API connectivity
python -c "from libs.data.coingecko_client import CoinGeckoClient; c = CoinGeckoClient(); print(c.get_market_data('BTC-USD'))"

# Restart V7 runtime with CoinGecko
kill 2085754
nohup .venv/bin/python3 apps/runtime/v7_runtime.py --iterations -1 --sleep-seconds 120 --aggressive --max-signals-per-hour 30 > /tmp/v7_coingecko.log 2>&1 &
```

---

### Phase 2: Documentation Cleanup (TODAY - 1 hour)

**Goal**: Reduce 266 docs to ~10 essential files

**Keep Only** (10 files):
1. `README.md` - Project overview
2. `CLAUDE.md` - Instructions for Claude (MASTER FILE)
3. `PROJECT_MEMORY.md` - Session continuity
4. `V7_RUNTIME_GUIDE.md` - V7 operation manual
5. `DEPLOYMENT.md` - Deployment steps
6. `TROUBLESHOOTING.md` - Common issues
7. `API_REFERENCE.md` - API endpoints
8. `CHANGELOG.md` - Version history
9. `TODO.md` - Active tasks (SINGLE SOURCE OF TRUTH)
10. `ARCHITECTURE.md` - System design

**Delete** (archive to `.archive/` folder):
- All `COINGECKO_*` files (outdated, CoinGecko now integrated)
- All `AMAZON_Q_*` files (not using Amazon Q anymore)
- All `COLAB_*` files (not using Colab)
- All `BUILDER_*` / `QC_*` files (merge into CLAUDE.md)
- All `V5_*` / `V6_*` files (archived versions)
- All `SESSION_*` / `HANDOFF_*` files (temporary)
- All `CRITICAL_*` / `URGENT_*` files (resolve and delete)
- All `STEP_*` files (merge into TODO.md)

**Command**:
```bash
mkdir -p .archive/old_docs
mv /root/crpbot/COINGECKO_*.md .archive/old_docs/
mv /root/crpbot/AMAZON_Q_*.md .archive/old_docs/
mv /root/crpbot/COLAB_*.md .archive/old_docs/
mv /root/crpbot/BUILDER_*.md .archive/old_docs/
mv /root/crpbot/V5_*.md .archive/old_docs/
mv /root/crpbot/V6_*.md .archive/old_docs/
mv /root/crpbot/SESSION_*.md .archive/old_docs/
mv /root/crpbot/CRITICAL_*.md .archive/old_docs/
mv /root/crpbot/STEP_*.md .archive/old_docs/

# Keep git history
git add .archive/
git commit -m "docs: archive 150+ obsolete documentation files"
```

---

### Phase 3: Create Single Source of Truth (30 min)

**File**: `TODO.md` (MASTER TASK LIST)
```markdown
# V7 Trading System - Active Tasks

**Last Updated**: 2025-11-19 14:45

## IN PROGRESS (Now)
- [ ] CoinGecko integration (2 hours)

## NEXT UP (This Week)
- [ ] Dashboard DeepSeek box showing live
- [ ] Telegram notifications for BUY/SELL
- [ ] Backtest V7 on historical data

## DONE (Completed)
- [x] V7 runtime with 6 theories
- [x] DeepSeek LLM integration
- [x] Dashboard with live prices
- [x] Budget increase to $5/day

## BLOCKED (Waiting)
- None

## BACKLOG (Future)
- [ ] Bayesian learning from trade outcomes
- [ ] Multi-timeframe analysis
- [ ] Paper trading mode
```

---

## 🎯 EXECUTION TIMELINE

### Today (Nov 19, 2025)

**14:45-15:15** (30 min): Create CoinGecko client
**15:15-15:45** (30 min): Integrate into theory analysis
**15:45-16:15** (30 min): Update V7 runtime
**16:15-16:45** (30 min): Test CoinGecko integration
**16:45-17:45** (60 min): Documentation cleanup

**17:45**: ✅ CoinGecko integrated, docs cleaned, TODO.md created

---

## 🔍 WHY THIS HAPPENED

### Documentation Sprawl
- Every session created new docs
- Never cleaned up old docs
- Multiple "CRITICAL" / "URGENT" files
- No single master plan

### No Task Tracking
- Steps planned but not tracked
- No checklist for "what's done vs. what's missing"
- Easy to lose track in 266 files

### Solution
1. **One master TODO.md** - Single source of truth
2. **Archive old docs** - Keep history but reduce clutter
3. **Update CLAUDE.md** - All instructions in one place
4. **Weekly cleanup** - Delete temp files every Friday

---

## 📊 SUCCESS CRITERIA

**CoinGecko Integration**:
- [ ] CoinGecko client created
- [ ] Market context theory implemented
- [ ] V7 runtime using CoinGecko data
- [ ] Dashboard showing CoinGecko metrics
- [ ] DeepSeek analysis includes market cap/volume

**Documentation Cleanup**:
- [ ] 266 files → 10 essential files
- [ ] TODO.md created as master task list
- [ ] .archive/ folder with old docs
- [ ] CLAUDE.md updated with clear instructions
- [ ] No more "which doc is correct?" confusion

**Process Fix**:
- [ ] Weekly doc cleanup scheduled
- [ ] TODO.md updated after every task
- [ ] No new docs without archiving old ones
- [ ] Single source of truth maintained

---

## 🚀 NEXT STEPS (START NOW)

1. **Create TODO list in TodoWrite tool** ✅
2. **Implement CoinGecko client** (30 min)
3. **Integrate into V7** (30 min)
4. **Test** (30 min)
5. **Clean up docs** (60 min)
6. **Update CLAUDE.md** (15 min)

**Total Time**: 2.5 hours
**Completion**: Tonight (Nov 19, 2025)

---

**This is a critical fix. We can't keep missing important steps because of documentation chaos.**
