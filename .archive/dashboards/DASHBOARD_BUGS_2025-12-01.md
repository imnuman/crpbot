# Dashboard Bugs Report - 2025-12-01

**Status**: Dashboard running but showing no data
**URL**: http://178.156.136.185:3000
**Process**: Running (PIDs: 3302799, 3302828, 3302829)

---

## 🐛 Critical Bugs Identified

### Bug #1: Process Name Mismatch
**File**: `apps/dashboard_reflex/dashboard_reflex/dashboard_reflex.py:93`

**Issue**:
```python
result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
self.hydra_running = 'hydra_runtime.py' in result.stdout
```

**Problem**: Dashboard looking for `hydra_runtime.py` but actual process is `mother_ai_runtime.py`

**Impact**: Dashboard incorrectly shows HYDRA as not running

**Fix**:
```python
# Check for both old HYDRA and new Mother AI
self.hydra_running = ('hydra_runtime.py' in result.stdout or
                     'mother_ai_runtime.py' in result.stdout)
```

---

### Bug #2: Data Source Incompatibility
**File**: `apps/dashboard_reflex/dashboard_reflex/dashboard_reflex.py:48-96`

**Issue**: Dashboard expects data from:
- `/root/crpbot/data/hydra/paper_trades.jsonl` (667 lines, old HYDRA format)
- `/root/crpbot/data/hydra/hydra.db` (SQLite, old HYDRA schema)

**Problem**: Mother AI 3.0 uses completely different data structure:
- No `paper_trades.jsonl` for Mother AI (that's old HYDRA)
- No gladiator tracking in paper_trades
- Mother AI stores portfolio data in `GladiatorPortfolio` objects (in-memory)

**Current Data Available**:
```bash
/root/crpbot/data/hydra/paper_trades.jsonl  # 667 trades (OLD HYDRA, not Mother AI)
/root/crpbot/data/hydra/hydra.db            # Old HYDRA database
/root/crpbot/data/hydra/tournament_votes.jsonl  # 2.1 MB of old voting data
```

**Mother AI Data** (not persisted yet):
- Portfolio data: In-memory only (`GladiatorPortfolio` objects)
- Tournament rankings: In-memory (`TournamentManager`)
- Trade history: Not being saved to disk

**Impact**: Dashboard shows old HYDRA data, not current Mother AI 3.0 data

---

### Bug #3: Schema Mismatch - Trade Fields
**File**: `apps/dashboard_reflex/dashboard_reflex/dashboard_reflex.py:82-86`

**Issue**: Dashboard expects trade fields that don't exist:
```python
self.gladiator_a_strategies = sum(1 for t in trades if t.get('gladiator') == 'A')
self.gladiator_b_approvals = sum(1 for t in trades if t.get('gladiator') == 'B')
```

**Problem**: Old HYDRA `paper_trades.jsonl` doesn't have `gladiator` field in that format

**Actual Schema** (from old HYDRA):
```json
{
  "asset": "BTC-USD",
  "direction": "LONG",
  "entry_price": 96000,
  "status": "OPEN",
  "pnl_percent": 0.0,
  "timestamp": "2025-12-01T18:00:00Z"
}
```

**Impact**: Gladiator counters always show 0

---

### Bug #4: Chat Interface Incompatibility
**File**: `apps/dashboard_reflex/dashboard_reflex/chat_page.py:14-21`

**Issue**: Chat trying to import and instantiate old HYDRA gladiators:
```python
from hydra.chat_interface import HydraChat
from hydra.gladiators.gladiator_a_deepseek import GladiatorA_DeepSeek
from hydra.gladiators.gladiator_b_claude import GladiatorB_Claude
from hydra.gladiators.gladiator_c_grok import GladiatorC_Grok
from hydra.gladiators.gladiator_d_gemini import GladiatorD_Gemini
```

**Problem**:
1. Old HYDRA gladiators have different class names and signatures
2. Mother AI 3.0 gladiators don't support chat interface
3. Chat interface expects voting/consensus system (removed in Mother AI)

**Mother AI 3.0 Reality**:
- Gladiators: `BaseGladiator` subclasses with tournament-focused methods
- No chat interface: `make_trade_decision()` only
- No voting system: Independent trading decisions

**Impact**: Chat page will crash on load

---

### Bug #5: Missing Mother AI Data Persistence
**File**: `libs/hydra/mother_ai.py` and `libs/hydra/gladiator_portfolio.py`

**Issue**: Mother AI 3.0 does NOT save any data to disk:
- Portfolio data stored in-memory only
- No database writes
- No JSONL output
- All data lost on restart

**Problem**: Dashboard cannot access Mother AI data because it doesn't exist on disk

**Impact**: Dashboard permanently out of sync with Mother AI

---

## 📊 Data Flow Analysis

### Old HYDRA (what dashboard expects):
```
HYDRA Runtime
  → Votes to paper_trades.jsonl
  → Tournament to hydra.db
  → Dashboard reads files
  → Display stats
```

### Mother AI 3.0 (actual system):
```
Mother AI Runtime
  → Gladiators trade independently
  → Portfolio data in-memory
  → Tournament rankings in-memory
  → NO PERSISTENCE ❌
  → Dashboard has no data source
```

---

## 🔧 Recommended Fixes

### Option A: Quick Fix (Minimal Changes)
1. Update dashboard to check for `mother_ai_runtime.py` process
2. Add warning: "Mother AI data not persisted - dashboard shows old HYDRA"
3. Disable chat interface (mark as "Coming Soon")
4. Keep showing old HYDRA data for reference

**Pros**: Quick, no major changes
**Cons**: Dashboard still not showing Mother AI data

---

### Option B: Add Data Persistence (Recommended)
1. **Add Mother AI logging** to `mother_ai.py`:
   ```python
   def _save_tournament_state(self):
       """Save current tournament state to disk"""
       state = {
           "timestamp": datetime.now(timezone.utc).isoformat(),
           "gladiators": {
               "A": self.tournament_manager.portfolios["A"].get_stats().__dict__,
               "B": self.tournament_manager.portfolios["B"].get_stats().__dict__,
               "C": self.tournament_manager.portfolios["C"].get_stats().__dict__,
               "D": self.tournament_manager.portfolios["D"].get_stats().__dict__,
           },
           "rankings": self.tournament_manager.rankings
       }
       with open("/root/crpbot/data/hydra/mother_ai_state.json", "w") as f:
           json.dump(state, f, indent=2)
   ```

2. **Update dashboard** to read `mother_ai_state.json`:
   ```python
   state_file = Path("/root/crpbot/data/hydra/mother_ai_state.json")
   if state_file.exists():
       with open(state_file) as f:
           state = json.load(f)
       # Update gladiator stats from state
   ```

3. **Update process check**:
   ```python
   self.hydra_running = 'mother_ai_runtime.py' in result.stdout
   ```

**Pros**: Dashboard shows real Mother AI data
**Cons**: Requires changes to both Mother AI and dashboard

---

### Option C: Real-time API (Future)
1. Add Flask/FastAPI endpoint to Mother AI runtime
2. Expose live portfolio data via REST API
3. Dashboard polls API for real-time updates
4. Add WebSocket for live updates

**Pros**: Real-time data, no file I/O
**Cons**: Most complex, requires significant development

---

## 🎯 Immediate Action Items

### Priority 1: Fix Process Detection
**File**: `apps/dashboard_reflex/dashboard_reflex/dashboard_reflex.py:93`
**Change**:
```python
self.hydra_running = ('mother_ai_runtime.py' in result.stdout or
                     'hydra_runtime.py' in result.stdout)
```

### Priority 2: Add Warning Banner
Add warning to dashboard homepage:
```
⚠️ Dashboard currently showing old HYDRA data.
Mother AI 3.0 data persistence coming soon.
```

### Priority 3: Disable Chat Interface
**File**: `apps/dashboard_reflex/dashboard_reflex/dashboard_reflex.py:165-167`
**Change**:
```python
rx.link(
    rx.button("Chat (Coming Soon)", variant="soft", color_scheme="gray", disabled=True),
    href="#",
),
```

### Priority 4: Add Mother AI Data Persistence
**File**: `libs/hydra/mother_ai.py`
Add method to save state after each cycle and call it in `run_trading_cycle()`

---

## 📝 Testing Checklist

- [ ] Dashboard loads without errors
- [ ] Shows "HYDRA running" badge when Mother AI active
- [ ] Displays gladiator stats (even if old data)
- [ ] Performance metrics calculate correctly
- [ ] Recent trades table renders
- [ ] Chat page disabled or shows "Coming Soon"
- [ ] No console errors in browser
- [ ] Refresh button works

---

## 🔍 Current Dashboard Status

**URL**: http://178.156.136.185:3000
**Process**: ✅ Running
**Data Loading**: ❌ Broken (looking for wrong process)
**Stats Display**: ❌ Empty (no Mother AI data)
**Chat Interface**: ❌ Will crash (incompatible with Mother AI)

**Gladiator Counts** (from old HYDRA data):
- Gladiator A: 0 actions
- Gladiator B: 0 actions
- Gladiator C: 0 actions
- Gladiator D: 0 actions

**Paper Trading Stats** (from old HYDRA):
- Total: 667 trades
- Open: Unknown (field parsing issue)
- Closed: Unknown
- Win Rate: 0% (calculation failing)

---

## 🚀 Next Steps

1. **Immediate**: Fix process detection (5 min)
2. **Short-term**: Add Mother AI data persistence (30 min)
3. **Medium-term**: Update dashboard to read Mother AI state (1 hour)
4. **Long-term**: Build real-time API for Mother AI (4-6 hours)

---

**Last Updated**: 2025-12-01
**Reporter**: Builder Claude
**Severity**: Medium (dashboard non-functional but not blocking Mother AI)
**Priority**: Medium (nice-to-have but not critical)
