# Learned From Previous Bugs

> *"There's something in between the present state of the software and the perfectly working version - it comes in form of bugs in software, in form of problems in life. It's the dark side of the moon that we never see, the negative energy blocking our success. The better we learn to handle it, the faster we reach success."*

---

## The Philosophy of Bugs

Every bug is a teacher. Every crash is a lesson. This document captures the wisdom gained from debugging HYDRA, so we never repeat the same mistakes.

**The Pattern**: Most bugs aren't random - they follow patterns. Once you see the pattern, you can prevent entire categories of bugs before they happen.

---

## Bug Pattern #1: Interface Mismatch

### The Lesson
**When two components talk to each other, they MUST speak the same language.**

### Real Examples from HYDRA

| Bug | Caller Expected | Callee Had | Result |
|-----|-----------------|------------|--------|
| Guardian | `check_before_trade()` | `validate_trade()` | Runtime crash |
| RegimeDetector | Returns `Tuple[str, Dict]` | Returns `Dict` | Unpacking error |
| Anti-Manip | `check_all_layers()` | `run_all_filters()` | AttributeError |
| Guardian Params | 5 parameters | 9 parameters | TypeError |

### Prevention Strategy
```python
# ALWAYS define interfaces in one place
# libs/hydra/interfaces.py

from typing import Protocol, Dict

class IGuardian(Protocol):
    def validate_trade(
        self,
        asset: str,
        direction: str,
        position_size: float
    ) -> Dict:
        """Validate a trade before execution.

        Returns:
            Dict with keys: 'approved', 'reason', 'adjusted_size'
        """
        ...

# Then in guardian.py:
class Guardian(IGuardian):
    def validate_trade(self, asset, direction, position_size) -> Dict:
        # Implementation matches interface
```

### Checklist Before Coding
- [ ] Read the method signature in the source file
- [ ] Match parameter names exactly
- [ ] Match return type exactly
- [ ] Write a quick test: `guardian.validate_trade("BTC", "BUY", 100)`

---

## Bug Pattern #2: Terminology Inconsistency

### The Lesson
**Pick one term and use it everywhere. BUY or LONG - not both.**

### Real Examples from HYDRA

| Location | Used | Should Be | Bug Type |
|----------|------|-----------|----------|
| paper_trader.py | "LONG"/"SHORT" | "BUY"/"SELL" | Direction mismatch |
| gladiator prompts | "LONG" in examples | "BUY" in code | LLM confusion |
| database comments | "LONG, SHORT" | "BUY, SELL" | Documentation lie |
| cross_asset_filter | checks "LONG" | receives "BUY" | Filter broken |

### Prevention Strategy
```python
# Define constants in ONE place
# libs/hydra/constants.py

class Direction:
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

# EVERYWHERE else, import and use:
from libs.hydra.constants import Direction

if signal["action"] == Direction.BUY:
    # ...
```

### The Rule
**GREP before you code:**
```bash
# Before adding any directional logic:
grep -r "LONG\|SHORT\|BUY\|SELL" libs/hydra/ | head -20
# See what's already being used, match it
```

---

## Bug Pattern #3: Division by Zero

### The Lesson
**Any division can fail. Assume the denominator might be zero.**

### Real Examples from HYDRA

| File | Line | Code | Fix |
|------|------|------|-----|
| guardian.py | 122 | `pnl / initial_balance` | `if initial_balance > 0` |
| guardian.py | 132 | `loss / balance` | `if balance > 0` |
| guardian.py | 184 | `elapsed.seconds / 60` | `elapsed.total_seconds() / 60` |
| paper_trader.py | 300 | `wins / total` | `if total > 0 else 0` |

### Prevention Strategy
```python
# ALWAYS use safe division
def safe_divide(a, b, default=0.0):
    """Divide a by b, return default if b is zero."""
    return a / b if b != 0 else default

# Usage:
win_rate = safe_divide(wins, total_trades, default=0.0)
pnl_percent = safe_divide(profit, initial_balance, default=0.0)
```

### The Rule
**Every time you type `/`, stop and think: "Can this ever be zero?"**

---

## Bug Pattern #4: None Propagation

### The Lesson
**None spreads like a virus. Catch it at the source.**

### Real Examples

| Symptom | Cause | Fix |
|---------|-------|-----|
| `'NoneType' has no attribute 'get'` | Function returned None | Check return value |
| `TypeError: unsupported operand` | None in calculation | Default value |
| `KeyError` | Dict access on None | Use `.get()` |

### Prevention Strategy
```python
# BAD: Assumes function always returns valid data
stats = paper_trader.get_stats()
win_rate = stats["win_rate"]  # CRASH if stats is None

# GOOD: Defensive access
stats = paper_trader.get_stats() or {}
win_rate = stats.get("win_rate", 0.0)

# BETTER: Early return pattern
def _update_metrics(self):
    stats = self.paper_trader.get_stats()
    if not stats:
        logger.warning("No stats available")
        return  # Early exit, don't crash

    # Safe to use stats now
    win_rate = stats["win_rate"]
```

### The Rule
**Treat every external data source as potentially None.**

---

## Bug Pattern #5: Schema Drift

### The Lesson
**When you change data structure, find ALL consumers.**

### Real Examples from HYDRA

| Change Made | Forgotten Consumer | Result |
|-------------|-------------------|--------|
| Changed trade format | Dashboard expected old format | Dashboard shows 0s |
| Renamed runtime file | Dashboard checks old filename | "Not Running" |
| Moved data storage | Dashboard reads old location | No data found |

### Prevention Strategy
```bash
# Before changing ANY data structure:

# 1. Find all files that use this structure
grep -r "paper_trades" --include="*.py" .
grep -r "get_stats" --include="*.py" .

# 2. List all consumers
# - hydra_runtime.py (producer)
# - dashboard.py (consumer)  <-- MUST UPDATE
# - tests/test_paper_trader.py (consumer) <-- MUST UPDATE

# 3. Update ALL consumers in same commit
```

### The Rule
**Producer and consumer changes must happen together.**

---

## Bug Pattern #6: F-String Syntax Errors

### The Lesson
**F-strings with conditionals are traps. Parenthesize everything.**

### Real Examples from HYDRA

```python
# BUG - Gladiator D (4 locations)
f"{value:.1%} if condition else 0}"  # SyntaxError

# FIX
f"{(value if condition else 0):.1%}"  # Works
```

### Prevention Strategy
```python
# Rule: Complex expressions go OUTSIDE the f-string

# BAD
f"Win rate: {wins/total if total > 0 else 0:.1%}"

# GOOD
win_rate = wins / total if total > 0 else 0
f"Win rate: {win_rate:.1%}"
```

### The Rule
**If your f-string has `if`, extract to a variable first.**

---

## Bug Pattern #7: Process Detection Brittleness

### The Lesson
**Process names change. Use robust detection.**

### Real Examples

| Detection Code | Failed When | Fix |
|----------------|-------------|-----|
| `'hydra_runtime.py' in ps` | Renamed to mother_ai_runtime.py | Check both |
| `'python' in cmdline` | Using `python3` | Check pattern |

### Prevention Strategy
```python
# BAD: Exact match
self.running = 'hydra_runtime.py' in subprocess_output

# GOOD: Pattern match with fallback
PROCESS_PATTERNS = [
    'hydra_runtime.py',
    'mother_ai_runtime.py',
    'hydra_runtime',  # Without .py
]

def is_hydra_running():
    output = subprocess.run(['ps', 'aux'], capture_output=True, text=True).stdout
    return any(pattern in output for pattern in PROCESS_PATTERNS)
```

---

## Bug Pattern #8: Silent Failures

### The Lesson
**When things fail silently, debugging becomes impossible.**

### Real Examples

| Silent Failure | How We Found It | Fix |
|----------------|-----------------|-----|
| Metrics return 0 | Manually checked endpoint | Add logging |
| Guardian not validating | Trade went through wrong | Add debug log |
| Regime detector stuck | Performance degraded | Add state logging |

### Prevention Strategy
```python
# Every function should log its state

def _update_prometheus_metrics(self):
    logger.debug("Starting metrics update")

    try:
        stats = self.paper_trader.get_stats()
        logger.debug(f"Got stats: {stats}")

        if not stats:
            logger.warning("No stats available - using defaults")
            return

        HydraMetrics.set_pnl(stats['pnl'])
        logger.debug(f"Set P&L metric: {stats['pnl']}")

    except Exception as e:
        logger.error(f"Metrics update failed: {e}", exc_info=True)
        HydraMetrics.errors_total.inc()
```

### The Rule
**If something can fail, log that it tried and what happened.**

---

## Bug Pattern #9: Time and Timezone

### The Lesson
**Time is complicated. Treat it with respect.**

### Real Examples

| Bug | Cause | Fix |
|-----|-------|-----|
| Signal rejected as expired | Timezone mismatch | Use UTC everywhere |
| Invalid timestamp display | Mixed aware/naive | Normalize to aware |
| Duplicate signals | Clock drift | Use server time |

### Prevention Strategy
```python
from datetime import datetime, timezone

# ALWAYS use timezone-aware UTC
def now() -> datetime:
    return datetime.now(timezone.utc)

# NEVER use naive datetime
# BAD
datetime.now()  # Naive, ambiguous

# GOOD
datetime.now(timezone.utc)  # Explicit UTC
```

---

## Bug Pattern #10: Import Cycles

### The Lesson
**Circular imports are architecture problems in disguise.**

### Prevention Strategy
```
# Good architecture: Dependency flows one way
constants.py  ← interfaces.py  ← implementations.py  ← runtime.py

# Bad architecture: Cycles
module_a imports module_b
module_b imports module_a  ← CYCLE!
```

---

## The Bug Prevention Checklist

Before merging any code, verify:

### Interface Safety
- [ ] Method names match between caller and callee
- [ ] Parameter counts and types match
- [ ] Return types match what caller expects

### Data Safety
- [ ] All divisions have zero checks
- [ ] All external data checked for None
- [ ] All dict access uses `.get()` with defaults

### Consistency
- [ ] Constants used instead of strings ("BUY" not "LONG")
- [ ] Time handling uses timezone-aware UTC
- [ ] Process detection handles renamed files

### Observability
- [ ] Debug logging added for key operations
- [ ] Error handling logs the exception
- [ ] Metrics capture failure counts

### Documentation
- [ ] Interface changes documented
- [ ] Schema changes noted
- [ ] All consumers updated

---

## The Meta-Lesson

> **Bugs cluster around boundaries.**

Most bugs happen at the edges:
- Between components (interface mismatch)
- Between data formats (schema drift)
- Between time zones (timezone bugs)
- Between expectations and reality (None propagation)

**Focus your defensive coding at boundaries.**

---

## Quick Reference Card

```
┌──────────────────────────────────────────────────────────────┐
│                    BUG PREVENTION QUICK REF                   │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  BEFORE CALLING A FUNCTION:                                  │
│  → Read its signature in source                              │
│  → Match parameters exactly                                  │
│  → Handle potential None return                              │
│                                                              │
│  BEFORE DIVIDING:                                            │
│  → Check denominator != 0                                    │
│  → Use safe_divide() helper                                  │
│                                                              │
│  BEFORE USING A STRING CONSTANT:                             │
│  → Import from constants.py                                  │
│  → Never type "BUY" or "SELL" directly                       │
│                                                              │
│  BEFORE CHANGING DATA SCHEMA:                                │
│  → grep for all consumers                                    │
│  → Update ALL in same commit                                 │
│                                                              │
│  BEFORE MERGING:                                             │
│  → Run existing tests                                        │
│  → Add test for new code                                     │
│  → Check logs for errors                                     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Bugs That Taught Us

| Bug ID | Date | Lesson | Files Touched |
|--------|------|--------|---------------|
| #1 | 2025-11-29 | Method names must match | guardian.py |
| #2 | 2025-11-29 | Return types must match | regime_detector.py |
| #3 | 2025-11-29 | Parameters must match | hydra_runtime.py |
| #4-5 | 2025-11-29 | F-strings need parens | gladiator_d.py |
| #6 | 2025-11-29 | Check for division by zero | guardian.py |
| #7 | 2025-11-29 | Standardize terminology | paper_trader.py |
| #47 | 2025-11-29 | vote vs direction key | tournament_tracker.py |
| Dashboard | 2025-12-01 | Schema drift breaks consumers | dashboard.py |
| Timezone | 2025-11-22 | UTC everywhere | signal_history.py |

---

## The Growth Mindset

Every bug fixed is:
- **Knowledge gained** - We understand the system better
- **Pattern recognized** - We can prevent similar bugs
- **Code improved** - The fix often makes code cleaner
- **Test added** - Future regressions prevented

**Embrace bugs as teachers, not enemies.**

---

*Last Updated: 2025-12-03*
*Bugs Documented: 21+*
*Lessons Learned: 10 major patterns*
