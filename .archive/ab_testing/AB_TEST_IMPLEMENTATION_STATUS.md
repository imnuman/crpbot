# A/B Testing Implementation Status

## Completed ✅

### Step 1: Database Migration
- **File**: `tradingai.db`
- **Change**: Added `strategy` column to `signals` table
- **SQL**: `ALTER TABLE signals ADD COLUMN strategy TEXT DEFAULT "v7_full_math"`
- **Status**: ✅ COMPLETE
- **Verification**: Column exists, default value set

### Step 2: Signal Model Update
- **File**: `libs/db/models.py:63`
- **Change**: Added `strategy` field to Signal class
- **Code**: `strategy = Column(String(50), default="v7_full_math")`
- **Status**: ✅ COMPLETE

### Step 3: Minimal Prompt Builder
- **File**: `libs/llm/signal_synthesizer.py:268-368`
- **Change**: Created `build_minimal_prompt()` method
- **Purpose**: Generate DeepSeek prompts WITHOUT mathematical theories
- **Status**: ✅ COMPLETE
- **Details**: Minimal prompt includes only symbol, price, recent candles - no Shannon, Hurst, etc.

### Step 4: Runtime Modifications
- **Files Modified**:
  - `libs/llm/signal_generator.py:172` - Added `strategy` parameter to generate_signal()
  - `libs/llm/signal_generator.py:229-244` - Strategy-based prompt selection
  - `apps/runtime/v7_runtime.py:549` - Added `strategy` parameter to generate_signal_for_symbol()
  - `apps/runtime/v7_runtime.py:366` - Added `strategy` parameter to _save_signal_to_db()
  - `apps/runtime/v7_runtime.py:427` - Save strategy field to database
  - `apps/runtime/v7_runtime.py:610` - Pass strategy to signal generator
  - `apps/runtime/v7_runtime.py:713-753` - Alternating strategy logic in run_single_scan()
- **Strategy**: Alternates between "v7_full_math" (even iterations) and "v7_deepseek_only" (odd iterations)
- **Status**: ✅ COMPLETE
- **Logging**: Added "🧪 A/B TEST" log messages for visibility

### Critical Bug Fixes (Earlier in Session)
1. **Fixed HOLD Signal Trading Bug** ✅
   - File: `libs/tracking/paper_trader.py:85-88`
   - Issue: Paper trader was trading HOLD signals as LONG positions
   - Fix: Added check to skip HOLD signals from paper trading
   - Impact: 6.9% win rate was due to trading 1,845 HOLD signals as longs

2. **Fixed Dashboard Timestamps** ✅
   - Added entry/exit timestamps to dashboard
   - Created `format_timestamp_est()` utility function
   - Fixed Reflex compilation errors
   - Dashboard: http://178.156.136.185:3000/performance

## Remaining Tasks 📋

### Step 2: Update Signal Model
**File**: `libs/db/models.py`
**Change**: Add `strategy` field to Signal class

```python
# Around line 50-60 in Signal class
strategy: Mapped[Optional[str]] = mapped_column(String, default="v7_full_math")
```

### Step 3: Create Minimal Prompt Builder
**File**: `libs/llm/signal_synthesizer.py`
**New Method**: `build_minimal_prompt()`

Purpose: Create DeepSeek prompts WITHOUT mathematical theories
- Only include: symbol, current price, timeframe, recent candles
- No Shannon, Hurst, Kolmogorov, etc.
- Let DeepSeek use its own knowledge

```python
def build_minimal_prompt(
    self,
    context: MarketContext,
    additional_context: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Build minimal prompt with ONLY price/volume data - NO theories
    For A/B testing to see if math theories actually help
    """
    user_prompt = f"""**Market Context:**
Symbol: {context.symbol}
Current Price: ${context.current_price:,.2f}
Timeframe: {context.timeframe}
Timestamp: {context.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

**Recent Price Action:**
[Last 50 candles - highs, lows, closes]

**Your Task:**
Analyze this cryptocurrency market using your knowledge.
Generate a trading signal: BUY, SELL, or HOLD.

SIGNAL: [BUY/SELL/HOLD]
CONFIDENCE: [0-100]%
ENTRY PRICE: $[number or N/A]
STOP LOSS: $[number or N/A]
TAKE PROFIT: $[number or N/A]
REASONING: [2-3 sentences]
"""

    return [
        {"role": "system", "content": self.SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
```

### Step 4: Modify V7 Runtime
**File**: `apps/runtime/v7_runtime.py`
**Changes**:

1. Add strategy parameter to signal generation
2. Run BOTH strategies in parallel (alternating or simultaneous)
3. Tag each signal with its strategy

```python
# Around line 550-650 in generate_signal_for_symbol()

# Option A: Alternate strategies
strategy = "v7_full_math" if iteration % 2 == 0 else "v7_deepseek_only"

# Option B: Run both simultaneously
signals = []
for strategy in ["v7_full_math", "v7_deepseek_only"]:
    signal = self.generate_signal_with_strategy(symbol, strategy)
    signals.append(signal)

# In save_signal():
signal.strategy = strategy  # Add this line
```

### Step 5: Add Strategy Filtering to Performance Tracker
**File**: `libs/tracking/performance_tracker.py`
**New Methods**:

```python
def get_win_rate(self, days: int = 30, strategy: Optional[str] = None) -> Dict[str, Any]:
    """Get win rate stats, optionally filtered by strategy"""
    # Add WHERE strategy = :strategy if strategy provided

def get_strategy_comparison(self, days: int = 30) -> Dict[str, Dict[str, Any]]:
    """Compare performance between strategies"""
    return {
        "v7_full_math": self.get_win_rate(days, "v7_full_math"),
        "v7_deepseek_only": self.get_win_rate(days, "v7_deepseek_only")
    }
```

### Step 6: Create Comparison Dashboard View
**File**: `apps/dashboard_reflex/dashboard_reflex/dashboard_reflex.py`
**New Page**: `/comparison`

Show side-by-side metrics:
- Win Rate: Full Math vs DeepSeek Only
- Profit Factor
- Average P&L
- Signal Count
- Confidence Distribution

## Testing Strategy

### Phase 1: Baseline (Current - 24-48 hours)
- Run v7_full_math with HOLD fix
- Establish baseline win rate
- Current: HOLD signals now skipped ✅

### Phase 2: A/B Test (Next 7 days)
- Run both strategies in parallel
- Same symbols, same times
- Compare performance

### Phase 3: Analysis
- Determine if math theories help
- Decide which strategy to keep

## Expected Outcomes

### If Math Helps (Expected):
- v7_full_math: 55-65% win rate
- v7_deepseek_only: 45-55% win rate
- **Conclusion**: Keep full math system

### If Math Doesn't Help (Surprising):
- Both strategies: ~50% win rate
- **Conclusion**: Simplify to DeepSeek-only, save computation

## Current System Status

### Running Processes
- V7 Runtime: ✅ Running with HOLD fix
- Dashboard: ✅ Running on port 3000
- Paper Trading: ✅ Active (skipping HOLD signals)

### Key Metrics (Before Fix)
- Total Trades: 757
- Win Rate: 6.9% (HOLD bug)
- Wins: 52
- Losses: 685

### Key Metrics (After Fix - Monitoring)
- Now skipping HOLD signals
- Only trading LONG and SHORT
- Need 24-48 hours of data

## Files Modified This Session

1. `libs/tracking/paper_trader.py` - Skip HOLD signals ✅
2. `libs/utils/timezone.py` - Add format_timestamp_est() ✅
3. `libs/tracking/performance_tracker.py` - Add entry_timestamp to queries ✅
4. `apps/dashboard_reflex/dashboard_reflex/dashboard_reflex.py` - Display timestamps ✅
5. `tradingai.db` - Add strategy column ✅
6. `DEEPSEEK_AB_TEST_PLAN.md` - Planning document ✅
7. `AB_TEST_IMPLEMENTATION_STATUS.md` - This file ✅

## Next Session TODO

1. Complete Steps 2-6 above
2. Test both strategies
3. Monitor results for 7 days
4. Make decision based on data

---
**Last Updated**: 2025-11-21 09:40 EST
**Session**: Timestamp fix + HOLD bug fix + A/B test prep
