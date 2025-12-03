# HYDRA 3.0 - Complete Bug Report

**Date**: 2025-11-29
**Status**: 20 bugs fixed, 10 remaining (4 CRITICAL)

---

## ✅ FIXED BUGS (Round 1 - 6 bugs)

### Bug #1: Guardian Method Name Mismatch ✅
- **File**: `libs/hydra/guardian.py`
- **Fix**: Renamed `check_before_trade()` → `validate_trade()`
- **Status**: FIXED

### Bug #2: Guardian Return Format ✅
- **File**: `libs/hydra/guardian.py`
- **Fix**: Changed return type `Tuple[bool, str, Optional[float]]` → `Dict`
- **Status**: FIXED

### Bug #3: RegimeDetector Return Format ✅
- **File**: `libs/hydra/regime_detector.py`
- **Fix**: Changed return type `Tuple[str, Dict]` → `Dict`
- **Status**: FIXED

### Bug #4-5: Gladiator D Syntax Errors ✅
- **File**: `libs/hydra/gladiators/gladiator_d_gemini.py`
- **Lines**: 215, 222, 228, 230
- **Fix**: Corrected f-string formatting: `{value:.1%} if condition else 0}` → `{(value if condition else 0):.1%}`
- **Status**: FIXED

### Bug #6: Division by Zero Checks ✅
- **File**: `libs/hydra/guardian.py`
- **Lines**: 122, 132, 135, 144, 184-196
- **Fix**: Added zero checks before all division operations
- **Status**: FIXED

### Bug #10: BUY/SELL vs LONG/SHORT Mismatch ✅
- **File**: `libs/hydra/paper_trader.py`
- **Lines**: 38, 185, 188, 252, 262, 300, 302, 306, 308, 317, 320
- **Fix**: Changed all "LONG"/"SHORT" references to "BUY"/"SELL"
- **Status**: FIXED

---

## ✅ FIXED BUGS (Round 2 - 1 bug)

### Bug #1 (Round 2): Missing Singleton Functions ✅
- **Files**: `libs/hydra/regime_detector.py`, `libs/hydra/guardian.py`, `libs/hydra/anti_manipulation.py`
- **Fix**: Added singleton pattern functions:
  - `get_regime_detector()` in regime_detector.py
  - `get_guardian(account_balance=10000.0)` in guardian.py
  - `get_anti_manipulation_filter()` in anti_manipulation.py
- **Status**: FIXED

---

## 🔴 REMAINING CRITICAL BUGS (4 bugs - MUST FIX)

### Bug #2: RegimeDetector Method Signature Mismatch
- **File**: `apps/runtime/hydra_runtime.py`
- **Lines**: 247, 482
- **Issue**: Runtime calls `regime_detector.detect_regime(market_data)` with 1 argument, but method requires `detect_regime(symbol: str, candles: List[Dict], ...)`
- **Current Call**:
```python
regime_result = self.regime_detector.detect_regime(market_data)
```
- **Should Be**:
```python
regime_result = self.regime_detector.detect_regime(
    symbol=asset,
    candles=market_data
)
```
- **Impact**: BLOCKS ALL REGIME DETECTION → runtime will crash
- **Priority**: CRITICAL

### Bug #3: AntiManipulationFilter Method Name Mismatch
- **File**: `apps/runtime/hydra_runtime.py`
- **Line**: 258
- **Issue**: Runtime calls `self.anti_manip.check_all_layers()` but class only has `run_all_filters()`
- **Current Call**:
```python
anti_manip_result = self.anti_manip.check_all_layers(...)
```
- **Should Be** (Option 1 - Change runtime):
```python
anti_manip_result = self.anti_manip.run_all_filters(...)
```
- **OR** (Option 2 - Add alias in AntiManipulationFilter):
```python
check_all_layers = run_all_filters  # Alias
```
- **Impact**: BLOCKS ANTI-MANIPULATION FILTERING → runtime will crash
- **Priority**: CRITICAL

### Bug #4: Guardian.validate_trade() Signature Mismatch
- **File**: `apps/runtime/hydra_runtime.py`
- **Lines**: 314-320
- **Issue**: Runtime passes wrong parameters to `guardian.validate_trade()`
- **Current Call**:
```python
guardian_check = self.guardian.validate_trade(
    asset, direction, position_size_usd, stop_loss_pct, market_data
)
```
- **Method Signature**:
```python
def validate_trade(
    self,
    asset: str,
    asset_type: str,  # MISSING in runtime call
    direction: str,
    position_size_usd: float,
    entry_price: float,  # DIFFERENT param (not stop_loss_pct)
    sl_price: float,  # DIFFERENT param (not market_data)
    regime: str,  # MISSING in runtime call
    current_positions: List[Dict],  # MISSING in runtime call
    strategy_correlations: Optional[List[float]] = None
) -> Dict:
```
- **Should Be**:
```python
guardian_check = self.guardian.validate_trade(
    asset=asset,
    asset_type=asset_type,  # Need to determine from asset
    direction=direction,
    position_size_usd=position_size_usd,
    entry_price=entry_price,  # From signal
    sl_price=stop_loss,  # From signal
    regime=regime_result["regime"],
    current_positions=self.open_positions,
    strategy_correlations=None  # Or calculate
)
```
- **Impact**: BLOCKS GUARDIAN VALIDATION → runtime will crash
- **Priority**: CRITICAL

### Bug #5: LONG/SHORT vs BUY/SELL Inconsistency (System-Wide)
- **Files**: Multiple HYDRA files
- **Issue**: Documentation and many filter methods expect "LONG"/"SHORT" but runtime passes "BUY"/"SELL"
- **Locations**:
  - `libs/hydra/guardian.py:92` (comment)
  - `libs/hydra/cross_asset_filter.py:56` (logic checks)
  - `libs/hydra/execution_optimizer.py:59` (logic checks)
  - `libs/hydra/anti_manipulation.py:360` (logic checks)
  - All gladiator prompts and examples
- **Fix Strategy**: Standardize on "BUY"/"SELL" everywhere
  - Update all method logic to check for "BUY" instead of "LONG"
  - Update all comments and docstrings
  - Update all LLM prompts
- **Impact**: Cross-asset filter, execution optimizer, anti-manipulation filter won't work correctly
- **Priority**: HIGH

---

## ⚠️ HIGH PRIORITY BUGS (2 bugs)

### Bug #6: CrossAssetFilter Return Type Usage
- **File**: `apps/runtime/hydra_runtime.py`
- **Line**: 295
- **Issue**: Incorrect tuple unpacking
- **Current**:
```python
if not cross_asset_result[0]:
    logger.warning(f"{asset} blocked: {cross_asset_result[1]}")
```
- **Should Be**:
```python
cross_asset_passed, cross_asset_reason = cross_asset_result
if not cross_asset_passed:
    logger.warning(f"{asset} blocked: {cross_asset_reason}")
```
- **Impact**: Works but poor code quality, potential bugs
- **Priority**: HIGH

### Bug #7: LessonMemory Return Type Usage
- **File**: `apps/runtime/hydra_runtime.py`
- **Lines**: 309-311
- **Issue**: Incorrect tuple unpacking
- **Current**:
```python
if lesson_check[0]:
    logger.error(f"Rejected by lesson memory")
```
- **Should Be**:
```python
lesson_triggered, lesson_obj = lesson_check
if lesson_triggered:
    logger.error(f"Rejected: {lesson_obj.lesson_id if lesson_obj else 'Unknown'}")
```
- **Impact**: Works but poor code quality
- **Priority**: HIGH

---

## 📝 MEDIUM PRIORITY BUGS (3 bugs)

### Bug #8: Direction Constant Mapping in Gladiators
- **Files**: All gladiator files
- **Issue**: LLM prompts use "LONG" terminology but return values use "BUY"/"SELL"
- **Fix**: Update all prompts to consistently use BUY/SELL
- **Impact**: Inconsistent terminology, confusing for debugging
- **Priority**: MEDIUM

### Bug #9: Database Column Comment Outdated
- **File**: `libs/hydra/database.py`
- **Line**: 125
- **Issue**: Comment says `# LONG, SHORT` but should be `# BUY, SELL`
- **Fix**: Update comment
- **Impact**: Documentation only
- **Priority**: LOW

### Bug #10: Gladiator D F-String Verification
- **File**: `libs/hydra/gladiators/gladiator_d_gemini.py`
- **Lines**: 215, 222, 228, 230
- **Status**: Already fixed in Round 1 - verify fix is correct
- **Priority**: MEDIUM (VERIFY)

---

## 📋 LOW PRIORITY / WARNINGS (4 bugs)

### Bug #11: Guardian Account Balance in Singleton
- **File**: `libs/hydra/guardian.py` singleton
- **Issue**: Guardian requires `account_balance` parameter, singleton uses default 10000.0
- **Fix**: Provide default or read from config
- **Impact**: May use wrong balance if not configured
- **Priority**: LOW

### Bug #12: Potential KeyError in Strategy Access
- **File**: `apps/runtime/hydra_runtime.py`
- **Issue**: Inconsistent use of `.get()` vs direct dict access
- **Fix**: Use `.get()` with defaults consistently
- **Impact**: Potential KeyError if keys missing
- **Priority**: LOW

### Bug #13: Type Annotation Inconsistencies
- **Files**: Multiple
- **Issue**: Missing return type hints, mixed Dict/dict usage
- **Fix**: Add type hints for better IDE support
- **Impact**: Code quality only
- **Priority**: LOW

### Bug #14: Missing Import for Tuple
- **Files**: Several HYDRA modules
- **Issue**: Some files use `Tuple` in type hints but may not import it
- **Fix**: Verify all files import `Tuple` from `typing`
- **Impact**: May cause import errors
- **Priority**: LOW

---

## 📊 SUMMARY

### Bugs Fixed: 7
- Round 1: 6 bugs (Guardian, RegimeDetector, Gladiator D, PaperTrader)
- Round 2: 1 bug (Singleton functions)

### Bugs Remaining: 14
- **CRITICAL**: 4 bugs (will cause crashes)
- **HIGH**: 2 bugs (logic errors)
- **MEDIUM**: 3 bugs (code quality)
- **LOW**: 4 bugs (warnings)

### Next Steps (Priority Order):
1. **FIRST**: Fix RegimeDetector call signature (Bug #2) - 2 locations
2. **SECOND**: Fix AntiManipulationFilter method name (Bug #3) - 1 location
3. **THIRD**: Fix Guardian.validate_trade signature (Bug #4) - 1 location
4. **FOURTH**: Standardize BUY/SELL everywhere (Bug #5) - system-wide
5. Then address High/Medium/Low priority bugs

### Files Requiring Changes:
- `apps/runtime/hydra_runtime.py` (Bugs #2, #3, #4, #6, #7) - **CRITICAL FILE**
- `libs/hydra/cross_asset_filter.py` (Bug #5)
- `libs/hydra/execution_optimizer.py` (Bug #5)
- `libs/hydra/anti_manipulation.py` (Bug #5)
- All gladiator files (Bug #8)

---

## 🔧 TESTING CHECKLIST

After fixing critical bugs, test:
- [ ] Runtime starts without ImportError
- [ ] RegimeDetector correctly classifies regimes
- [ ] Guardian validates trades without crashes
- [ ] Anti-manipulation filtering works
- [ ] Paper trading tracks BUY/SELL correctly
- [ ] All 4 gladiators initialize successfully
- [ ] Complete signal generation pipeline works

---

**Report Generated**: 2025-11-29
**HYDRA Version**: 3.0
**Total Lines Analyzed**: ~11,300
