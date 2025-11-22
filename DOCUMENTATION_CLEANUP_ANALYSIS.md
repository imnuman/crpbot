# Documentation Cleanup Analysis

**Date**: 2025-11-21
**Analyst**: QC Claude (Local Machine)
**Purpose**: Identify obsolete, redundant, and conflicting documentation

---

## CURRENT STATE

**Total .md files in root**: 30 files
**Claimed "essential" files**: 7 files
**Files archived**: 165 files moved to `.archive/old_docs/`

**Issue**: Still 30 docs in root, not 7. Need to identify what's needed vs redundant.

---

## DOCUMENTATION INVENTORY

### Category 1: ESSENTIAL (Must Keep) - 7 Files ✅

1. **README.md** - Project overview
2. **CLAUDE.md** - Instructions for Claude instances (MASTER)
3. **PROJECT_MEMORY.md** - Session continuity
4. **MASTER_TRAINING_WORKFLOW.md** - Training guide (for future V8/V9)
5. **TODO.md** - Active task list
6. **TROUBLESHOOTING.md** - Common issues and solutions
7. **DATABASE_SCHEMA.md** - Database documentation

**Total**: 7 files ✅

---

### Category 2: V7 REFERENCE (Keep) - 6 Files ✅

8. **V7_COMPLETE_SYSTEM_DIAGRAM.md** - Complete V7 architecture
9. **V7_MASTER_PLAN.md** - V7 implementation plan
10. **V7_MATHEMATICAL_THEORIES.md** - Theory documentation
11. **V7_MONITORING.md** - Production monitoring guide
12. **V7_CLOUD_DEPLOYMENT.md** - Deployment guide
13. **DASHBOARD_QUICK_REF.md** - Dashboard quick reference

**Recommendation**: Keep these 6 for V7 reference

**Total**: 6 files ✅

---

### Category 3: IMPLEMENTATION HISTORY (Archive) - 10 Files ⚠️

These document completed steps - useful for history but not for day-to-day work:

14. **STEP4_COMPLETION_SUMMARY.md** - STEP 4 done
15. **STEP5_COMPLETION_SUMMARY.md** - STEP 5 done
16. **STEP6_PAPER_TRADING_SUMMARY.md** - STEP 6 done
17. **IMPLEMENTATION_STEPS.md** - Step-by-step guide
18. **ADVANCED_THEORIES_COMPLETE.md** - Advanced theories done
19. **NEW_THEORIES_IMPLEMENTATION_PLAN.md** - New theories plan
20. **PERFORMANCE_TRACKING_FIX_SUMMARY.md** - Performance fix done
21. **V7_SIGNAL_FIXES.md** - Signal fixes applied
22. **V7_MOMENTUM_OVERRIDE_SUCCESS.md** - Momentum override done
23. **AWS_AUTO_TERMINATION_SUMMARY.md** - Auto-termination implemented

**Recommendation**:
- **Archive to `.archive/v7_implementation_history/`**
- Keep in git history but move out of root
- Only consult when debugging or understanding why something was done

**Total**: 10 files → ARCHIVE

---

### Category 4: ACTIVE ISSUES (Keep Temporarily) - 3 Files ⏳

These document CURRENT issues that need resolution:

24. **CRITICAL_AB_TEST_ISSUES.md** - AB test bugs (NEEDS FIXING)
25. **AB_TEST_IMPLEMENTATION_STATUS.md** - AB test status
26. **V7_AB_TEST_CLARIFICATION.md** - AB test clarification

**Recommendation**:
- **Keep until AB test issues are resolved**
- Once fixed, create single "AB_TEST_FINAL_RESULTS.md" and archive these 3

**Total**: 3 files → KEEP FOR NOW

---

### Category 5: DASHBOARD DUPLICATION (Consolidate) - 2 Files ⚠️

27. **REFLEX_DASHBOARD_SETUP.md** - Reflex setup
28. **REFLEX_DASHBOARD_BACKEND_GUIDE.md** - Reflex backend guide

**Questions for Builder Claude**:
- Is Reflex dashboard actually being used?
- Or is Flask dashboard (`apps/dashboard/app.py`) in production?

**Recommendation**:
- **If using Reflex**: Keep these 2, delete Flask docs
- **If using Flask**: Archive these 2 to `.archive/alternate_dashboards/`
- **Don't keep both** - confusing to have 2 dashboard implementations

**Total**: 2 files → DEPENDS ON BUILDER'S ANSWER

---

### Category 6: PLANNING DOCS (Archive) - 1 File ⚠️

29. **DEEPSEEK_AB_TEST_PLAN.md** - AB test plan (58 lines)

**Recommendation**:
- Archive to `.archive/v7_implementation_history/`
- Plan is done, actual status tracked in AB_TEST_IMPLEMENTATION_STATUS.md

**Total**: 1 file → ARCHIVE

---

### Category 7: REVIEW DOCS (Temporary) - 1 File ✅

30. **QC_REVIEW_BUILDER_CLAUDE_2025-11-21.md** - This QC review (NEW)

**Recommendation**:
- Keep until Builder Claude completes it
- After review complete, archive to `.archive/qc_reviews/`

**Total**: 1 file → TEMPORARY

---

## SUMMARY

| Category | Count | Action |
|----------|-------|--------|
| Essential (Must Keep) | 7 | ✅ KEEP |
| V7 Reference | 6 | ✅ KEEP |
| Implementation History | 10 | ⚠️ ARCHIVE |
| Active Issues | 3 | ⏳ KEEP TEMP |
| Dashboard Duplication | 2 | ❓ TBD |
| Planning Docs | 1 | ⚠️ ARCHIVE |
| Review Docs | 1 | ✅ TEMPORARY |
| **TOTAL** | **30** | |

---

## RECOMMENDED FILE STRUCTURE

### Keep in Root (16-18 files max)

**Essential (7)**:
- README.md
- CLAUDE.md
- PROJECT_MEMORY.md
- MASTER_TRAINING_WORKFLOW.md
- TODO.md
- TROUBLESHOOTING.md
- DATABASE_SCHEMA.md

**V7 Reference (6)**:
- V7_COMPLETE_SYSTEM_DIAGRAM.md
- V7_MASTER_PLAN.md
- V7_MATHEMATICAL_THEORIES.md
- V7_MONITORING.md
- V7_CLOUD_DEPLOYMENT.md
- DASHBOARD_QUICK_REF.md

**Active Issues (3)** - temporary until resolved:
- CRITICAL_AB_TEST_ISSUES.md
- AB_TEST_IMPLEMENTATION_STATUS.md
- V7_AB_TEST_CLARIFICATION.md

**Total in root**: 16 files (or 18 if keeping Reflex docs)

---

### Archive to `.archive/v7_implementation_history/` (12 files)

- STEP4_COMPLETION_SUMMARY.md
- STEP5_COMPLETION_SUMMARY.md
- STEP6_PAPER_TRADING_SUMMARY.md
- IMPLEMENTATION_STEPS.md
- ADVANCED_THEORIES_COMPLETE.md
- NEW_THEORIES_IMPLEMENTATION_PLAN.md
- PERFORMANCE_TRACKING_FIX_SUMMARY.md
- V7_SIGNAL_FIXES.md
- V7_MOMENTUM_OVERRIDE_SUCCESS.md
- AWS_AUTO_TERMINATION_SUMMARY.md
- DEEPSEEK_AB_TEST_PLAN.md
- QC_REVIEW_BUILDER_CLAUDE_2025-11-21.md (after review complete)

---

## CONFLICTING INFORMATION FOUND

### 1. Theory Count Inconsistency

**CRITICAL CONFUSION**:
- `TODO.md` says "6 theories"
- `V7_COMPLETE_SYSTEM_DIAGRAM.md` says "11 theories"
- `CLAUDE.md` says "7 theories"
- `V7_MATHEMATICAL_THEORIES.md` lists different counts

**QUESTION FOR BUILDER**: How many theories are ACTUALLY implemented and running?

**Need to:**
1. Count actual files in `libs/theories/`
2. Check which ones are used in `apps/runtime/v7_runtime.py`
3. Update all docs to show ONE consistent number

---

### 2. Dashboard Implementation Confusion

**Found 3 dashboard implementations**:
1. `apps/dashboard/app.py` (Flask)
2. `apps/dashboard_flask_backup/` (Flask backup)
3. `apps/dashboard_reflex/` (Reflex)

**QUESTION FOR BUILDER**: Which is running in production?

**Need to:**
1. Identify production dashboard
2. Delete unused implementations
3. Update docs to reference only production version

---

### 3. Runtime Version Confusion

**Found 4 runtime files**:
1. `apps/runtime/v7_runtime.py`
2. `apps/runtime/v6_runtime.py`
3. `apps/runtime/v6_fixed_runtime.py`
4. `apps/runtime/v6_statistical_adapter.py`

**QUESTION FOR BUILDER**: Which is running? Can we delete the others?

---

## ACTION ITEMS FOR BUILDER CLAUDE

### Immediate Actions

1. **Answer QC_REVIEW_BUILDER_CLAUDE_2025-11-21.md**
   - Fill in all sections with actual data
   - Run all commands and paste real outputs
   - Be specific about what works vs broken

2. **Resolve Theory Count**
   - Count actual theory files
   - Update all docs to consistent number
   - List which theories are active vs unused

3. **Resolve Dashboard Confusion**
   - Confirm which dashboard is production
   - Delete unused dashboard code
   - Update DASHBOARD_QUICK_REF.md

4. **Resolve Runtime Confusion**
   - Confirm V7 is production runtime
   - Delete unused V6 runtime files (or clearly mark as deprecated)

### Documentation Cleanup Tasks

5. **Archive Implementation History**
   ```bash
   mkdir -p .archive/v7_implementation_history
   mv STEP4_COMPLETION_SUMMARY.md .archive/v7_implementation_history/
   mv STEP5_COMPLETION_SUMMARY.md .archive/v7_implementation_history/
   mv STEP6_PAPER_TRADING_SUMMARY.md .archive/v7_implementation_history/
   mv IMPLEMENTATION_STEPS.md .archive/v7_implementation_history/
   mv ADVANCED_THEORIES_COMPLETE.md .archive/v7_implementation_history/
   mv NEW_THEORIES_IMPLEMENTATION_PLAN.md .archive/v7_implementation_history/
   mv PERFORMANCE_TRACKING_FIX_SUMMARY.md .archive/v7_implementation_history/
   mv V7_SIGNAL_FIXES.md .archive/v7_implementation_history/
   mv V7_MOMENTUM_OVERRIDE_SUCCESS.md .archive/v7_implementation_history/
   mv AWS_AUTO_TERMINATION_SUMMARY.md .archive/v7_implementation_history/
   mv DEEPSEEK_AB_TEST_PLAN.md .archive/v7_implementation_history/
   git add .archive/
   git commit -m "docs: archive V7 implementation history (keep git history)"
   ```

6. **Update TODO.md**
   - Remove completed tasks from 2025-11-19
   - Add current priority tasks
   - Reference active issue docs

7. **Fix AB Test Issues**
   - Once CRITICAL_AB_TEST_ISSUES.md bugs are fixed
   - Create AB_TEST_FINAL_RESULTS.md
   - Archive the 3 AB test docs

---

## EXPECTED END STATE

**Root directory docs** (16 files):
```
README.md                              # Project overview
CLAUDE.md                              # Claude instructions
PROJECT_MEMORY.md                      # Session continuity
MASTER_TRAINING_WORKFLOW.md           # Training guide
TODO.md                                # Active tasks
TROUBLESHOOTING.md                     # Common issues
DATABASE_SCHEMA.md                     # Database docs

V7_COMPLETE_SYSTEM_DIAGRAM.md         # V7 architecture
V7_MASTER_PLAN.md                      # V7 plan
V7_MATHEMATICAL_THEORIES.md            # Theory docs
V7_MONITORING.md                       # Monitoring guide
V7_CLOUD_DEPLOYMENT.md                 # Deployment guide
DASHBOARD_QUICK_REF.md                 # Dashboard reference

CRITICAL_AB_TEST_ISSUES.md             # Active issue (temp)
AB_TEST_IMPLEMENTATION_STATUS.md       # Active status (temp)
V7_AB_TEST_CLARIFICATION.md            # Active clarification (temp)
```

**Archived** (all in git history):
- `.archive/old_docs/` (165 files) - Already done
- `.archive/v7_implementation_history/` (12 files) - To be done
- `.archive/qc_reviews/` (1+ files) - After reviews complete

---

## NOTES FOR QC CLAUDE

After Builder Claude completes QC_REVIEW_BUILDER_CLAUDE_2025-11-21.md:

1. Review their answers
2. Identify remaining issues
3. Create prioritized action plan
4. Decide if V7 is production-ready or needs fixes
5. Update this cleanup analysis based on their clarifications

---

**Status**: ⏳ AWAITING BUILDER CLAUDE RESPONSE
**Next**: Builder fills out QC_REVIEW_BUILDER_CLAUDE_2025-11-21.md
