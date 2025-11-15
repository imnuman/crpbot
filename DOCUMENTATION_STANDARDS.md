# 📝 Documentation Standards - CRPBot Project

**Last Updated**: 2025-11-15
**Purpose**: Ensure consistent, maintainable documentation across all sessions

---

## 🎯 Core Principles

1. **Use ISO Date Format**: Always `YYYY-MM-DD`
2. **No Relative Dates**: Never use "TODAY", "YESTERDAY", "TOMORROW"
3. **Timestamp Critical Changes**: Include date/time in important updates
4. **Version Everything**: Use clear version markers (V1, V2, V3, etc.)
5. **Update Logs**: Track who changed what and when

---

## 📁 File Naming Conventions

### ✅ CORRECT Examples

```
SESSION_SUMMARY_2025-11-14.md
AWS_SETUP_CHECKLIST_2025-11-15.md
QC_REVIEW_2025-11-15_1430.md
BUILDER_CLAUDE_SUMMARY_2025-11-15.md
V5_PHASE1_PLAN.md (version-based, not date-based)
PROJECT_STATUS_2025-11-15.md
```

### ❌ INCORRECT Examples

```
SESSION_SUMMARY_TODAY.md              ❌ Use: SESSION_SUMMARY_2025-11-15.md
AWS_SETUP_CHECKLIST_FRIDAY.md         ❌ Use: AWS_SETUP_CHECKLIST_2025-11-15.md
QC_REVIEW_LATEST.md                   ❌ Use: QC_REVIEW_2025-11-15.md
BUILDER_SUMMARY_YESTERDAY.md          ❌ Use: BUILDER_CLAUDE_SUMMARY_2025-11-14.md
PROJECT_STATUS_CURRENT.md             ❌ Use: PROJECT_STATUS_2025-11-15.md
```

---

## 🏷️ Naming Pattern Reference

### Pattern 1: Date-Specific Documents
**Format**: `{TYPE}_{YYYY-MM-DD}.md`

**Use when**: Document is tied to a specific date/session

**Examples**:
- `SESSION_SUMMARY_2025-11-15.md`
- `DAILY_REPORT_2025-11-15.md`
- `AWS_SETUP_2025-11-15.md`

---

### Pattern 2: Date + Time (When Multiple Per Day)
**Format**: `{TYPE}_{YYYY-MM-DD}_{HHMM}.md`

**Use when**: Multiple documents of same type created on same day

**Examples**:
- `QC_REVIEW_2025-11-15_0930.md` (morning review)
- `QC_REVIEW_2025-11-15_1430.md` (afternoon review)
- `EMERGENCY_FIX_2025-11-15_2145.md`

---

### Pattern 3: Version-Based (Not Date-Based)
**Format**: `{TYPE}_V{N}.md` or `{TYPE}_{VERSION}.md`

**Use when**: Document tracks project versions/phases, not dates

**Examples**:
- `V5_SIMPLE_PLAN.md` (version 5 plan)
- `V5_BUDGET_PLAN.md`
- `PHASE6_5_RESTART_PLAN.md`
- `CLAUDE.md` (living document, frequently updated)
- `PROJECT_MEMORY.md` (living document)

**Update Rule**: Include "Last Updated: YYYY-MM-DD" at the top

---

### Pattern 4: Agent/Role-Specific
**Format**: `{AGENT}_{TYPE}_{YYYY-MM-DD}.md`

**Use when**: Document is created by specific agent

**Examples**:
- `BUILDER_CLAUDE_SUMMARY_2025-11-15.md`
- `QC_CLAUDE_REVIEW_2025-11-15.md`
- `AMAZONQ_AWS_DEPLOYMENT_2025-11-15.md`

---

### Pattern 5: Topic-Specific (Timeless)
**Format**: `{TOPIC}_{DESCRIPTION}.md`

**Use when**: Document is reference material, not time-specific

**Examples**:
- `DATA_STRATEGY_COMPLETE.md`
- `FEATURE_ENGINEERING_WORKFLOW.md`
- `DOCUMENTATION_STANDARDS.md` (this file)

**Update Rule**: Include "Last Updated: YYYY-MM-DD" at the top

---

## 📅 Date Format Standards

### ISO 8601 Format (Preferred)
```
Date: 2025-11-15
DateTime: 2025-11-15T14:30:00Z
Time: 14:30 (24-hour format)
```

### ✅ Good Date References
```markdown
**Last Updated**: 2025-11-15 13:20 EST (Toronto)
**Created**: 2025-11-15 14:30 EST (Toronto)
**Session Date**: November 15, 2025 (2025-11-15)
**Modified**: 2025-11-15T14:30:00-05:00
```

### ❌ Bad Date References
```markdown
**Last Updated**: Today
**Created**: Friday afternoon
**Session Date**: Last week
**Modified**: Yesterday
```

### 🌍 Timezone Standard
**All timestamps use Toronto time (America/Toronto timezone)**
- Winter: EST (UTC-5)
- Summer: EDT (UTC-4)
- Always include timezone: `EST (Toronto)` or `EDT (Toronto)`
- Get current time: `TZ="America/Toronto" date "+%Y-%m-%d %H:%M %Z"`

---

## 🔄 Living Documents

Some documents are updated frequently and don't need dates in filename:

### Always Current (No Date in Filename)
- `CLAUDE.md` - Include "Last Updated: YYYY-MM-DD" at top
- `PROJECT_MEMORY.md` - Include "Last Updated: YYYY-MM-DD" at top
- `README.md` - Include version or date
- `WORK_PLAN.md` - Include "Last Updated: YYYY-MM-DD" at top

### Must Include Update Log
```markdown
## Update Log

| Date | Update | Author |
|------|--------|--------|
| 2025-11-15 | Added AWS setup section | QC Claude |
| 2025-11-14 | V5 pivot documented | Builder Claude |
| 2025-11-13 | Initial creation | Local Claude |
```

---

## 📊 Document Headers

### Standard Header Template
```markdown
# {Document Title}

**Created**: 2025-11-15 14:30 EST (Toronto)
**Last Updated**: 2025-11-15 14:30 EST (Toronto)
**Author**: {Claude Role} (QC/Builder/Amazon Q)
**Status**: {Draft/Review/Complete/Obsolete}
**Version**: {If applicable}
**Purpose**: {One-line description}

---
```

### Example
```markdown
# AWS Setup Checklist

**Created**: 2025-11-15 13:45 EST (Toronto)
**Last Updated**: 2025-11-15 13:45 EST (Toronto)
**Author**: QC Claude
**Status**: Ready for Execution
**Purpose**: Complete AWS infrastructure setup for V5 deployment

---
```

### For Living Documents (Frequently Updated)
```markdown
# Project Memory

**Created**: 2025-11-13 10:00 EST (Toronto)
**Last Updated**: 2025-11-15 13:11 EST (Toronto)
**Author**: Multiple (QC Claude, Builder Claude)
**Status**: Living Document
**Purpose**: Persistent memory for Claude instances

---
```

---

## 🗂️ Archive Strategy

### When to Archive
- Document becomes obsolete (e.g., V4 plans when on V5)
- Replaced by newer version
- Completed one-time task (e.g., setup checklists)

### Archive Location
```
archive/
├── 2025-11/
│   ├── V4_PLANS/
│   │   ├── V4_TRAINING_PLAN.md
│   │   └── V4_FEATURES.md
│   └── COMPLETED_TASKS/
│       └── AWS_SETUP_CHECKLIST_2025-11-15.md (after completion)
└── 2025-10/
    └── ...
```

### Archive Naming
Keep original filename + add archive date:
```
Original: AWS_SETUP_CHECKLIST_2025-11-15.md
Archived: archive/2025-11/AWS_SETUP_CHECKLIST_2025-11-15_COMPLETED.md
```

---

## 🚨 Common Mistakes to Avoid

### ❌ Mistake 1: Relative Dates
```markdown
❌ "Created yesterday"
✅ "Created 2025-11-14"

❌ "Updated last week"
✅ "Updated 2025-11-08"

❌ "Tomorrow's tasks"
✅ "Tasks for 2025-11-16"
```

### ❌ Mistake 2: Ambiguous Time References
```markdown
❌ "This morning"
✅ "2025-11-15 09:30 EST"

❌ "Earlier today"
✅ "2025-11-15 14:00 EST"

❌ "Soon"
✅ "Estimated completion: 2025-11-15 17:00"
```

### ❌ Mistake 3: Missing Update Tracking
```markdown
❌ Just update content without noting change
✅ Add to changelog or update "Last Updated" field
```

---

## 📝 Git Commit Message Standards

### Include Dates When Relevant
```bash
✅ Good:
git commit -m "docs: add AWS setup checklist for 2025-11-15"
git commit -m "fix: update PROJECT_MEMORY.md (2025-11-15)"
git commit -m "refactor: archive V4 plans (obsolete as of 2025-11-15)"

❌ Bad:
git commit -m "docs: add AWS setup checklist for today"
git commit -m "fix: update memory file"
git commit -m "refactor: archive old plans"
```

---

## 🔍 Quick Checklist Before Creating New Document

- [ ] Does filename include ISO date if document is date-specific?
- [ ] Does header include "Created" timestamp (YYYY-MM-DD HH:MM TZ)?
- [ ] Does header include "Last Updated" timestamp?
- [ ] Are all date references using ISO format (YYYY-MM-DD)?
- [ ] No words like "today", "yesterday", "tomorrow"?
- [ ] If living document, includes update log?
- [ ] Git commit message includes date if relevant?
- [ ] Author clearly identified?

---

## 📚 Reference: Project Naming Examples

```
✅ GOOD EXISTING FILES:
- SESSION_SUMMARY_2025-11-14.md
- BUILDER_CLAUDE_SUMMARY_2025-11-15.md
- V5_SIMPLE_PLAN.md (version-based, OK)
- V5_BUDGET_PLAN.md (version-based, OK)
- CLAUDE.md (living doc, has "Last Updated")
- PROJECT_MEMORY.md (living doc, has "Last Updated")

✅ RECENTLY FIXED:
- AWS_SETUP_CHECKLIST_TODAY.md → AWS_SETUP_CHECKLIST_2025-11-15.md

⚠️ NEEDS FIXING (if found):
- Any file with "TODAY", "LATEST", "CURRENT", "YESTERDAY" in name
- Any date reference without year
- Any relative time references
```

---

## 🎯 Summary: Golden Rules

1. **Always use ISO dates**: `YYYY-MM-DD`
2. **Never use relative dates**: "today", "yesterday", etc.
3. **Include creation timestamp**: `Created: YYYY-MM-DD HH:MM TZ` in every document header
4. **Include last updated**: `Last Updated: YYYY-MM-DD HH:MM TZ` in every document header
5. **Update logs**: Track all changes in living documents
6. **Archive properly**: Keep history organized

---

**This document**: `DOCUMENTATION_STANDARDS.md`
**Created**: 2025-11-15 13:50 EST (Toronto)
**Last Updated**: 2025-11-15 13:20 EST (Toronto)
**Author**: QC Claude
**Status**: Living Document (update as needed)
