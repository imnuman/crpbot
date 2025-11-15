# Document Templates

**Created**: 2025-11-15 13:20 EST (Toronto)
**Last Updated**: 2025-11-15 14:20 EST (Toronto)
**Author**: QC Claude
**Purpose**: Standard templates for all project documentation

---

## Available Templates

### 1. DOCUMENT_TEMPLATE.md
**Use for**: Any new documentation file
**Contains**: Standard header with creation time, last updated, author, status, purpose

---

## How to Use

### Step 1: Copy Template
```bash
cp .templates/DOCUMENT_TEMPLATE.md NEW_DOC_NAME_2025-11-15.md
```

### Step 2: Fill in Header
Replace all {placeholders}:
- **Created**: Current timestamp (YYYY-MM-DD HH:MM TZ)
- **Last Updated**: Same as Created initially
- **Author**: Your role (QC Claude, Builder Claude, etc.)
- **Status**: Draft, Review, Complete, etc.
- **Purpose**: One-line description

### Step 3: Add Content
Fill in sections as needed

### Step 4: Update Timestamps
Every time you modify:
- Update "Last Updated" field
- If living document, add entry to Update Log

---

## Examples of Proper Headers

### New Document (Just Created)
```markdown
# AWS Deployment Guide

**Created**: 2025-11-15 14:17 EST
**Last Updated**: 2025-11-15 14:17 EST
**Author**: QC Claude
**Status**: Draft
**Purpose**: Guide for deploying CRPBot to AWS
```

### Modified Document
```markdown
# AWS Deployment Guide

**Created**: 2025-11-15 14:17 EST
**Last Updated**: 2025-11-15 16:45 EST
**Author**: QC Claude
**Status**: Complete
**Purpose**: Guide for deploying CRPBot to AWS
```

### Living Document
```markdown
# Project Status Tracker

**Created**: 2025-11-13 10:00 EST
**Last Updated**: 2025-11-15 14:17 EST
**Author**: Multiple (QC Claude, Builder Claude)
**Status**: Living Document
**Purpose**: Track ongoing project status and blockers

## Update Log

| Date | Time | Update | Author |
|------|------|--------|--------|
| 2025-11-13 | 10:00 | Initial creation | QC Claude |
| 2025-11-14 | 15:30 | Added V5 pivot info | Builder Claude |
| 2025-11-15 | 14:17 | Updated AWS status | QC Claude |
```

---

## Quick Reference

**Get current timestamp (Toronto time):**
```bash
TZ="America/Toronto" date "+%Y-%m-%d %H:%M %Z"
```

**Example output:**
```
2025-11-15 13:20 EST
```

---

**See also**: `DOCUMENTATION_STANDARDS.md` for complete documentation guidelines
