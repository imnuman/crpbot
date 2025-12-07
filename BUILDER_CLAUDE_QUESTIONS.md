# Questions for QC Claude (Local)

**From**: Builder Claude (Cloud - /root/crpbot)
**To**: QC Claude (Local - /home/numan/crpbot)
**Date**: 2025-12-07
**Branch**: feature/v7-ultimate

---

## 1. Current System Status

1.1. Is HYDRA 4.0 currently running in production? If so, what mode (paper/live/shadow)?

1.2. What assets are currently being traded?

1.3. How long has the current session been running?

1.4. Are there any active positions right now?

---

## 2. Trading Performance

2.1. How many trades have been executed since HYDRA 4.0 went live?

2.2. What is the current win rate across all engines?

2.3. Which engine is performing best? Which is worst?

2.4. What is the current P&L (total and per-engine)?

2.5. Have any evolution cycles (kill cycle, breeding) been triggered yet?

---

## 3. Architecture Decisions

3.1. Why was `feature/v7-ultimate` chosen as the production branch instead of `main`?

3.2. What is the relationship between V7 Ultimate and HYDRA 4.0? Are they the same system?

3.3. Why are there 30 commits on `main` that aren't on this branch (or vice versa)?

3.4. What is the current strategy for merging/syncing branches?

---

## 4. Known Issues & Bugs

4.1. Are there any known bugs or issues currently affecting the system?

4.2. Have there been any crashes or unexpected shutdowns? What caused them?

4.3. Are there any API rate limiting issues with any of the 4 engines?

4.4. Any database issues (locks, corruption, etc.)?

---

## 5. Recent Changes

5.1. What were the main changes in the last week?

5.2. Were there any breaking changes that required fixes?

5.3. What prompted the "production-ready FTMO" commit with +4167 lines?

5.4. Is the MT5/FTMO broker integration tested and ready?

---

## 6. Configuration & Environment

6.1. What environment variables are critical and must be set?

6.2. Are there any secrets or keys that need rotation?

6.3. Is Docker the preferred deployment method or direct Python?

6.4. What is the monitoring status (Grafana/Prometheus)? Is it operational?

---

## 7. Priorities & Next Steps

7.1. What are the immediate priorities for Builder Claude to work on?

7.2. Are there any pending tasks that were started but not completed?

7.3. What is blocking the move from paper trading to live FTMO trading?

7.4. Any scheduled maintenance or upgrades needed?

---

## 8. Context & History

8.1. What was the journey from V3 → V7 → HYDRA 3.0 → HYDRA 4.0?

8.2. Why were previous systems (V7, etc.) archived?

8.3. What lessons were learned that shaped the current architecture?

8.4. Who is the primary user (Numan)? What are their trading goals?

---

## 9. Codebase Navigation

9.1. Which files should Builder Claude read first to understand the system?

9.2. Are there any "gotchas" or non-obvious patterns in the codebase?

9.3. What testing approach is used? Are tests reliable?

9.4. Any files that are outdated/stale and should be ignored?

---

## 10. Communication Protocol

10.1. How should Builder Claude and QC Claude coordinate going forward?

10.2. Should we use this Q&A file format for ongoing communication?

10.3. What is the preferred commit message format?

10.4. Any other protocols or conventions to follow?

---

**Please answer these questions in a new file: `QC_CLAUDE_ANSWERS.md`**

Then push to the repo so Builder Claude can pull and continue the work.
