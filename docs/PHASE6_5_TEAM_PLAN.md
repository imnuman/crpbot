# Phase 6.5 Collaboration Plan

## 🎯 Goal
Run the 3–5 day silent observation smoothly with clear ownership across Cursor, Claude, and Amazon Q so we can advance to Phase 7 without rework or surprises.

---

## 🤝 Roles & Responsibilities

**Cursor (Local Runtime Owner)**
- Operate runtime (`make run-dry`), collect daily evidence, update `reports/phase6_5/dayX.md`.
- Run automation helpers (`make export-metrics`, `make phase6_5-daily DAY=...`) and keep `tradingai.db` healthy.
- Escalate Sev‑1/Sev‑2 issues immediately and document remediation steps.

**Claude (Code & QA Reviewer)**
- Daily review of automated snapshots + logs; flag anomalies or regression risks.
- Validate observation journal entries and sign off on go/no-go in `reports/phase6_5/summary.md`.
- Provide code/doc feedback post-observation (before Phase 7) if adjustments are needed.

**Amazon Q (AWS Specialist)**
- Standby for infra adjustments if alarms fire or throughput tuning is required.
- Twice during the observation window (Day 1 and Day 3) confirm CloudWatch dashboard health via AWS console.
- After wrap-up, prepare any required AWS changes for Phase 7 (e.g., scaling policies, new SNS targets).

---

## 📆 Timeline & Task Sequence

| Day | Cursor | Claude | Amazon Q |
|-----|--------|--------|----------|
| **Prep (T‑0)** | Confirm checklist in `day0.md`, start runtime | Review readiness (docs/PROJECT_STATUS.md) | Verify CloudFormation stacks, dashboards render |
| **Day 1** | `make export-metrics`, `make phase6_5-daily DAY=day1`, update journal | Review automated snapshot, Telegram logs | Dashboard spot-check, note anomalies |
| **Day 2** | Repeat automation, run simulated guardrail test if planned | Validate guardrail results, comment in journal | Standby (no action unless escalated) |
| **Day 3** | Final required day of observation, capture metrics | Review trend vs Day 1/2, prep go/no-go notes | Second dashboard audit, ensure alarms quiet |
| **Day 4/5 (optional)** | Extend observation if issues occurred | Validate recovery notes | Assist with escalation if needed |
| **Wrap-up** | Complete `summary.md`, assemble evidence bundle | Provide go/no-go approval | Draft follow-up AWS tasks for Phase 7 |

---

## ✅ Daily Automation Checklist (Cursor)
1. `make export-metrics WINDOW=24 OUT=reports/phase6_5/dayX_metrics.json`
2. `make phase6_5-daily DAY=dayX`
3. Capture CloudWatch screenshots (store under `reports/phase6_5/screenshots/` if needed)
4. Update the journal template with qualitative notes (manual observations)

Claude reviews steps 2–4 daily; Amazon Q reviews screenshots on assigned days.

---

## 🔄 Escalation Flow
1. **Cursor** detects issue via alarms/logs.
2. Document in `reports/phase6_5/dayX.md` under “Issues / Actions”.
3. Notify **Claude** (analysis) and **Amazon Q** (AWS fixes if required).
4. Pause observation window if Sev‑1/Sev‑2; restart clock once resolved.

---

## 📈 After Observation
- Cursor compiles `reports/phase6_5/summary.md` with metrics and evidence.
- Claude reviews and signs off on the go/no-go decision for Phase 7.
- Amazon Q prepares any infrastructure changes needed ahead of micro-lot testing.
- Bugbot reviews the wrap-up pull request (`@Bugbot review`) before merge to catch regressions.

This sequence keeps everyone aligned and minimizes context-switching as we move toward Phase 7 and beyond.***

