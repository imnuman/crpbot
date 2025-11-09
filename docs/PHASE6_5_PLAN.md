# Phase 6.5 Silent Observation Runbook

## 🎯 Objective
- Operate the full CRPBot stack in **dry-run** mode for **3–5 consecutive days**.
- Validate system stability, signal quality, and guardrail enforcement **without placing live trades**.
- Produce an evidence bundle that Phase 7 (micro-lot testing) can safely begin.

---

## 🗓️ Schedule & Ownership
| Day | Focus | Owner | Evidence |
|-----|-------|-------|----------|
| T‑0 (Prep) | Configure runtime, confirm CloudWatch dashboards/alarms, snapshot configs | Cursor | Checklist in `reports/phase6_5/day0.md` |
| T+1 | Start runtime (dry-run mode), verify first 12h of signals/logs | Cursor | Log excerpts, CloudWatch dashboard screenshot |
| T+2 | Mid-run review: guardrail metrics, Telegram responsiveness | Cursor | Metrics export (`reports/phase6_5/day2_metrics.json`) |
| T+3/4/5 | Continue monitoring (minimum 72h total); extend to 120h if any Sev‑2 issues arise | Cursor | Daily run log |
| Wrap-up | Compile summary + go/no-go decision for Phase 7 | Cursor + User | `reports/phase6_5/summary.md` signed off |

> 🔄 **If any critical alarm fires** (Lambda errors, SNS failures, guardrail breach) pause the observation, fix root cause, and restart the clock.

---

## ✔️ Readiness Checklist (T‑0)
1. **Code & Config**
   - [ ] `main` branch synced with `aws/rds-setup` deployment docs.
   - [ ] `.env` includes FTMO demo credentials (verify by running `python scripts/check_credentials.py ftmo`).
   - [ ] `Settings.kill_switch` set to `False`; `dry_run` flag enabled.
   - [ ] Runtime confidence thresholds match latest calibration (`libs/constants.py`).
2. **AWS Infrastructure**
   - [ ] Lambda signal processor / risk monitor / Telegram relay stacks in `CREATE_COMPLETE`.
   - [ ] CloudWatch dashboards (`CRPBot-Trading-dev`, `CRPBot-System-dev`) load without errors.
   - [ ] CloudWatch alarms all in `OK` or `INSUFFICIENT_DATA`.
3. **Observability**
   - [ ] `logs/` directory writeable; log rotation configured.
   - [ ] Structured JSON logs validated (`python scripts/validate_logs.py --sample logs/runtime/*.jsonl`).
   - [ ] Telegram `/check` command responds within 5s.
4. **Data & Models**
   - [ ] Latest promoted models synced to `models/promoted/`.
   - [ ] Feature store contains last 30 days of data (`data/features/*.parquet`).
   - [ ] Backtest snapshot recorded for baseline (`reports/backtests/latest.json`).

---

## 🔁 Daily Operating Procedure
1. **Startup (09:00 UTC)**
   - Activate runtime: `python apps/runtime/main.py --mode dry-run --iterations -1`.
   - Confirm first signal logged; ensure FTMO and rate-limiter checks pass in logs.
   - Note EventBridge → Lambda invocations in CloudWatch metrics.
2. **Midday Checks (14:00 UTC)**
   - Review dashboards for anomalies (invocations, errors, latency).
   - Inspect `RiskBookSnapshot` table for daily loss/total loss utilization.
   - Trigger manual Telegram `/stats` to confirm fresh data.
3. **Evening Review (21:00 UTC)**
   - Tail structured logs for WARN/ERROR.
   - Export metrics via `python scripts/export_metrics.py --window 24h --out reports/phase6_5/dayX_metrics.json`.
   - Update observation journal (`reports/phase6_5/dayX.md`) with:
     - Total signals
     - Tier distribution
     - Any alarms/alerts
     - Manual interventions if any
4. **Overnight**
   - Leave runtime running. If using tmux/supervised process, ensure auto-restart is disabled (manual control only during observation).

---

## 📊 Metrics & Evidence Capture
- **Signals**: `Signal` table exports (`scripts/dump_signals.py --since 24h`).
- **Risk**: Loss utilization charts (daily vs total) from dashboard or SQL query.
- **Latency**: Lambda and runtime latency metrics (P50/P90) recorded daily.
- **Alerts**: CloudWatch alarm history exported at wrap-up.
- **Telegram**: Screenshot/log of notifications and command responses.
- **Costs**: Monitor AWS Cost Explorer to ensure expected ~$5.26/month baseline.

Store artifacts under `reports/phase6_5/` with date-stamped filenames.

---

## 🚦 Exit Criteria
All must be satisfied to declare Phase 6.5 complete:
1. ≥72h continuous runtime with **zero crash loops** and no Sev‑1 alarms.
2. FTMO guardrails enforced in at least one simulated breach scenario (e.g., inject test loss via `scripts/simulate_loss.py`).
3. Telegram notifications delivered for every high-tier signal and any risk alert.
4. Observed win-rate & tier distributions within ±5% of backtest baselines.
5. Final summary report approved, with explicit go/no-go recommendation for Phase 7.

If criteria are not met, resolve issues and repeat a fresh 72h observation window.

---

## 📌 Dependencies & Action Items
- [ ] Populate `reports/phase6_5/` directory scaffold (`day0.md`, `day1.md`, etc.).
- [ ] Update `Makefile` with shortcuts:
  - `make run-dry` → starts runtime in dry-run mode.
  - `make export-metrics` → wrapper for metrics export script.
- [ ] Coordinate FTMO Challenge purchase timing (must be ready by Phase 7 kickoff).
- [ ] Communicate observation start/stop to collaborators (Claude, Amazon Q) to avoid conflicting deployments.

---

## 🆘 Escalation Matrix
| Severity | Example | Action | Contact |
|----------|---------|--------|---------|
| Sev‑1 | Runtime crash, repeated alarms, data corruption | Stop observation, investigate immediately | Cursor (primary), User (escalation) |
| Sev‑2 | Single alarm spike, delayed Telegram message | Investigate within 4h, document impact | Cursor |
| Sev‑3 | Minor log warnings, cosmetic dashboard issue | Log in journal, fix post-observation | Cursor |

---

## ✅ Ready to Launch
Once the checklist is green, log `reports/phase6_5/day0.md`, start the runtime, and set a calendar reminder for the daily reviews. At wrap-up, collate artifacts and prepare the Phase 7 micro-lot testing plan.

