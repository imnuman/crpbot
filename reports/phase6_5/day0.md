# Phase 6.5 – Day 0 Readiness Checklist

Date: <!-- YYYY-MM-DD -->

## Configuration
- [ ] `main` branch synced with `aws/rds-setup`
- [ ] `.env` updated (FTMO demo, Telegram, Coinbase)
- [ ] `RUNTIME_MODE=dryrun`
- [ ] Models synced in `models/promoted/`

## Infrastructure
- [ ] Lambda stacks (`signal`, `risk`, `telegram`) `CREATE_COMPLETE`
- [ ] CloudWatch dashboards accessible
- [ ] CloudWatch alarms in `OK` / `INSUFFICIENT_DATA`
- [ ] S3 buckets reachable (`make export-metrics` smoke test)

## Observability
- [ ] `logs/` writable, rotation verified
- [ ] Structured log validation (`python scripts/validate_logs.py --sample logs/runtime`)
- [ ] Telegram `/check` command response <5s
- [ ] Export baseline metrics (`make export-metrics WINDOW=24 OUT=reports/phase6_5/day0_metrics.json`)

## Notes
- 

## Go/No-Go
- [ ] Go for runtime start
- [ ] Issues found (document above)

