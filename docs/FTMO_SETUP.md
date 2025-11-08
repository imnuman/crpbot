# FTMO Account Setup Guide

## 1. Choose Which FTMO Account to Use

| Use Case | Recommendation | Notes |
| --- | --- | --- |
| Execution-model measurement (Phase 2.3) | **FTMO Demo (free)** | Enough to record spreads/slippage. Creates read/write MT5 login instantly. |
| Micro-lot testing (Phase 7) | **FTMO Challenge (paid)** | Needed once models pass validation; buy the challenge no later than Phase 6.5 so credentials are ready for Phase 7. |
|

> 💡 Start with the free **FTMO Demo** so you can integrate credentials immediately, then upgrade to a Challenge account once the silent observation period (Phase 6.5) begins.

## 2. Create the FTMO Account

1. Sign in at [https://client.ftmo.com](https://client.ftmo.com).
2. Navigate to **Accounts → Demo** (or **Orders → Get FTMO Challenge** if purchasing).
3. Click **Create new demo** (or finish the Challenge purchase wizard).
4. Select **MetaTrader 5** and choose the server region closest to your VPS (e.g., `FTMO-Demo`, `FTMO-DemoEU`, `FTMO-Server` for live).
5. Copy the generated credentials:
   - **Login ID** (numeric)
   - **Password** (trading password)
   - **Server** (e.g., `FTMO-Demo`, `FTMO-Server`)

Save them in your password manager—the password is only shown once on creation.

## 3. Add Credentials to `.env`

Open `/home/numan/crpbot/.env` and add/verify the following entries:

```bash
FTMO_LOGIN=12345678
FTMO_PASS=your_trading_password
FTMO_SERVER=FTMO-Demo
```

> ✅ `.env` is already git-ignored, so the values stay local.

## 4. Update Documentation Checklists

After populating `.env`, mark the FTMO section in `docs/CREDENTIALS_CHECKLIST.md` as complete (or note the account type you created).

## 5. Verify the Credentials

1. **Smoke Test:**
   ```bash
   source .venv/bin/activate
   python scripts/nightly_exec_metrics.py --once
   ```
   This runs the nightly execution metrics script a single time. A successful run logs sampled spread/slippage values to `data/execution_metrics/`.

2. **Cron Integration:** Ensure `infra/scripts/nightly_exec_metrics.sh` will run with the same `.env` file on the VPS so the execution model stays current.

3. **Manual Validation (Optional):** Log in to MT5 manually using the credentials to confirm the account is active and receiving ticks.

## 6. Observe FTMO Terms & Safeguards

- Demo accounts reset automatically; note the balance for daily-loss calculations.
- Read-only mode is not offered—protect the credentials carefully.
- Never automate actions that violate FTMO’s ToS (e.g., high-frequency request bursts or prohibited expert advisors). Our runtime only reads market data and enforces FTMO risk guardrails.

## 7. Integration Checklist

- [ ] FTMO Demo credentials created and stored securely
- [ ] `.env` updated with `FTMO_LOGIN`, `FTMO_PASS`, `FTMO_SERVER`
- [ ] `scripts/nightly_exec_metrics.py --once` runs successfully
- [ ] Cron/VPS plan ready for nightly execution metrics
- [ ] Challenge account purchased (before Phase 7)

Once these items are checked off, Phase 6.5 (silent observation) can begin with real FTMO execution data, and Phase 7 will already have working credentials.
