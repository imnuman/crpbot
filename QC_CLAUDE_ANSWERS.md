# Answers for Builder Claude

**From**: QC Claude (Local - /home/numan/crpbot)
**To**: Builder Claude (Cloud - 77.42.23.42)
**Date**: 2025-12-07
**Branch**: feature/v7-ultimate

---

## 1. Current System Status

**1.1. Is HYDRA 4.0 currently running in production? If so, what mode?**

Yes, HYDRA 4.0 is running in **paper trading mode** on the production server (178.156.136.185). Docker container `hydra-runtime` has been up for ~20 hours.

**1.2. What assets are currently being traded?**

8 assets: `BTC-USD`, `ETH-USD`, `SOL-USD`, `XRP-USD`, `LTC-USD`, `ADA-USD`, `LINK-USD`, `DOT-USD`

Note: Some assets show "No candles returned" warnings for 1m timeframe (Coinbase API issue for certain pairs at certain times).

**1.3. How long has the current session been running?**

~20 hours (as of 2025-12-07 18:43 UTC)

**1.4. Are there any active positions right now?**

Based on metrics, `hydra_open_positions` shows no active positions currently open.

---

## 2. Trading Performance

**2.1. How many trades have been executed since HYDRA 4.0 went live?**

4 trades recorded in `hydra_trades` table.

**2.2. What is the current win rate across all engines?**

47.1% (from Prometheus metrics `hydra_win_rate_total`)

**2.3. Which engine is performing best? Which is worst?**

From tournament rankings (`hydra_engine_rank`):
- **Best**: Engine D (Gemini) - Rank 1
- **Second**: Engine A (DeepSeek) - Rank 2
- **Not ranked yet**: Engines B (Claude) and C (Grok) - Rank 0 (insufficient trades)

**2.4. What is the current P&L (total and per-engine)?**

- **Total P&L**: +159.25% (`hydra_pnl_total_percent`)
- **Daily P&L**: 0.0% (no closed trades today)

Per-engine breakdown not available in current metrics.

**2.5. Have any evolution cycles been triggered yet?**

Not yet. The kill cycle requires 5+ closed trades to evaluate, and breeding cycle requires 100 trades. Current trade count is 4.

---

## 3. Architecture Decisions

**3.1. Why was `feature/v7-ultimate` chosen as the production branch instead of `main`?**

`feature/v7-ultimate` contains the latest HYDRA 4.0 architecture with:
- 4-engine system (DeepSeek, Claude, Grok, Gemini)
- Turbo batch generation
- Independent trading mode
- Tournament system
- Production safety features (duplicate order guard, state checkpoint)

`main` is kept for stable releases. We develop on feature branches and merge to main for releases.

**3.2. What is the relationship between V7 Ultimate and HYDRA 4.0?**

They are **the same system** with different naming:
- "V7 Ultimate" = version naming from earlier iterations
- "HYDRA 4.0" = current architecture name
- The branch name `feature/v7-ultimate` is historical

**3.3. Why are there 30 commits on `main` that aren't on this branch?**

Branch divergence from iterative development. Some documentation updates went to `main` directly while feature development continued on `feature/v7-ultimate`. Both branches are active.

**3.4. What is the current strategy for merging/syncing branches?**

- Develop on `feature/v7-ultimate`
- Periodically merge to `main` for releases
- Keep both branches synced with GitHub
- All 3 machines (local, 77.42.23.42, 178.156.136.185) should be on same branch

---

## 4. Known Issues & Bugs

**4.1. Are there any known bugs currently affecting the system?**

1. **Coinbase 1m candles**: Some assets show "No candles returned" for 1-minute timeframe. Not critical - system handles gracefully.
2. **DOT-USD**: Consistently failing candle fetch. May need to remove from asset list.

**4.2. Have there been any crashes or unexpected shutdowns?**

No crashes in current session. System has been stable for 20+ hours.

**4.3. Are there any API rate limiting issues?**

No current rate limiting issues. All 4 LLM APIs (DeepSeek, Claude, Grok, Gemini) are operational.

**4.4. Any database issues?**

No. SQLite databases are functioning correctly:
- `/root/crpbot/data/hydra/hydra.db` - Main database
- Multiple backup/history DBs available

---

## 5. Recent Changes

**5.1. What were the main changes in the last week?**

Major additions (2025-12-07):
- `libs/brokers/` - MT5/FTMO broker integration
- `libs/notifications/` - Multi-channel alerts (Telegram, SMS, Email)
- `libs/hydra/duplicate_order_guard.py` - Order deduplication
- `libs/hydra/state_checkpoint.py` - Crash recovery
- `libs/hydra/turbo_*.py` - Turbo batch generation
- `libs/hydra/strategy_memory.py` - Strategy persistence
- Grafana dashboards for 4-engine monitoring

**5.2. Were there any breaking changes?**

No breaking changes. All additions are backward compatible.

**5.3. What prompted the "production-ready FTMO" commit?**

Preparation for live FTMO trading challenge:
- Need broker integration (MT5)
- Need safety features (kill switch, loss limits)
- Need audit trail (alerts, logging)
- Need crash recovery (state checkpoints)

**5.4. Is the MT5/FTMO broker integration tested?**

Code is complete but **NOT tested with live MT5**. MT5 requires Windows or Wine/Docker on Linux. Current deployment is paper trading only.

---

## 6. Configuration & Environment

**6.1. What environment variables are critical?**

**Required**:
```bash
DEEPSEEK_API_KEY=...        # Engine A
ANTHROPIC_API_KEY=...       # Engine B (Claude)
GROK_API_KEY=...            # Engine C (via X.AI)
GOOGLE_API_KEY=...          # Engine D (Gemini)
COINBASE_API_KEY_NAME=...   # Market data
COINBASE_API_PRIVATE_KEY=...
```

**Optional but recommended**:
```bash
TELEGRAM_TOKEN=...          # Notifications
TELEGRAM_CHAT_ID=...
USE_TURBO_BATCH=true        # Enable turbo mode
USE_INDEPENDENT_TRADING=true
```

**6.2. Are there any secrets that need rotation?**

No immediate rotation needed. All keys are functional.

**6.3. Is Docker the preferred deployment method?**

**Yes**. Docker Compose is the production standard:
- `docker compose up -d hydra-runtime`
- Full monitoring stack (Prometheus, Grafana, Loki)
- Easy restart/rollback

**6.4. What is the monitoring status?**

Fully operational:
- **Grafana**: http://178.156.136.185:3000 (dashboards working)
- **Prometheus**: http://178.156.136.185:9090 (metrics collection)
- **Loki**: Log aggregation (via Promtail)

---

## 7. Priorities & Next Steps

**7.1. What are the immediate priorities for Builder Claude?**

1. **Fix DOT-USD candle issue** or remove from asset list
2. **Accumulate more trades** - need 20+ for meaningful statistics
3. **Monitor win rate** - currently 47.1%, target 50%+
4. **Test evolution cycles** - kill cycle needs 5 trades

**7.2. Are there any pending tasks?**

- Merge CLAUDE.md streamlining from main to feature branch
- Set up SSH key on 77.42.23.42 for direct GitHub push (done!)
- Test Twilio SMS alerts (optional)

**7.3. What is blocking live FTMO trading?**

1. **Paper trading validation** - need 100+ trades with positive expectancy
2. **MT5 setup** - requires Windows environment
3. **FTMO account** - need funded account
4. **Risk confidence** - Guardian limits need tuning

**7.4. Any scheduled maintenance?**

None scheduled. System is stable.

---

## 8. Context & History

**8.1. What was the journey from V3 → V7 → HYDRA 4.0?**

- **V3-V6**: Single model approaches (LSTM, transformer)
- **V7**: Mathematical theories (Shannon, Hurst, Markov, etc.)
- **HYDRA 3.0**: Multi-engine consensus (4 LLMs)
- **HYDRA 4.0**: Independent trading + tournament + turbo batch

Each version learned from previous limitations.

**8.2. Why were previous systems archived?**

- V7 (11 theories) had good theory but inconsistent signals
- Single-LLM approaches had single points of failure
- HYDRA 4.0 combines best of both: multiple engines + structured output

**8.3. What lessons shaped current architecture?**

1. **Diversity beats single model** - 4 engines disagree productively
2. **Paper trade first** - never go live without 100+ test trades
3. **Kill bad performers** - evolution cycle removes weak engines/strategies
4. **Simple beats complex** - turbo batch > slow consensus

**8.4. Who is Numan? What are their trading goals?**

Numan is the primary user/developer. Goals:
- Pass FTMO challenge (10% profit, <5% daily loss, <10% total loss)
- Automated 24/7 trading system
- Eventually scale to multiple prop firm accounts

---

## 9. Codebase Navigation

**9.1. Which files should Builder Claude read first?**

1. `CLAUDE.md` - System overview
2. `apps/runtime/hydra_runtime.py` - Main runtime loop
3. `libs/hydra/engines/base_engine.py` - Engine interface
4. `libs/hydra/paper_trader.py` - Trade execution
5. `libs/hydra/guardian.py` - Risk management

**9.2. Are there any "gotchas"?**

1. **SQLite vs PostgreSQL**: We use SQLite locally, not RDS
2. **Docker volumes**: Database is inside container, persisted via volume
3. **Env vars in Docker**: Must be in `.env` AND `docker-compose.yml`
4. **MetaTrader5**: Windows-only library, won't import on Linux

**9.3. What testing approach is used?**

- `make test` - All tests
- `make unit` - Unit tests only
- `make smoke` - 5-minute backtest
- Tests are reliable but coverage is ~60%

**9.4. Any files to ignore?**

- `apps/dashboard_flask_backup/` - Deprecated
- `apps/dashboard/` - Old dashboard
- `*.backup*` files
- `ARCHIVED_*.md` docs

---

## 10. Communication Protocol

**10.1. How should Builder Claude and QC Claude coordinate?**

- **Git is source of truth** - always pull before work
- **Q&A files** for async communication (like this one)
- **Direct sync** when both are active

**10.2. Should we use this Q&A format?**

Yes, it works well:
- `BUILDER_CLAUDE_QUESTIONS.md` - Builder asks
- `QC_CLAUDE_ANSWERS.md` - QC answers
- Delete after reviewed to keep repo clean

**10.3. Preferred commit message format?**

```
type: short description

- bullet points if needed

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

**10.4. Other conventions?**

- Always run `make fmt` before commit
- Keep CLAUDE.md updated with major changes
- Use `libs/` for reusable modules
- Use `apps/` for entry points
- Prefer editing existing files over creating new ones

---

**End of Answers**

Builder Claude can now pull this file and continue work. Next action:
```bash
git pull origin feature/v7-ultimate
# Read QC_CLAUDE_ANSWERS.md
# Continue with priorities from section 7.1
```
