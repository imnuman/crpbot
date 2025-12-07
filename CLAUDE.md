# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Dual-Environment Setup

This project operates across **two environments** - identify yours first:

```bash
pwd
# /home/numan/crpbot  -> QC Claude (Local - Development & Testing)
# /root/crpbot        -> Builder Claude (Cloud - Production)
```

| Environment | Path | Role |
|------------|------|------|
| QC Claude | `/home/numan/crpbot` | Development, testing, documentation |
| Builder Claude | `/root/crpbot` | Production runtime, deployment, monitoring |

**Sync Protocol**: Always `git pull origin main` before work, `git push` after.

---

## Architecture Overview

**HYDRA 4.0 = Mother AI + 4 Independent Gladiator Engines**

```
                    MOTHER AI (Orchestrator)
                           │
         ┌─────────┬───────┴───────┬─────────┐
         │         │               │         │
      Engine A  Engine B       Engine C  Engine D
     (DeepSeek) (Claude)       (Grok)   (Gemini)
         │         │               │         │
         └─────────┴───────┬───────┴─────────┘
                           │
                  Tournament Manager → Guardian → Paper Trader → SQLite
```

**Key Design Principles**:
- Each engine trades INDEPENDENTLY with own P&L tracking ($25k portfolio each)
- Tournament ranking by actual P&L: #1=40%, #2=30%, #3=20%, #4=10% weight
- Guardian enforces FTMO-compliant risk limits (4.5% daily, 9% max drawdown)

**Trading Modes** (set via env vars):
- `USE_INDEPENDENT_TRADING=true` (DEFAULT) - Engines trade independently
- `USE_TURBO_BATCH=true` - Batch strategy generation with TurboTournament
- Both false - Consensus voting mode

---

## Quick Commands

### Development
```bash
make setup              # Initial setup (deps + pre-commit hooks)
make fmt                # Format with ruff
make lint               # Lint with ruff + mypy
make test               # All tests
make unit               # Unit tests only
make smoke              # Smoke tests (5-min backtest)

# Run single test
pytest tests/unit/test_ftmo_rules.py -v
pytest tests/unit/test_ftmo_rules.py::test_specific_function -v
```

### HYDRA Runtime
```bash
# Check status
ps aux | grep hydra_runtime | grep -v grep

# Monitor logs
tail -f /tmp/hydra_runtime*.log

# Start (direct)
cd /root/crpbot && source .env
nohup .venv/bin/python apps/runtime/hydra_runtime.py \
  --paper --assets BTC-USD ETH-USD SOL-USD \
  > /tmp/hydra_runtime_$(date +%Y%m%d_%H%M).log 2>&1 &

# Stop
pkill -f hydra_runtime.py
```

### Docker Operations
```bash
# Full stack (runtime + monitoring)
docker compose up -d --build

# View logs
docker logs hydra-runtime 2>&1 | tail -100

# Restart after code changes
docker compose down hydra-runtime && docker compose build hydra-runtime && docker compose up -d hydra-runtime

# Debug inside container
docker exec -it hydra-runtime bash
docker exec hydra-runtime printenv | grep -E 'USE_|API_KEY'
```

### Database Queries
```bash
# Engine performance
sqlite3 data/hydra/hydra.db "
SELECT engine, COUNT(*) as trades,
       SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) as wins,
       ROUND(AVG(pnl_percent), 2) as avg_pnl
FROM hydra_trades WHERE status='CLOSED'
GROUP BY engine ORDER BY avg_pnl DESC;"

# Recent trades
sqlite3 data/hydra/hydra.db "
SELECT timestamp, engine, asset, direction, status, pnl_percent
FROM hydra_trades ORDER BY timestamp DESC LIMIT 20;"

# Tournament leaderboard
sqlite3 data/hydra/hydra.db "SELECT * FROM tournament_scores ORDER BY total_points DESC;"
```

---

## Key Directories

| Path | Purpose |
|------|---------|
| `apps/runtime/hydra_runtime.py` | Main HYDRA orchestrator |
| `libs/hydra/` | Core library (mother_ai, guardian, tournament, paper_trader) |
| `libs/hydra/engines/` | 4 gladiator engines (DeepSeek, Claude, Grok, Gemini) |
| `libs/hydra/cycles/` | Evolution cycles (kill, breeding, weight adjustment) |
| `libs/brokers/` | MT5/FTMO integration |
| `libs/notifications/` | Multi-channel alerts (Telegram, SMS, Email) |
| `libs/data/` | Market data clients (Coinbase, CoinGecko) |
| `data/hydra/` | SQLite database, trade history, lessons |
| `monitoring/` | Grafana + Prometheus configs |
| `.archive/` | Archived systems (v3, v7, old dashboards) |

---

## Critical Concepts

### Guardian Risk Limits
| Rule | Limit | Action |
|------|-------|--------|
| Daily Loss | 4.5% | Stop trading |
| Max Drawdown | 9% | Kill switch |
| Position Size | 1-2% | Auto-adjust |

**Kill Switch**:
```bash
# Emergency stop
export KILL_SWITCH=true  # In .env
sqlite3 data/hydra/hydra.db "UPDATE guardian_state SET kill_switch=1;"
```

### Evolution System
| Mechanism | Interval | Requirements |
|-----------|----------|--------------|
| Tournament Elimination | 24h | 3+ trades per strategy |
| Kill Cycle | 24h | 5 closed trades |
| Breeding Cycle | 4 days | 100 trades, 60% WR, 1.5 Sharpe |

### Production Safety
- `duplicate_order_guard.py` - 5min cooldown, rate limits
- `state_checkpoint.py` - Auto-save every 60s for crash recovery
- All file writes are atomic

---

## Environment Variables

**Required**:
```bash
# Data
COINBASE_API_KEY_NAME=organizations/.../apiKeys/...
COINBASE_API_PRIVATE_KEY=-----BEGIN EC PRIVATE KEY-----...

# LLM APIs
DEEPSEEK_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
XAI_API_KEY=...
GOOGLE_API_KEY=...

# Trading Modes
USE_INDEPENDENT_TRADING=true
USE_TURBO_BATCH=true
KILL_SWITCH=false
```

**Optional**:
```bash
# Notifications
TELEGRAM_TOKEN=...
TELEGRAM_CHAT_ID=...
TWILIO_SID=...           # SMS (critical alerts only)
SMTP_HOST=...            # Email backup

# Live Trading (FTMO)
FTMO_LOGIN=...
FTMO_PASS=...
FTMO_SERVER=FTMO-Demo
EXECUTION_MODE=paper     # paper, live, or shadow
```

---

## Common Issues

| Problem | Solution |
|---------|----------|
| HYDRA not running | `tail -100 /tmp/hydra_runtime*.log`, restart with commands above |
| Database locked | `fuser -k data/hydra/hydra.db` |
| Kill switch triggered | `sqlite3 data/hydra/hydra.db "UPDATE guardian_state SET kill_switch=0;"` |
| API rate limits | Each engine has own key; check quotas |

---

## Monitoring Stack

**Ports**: Grafana `:3000`, Prometheus `:9090`, Metrics `:9100`, Alertmanager `:9093`

```bash
# Access (when running)
http://178.156.136.185:3000   # Grafana dashboards
http://178.156.136.185:9090   # Prometheus queries
```

---

**Last Updated**: 2025-12-07 | **Branch**: `main` | **System**: HYDRA 4.0 (Paper Trading)
