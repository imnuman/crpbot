# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## CRITICAL: Dual-Environment Setup

This project operates across **two environments** - identify yours first:

```bash
pwd
# /home/numan/crpbot  -> QC Claude (Local - Quality Control & Development)
# /root/crpbot        -> Builder Claude (Cloud - Production & Deployment)
```

### QC Claude (Local: `/home/numan/crpbot`)
- **Quality Control**: Review before deployment
- **Development**: Code changes and testing
- **Documentation**: Maintain CLAUDE.md, handoff docs
- **Testing**: Local verification

### Builder Claude (Cloud: `root@178.156.136.185`)
- **Production Runtime**: HYDRA 4.0 running 24/7
- **Deployment**: Primary deployment and monitoring
- **Database**: SQLite (`/root/crpbot/data/hydra/hydra.db`)

### Sync Protocol
```bash
# ALWAYS sync before work
git pull origin main

# Push changes after work
git add . && git commit -m "message" && git push origin main
```

---

## HYDRA 4.0 - Current Production System

**STATUS**: OPERATIONAL - Paper trading mode
**Last Updated**: 2025-12-03
**Dashboard**: Grafana (Coming Soon - Port 3000)

### Architecture Overview

**HYDRA 4.0 = Mother AI + 4 Independent Gladiator Engines**

```
                    MOTHER AI (Orchestrator)
                           |
         +-----------------+-----------------+
         |        |        |        |        |
      Engine A  Engine B  Engine C  Engine D
     (DeepSeek) (Claude)  (Grok)   (Gemini)
         |        |        |        |
    Independent P&L Tracking Per Engine
                           |
                    Tournament Manager
                    (Rankings & Weights)
                           |
                      Guardian
                    (Risk Control)
                           |
                    Paper Trader
                    (Execution)
```

**L1 - Mother AI** (Supreme Orchestrator):
- Tournament management and rankings
- Final trade approval with FTMO governance
- Weight adjustment (24-hour cycle)
- Breeding mechanism (4-day cycle)
- Emergency shutdown authority

**L2 - 4 Gladiator Engines** (Independent Competitors):
- Engine A (DeepSeek) - Structural edge hunter
- Engine B (Claude) - Logic validator
- Engine C (Grok) - Historical pattern matcher
- Engine D (Gemini) - Synthesis engine

**Key Design Principles**:
- Each engine trades INDEPENDENTLY (not consensus voting)
- Each engine has OWN P&L tracking and portfolio
- Parallel execution: A||B||C||D
- Tournament ranking by actual P&L performance
- Weight distribution: #1=40%, #2=30%, #3=20%, #4=10%

### Key Files

**Runtime**:
- `apps/runtime/hydra_runtime.py` - Main HYDRA runtime (33KB)
- `apps/runtime/hydra_guardian.py` - Risk management

**Core Library** (`libs/hydra/`):
- `mother_ai.py` - L1 orchestrator
- `guardian.py` - Risk control and kill switch
- `tournament_manager.py` - Rankings and weights
- `tournament_tracker.py` - Performance tracking
- `paper_trader.py` - Trade execution
- `engine_portfolio.py` - Per-engine P&L tracking

**Engines** (`libs/hydra/engines/`):
- `base_engine.py` - Abstract base class
- `engine_a_deepseek.py` - DeepSeek integration
- `engine_b_claude.py` - Claude integration
- `engine_c_grok.py` - Grok integration
- `engine_d_gemini.py` - Gemini integration

**Data Feeds** (`libs/hydra/data_feeds/`):
- `market_data_feeds.py` - Funding, liquidations
- `orderbook_feed.py` - Order book analysis
- `internet_search.py` - News & edge discovery

**Safety** (`libs/hydra/safety/`):
- Risk validators and safety checks

---

## Quick Commands

### Development
```bash
make setup              # Initial setup (deps + pre-commit hooks)
make fmt                # Format with ruff
make lint               # Lint with ruff + mypy
make test               # All tests
make unit               # Unit tests only
```

### HYDRA Runtime (Production)
```bash
# Check HYDRA status
ps aux | grep hydra_runtime | grep -v grep

# Monitor logs
tail -f /tmp/hydra_runtime*.log

# Start HYDRA (paper trading mode)
cd /root/crpbot && source .env
nohup .venv/bin/python apps/runtime/hydra_runtime.py \
  --paper \
  --assets BTC-USD ETH-USD SOL-USD \
  > /tmp/hydra_runtime_$(date +%Y%m%d_%H%M).log 2>&1 &

# Stop HYDRA
pkill -f hydra_runtime.py
```

### Database Operations
```bash
# HYDRA database location
# Local: /home/numan/crpbot/data/hydra/hydra.db
# Cloud: /root/crpbot/data/hydra/hydra.db

# Check engine performance
sqlite3 data/hydra/hydra.db "
SELECT engine,
       COUNT(*) as trades,
       SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) as wins,
       ROUND(AVG(pnl_percent), 2) as avg_pnl
FROM hydra_trades
WHERE status='CLOSED'
GROUP BY engine
ORDER BY avg_pnl DESC;"

# Recent trades
sqlite3 data/hydra/hydra.db "
SELECT timestamp, engine, asset, direction, status, pnl_percent
FROM hydra_trades
ORDER BY timestamp DESC LIMIT 20;"

# Tournament leaderboard
sqlite3 data/hydra/hydra.db "
SELECT * FROM tournament_scores ORDER BY total_points DESC;"
```

---

## Project Structure

```
crpbot/
├── apps/
│   ├── runtime/
│   │   ├── hydra_runtime.py      # Main HYDRA orchestrator
│   │   ├── hydra_guardian.py     # Risk management
│   │   └── telegram_bot.py       # Notifications
│   └── trainer/                  # Model training (AWS GPU only)
├── libs/
│   ├── hydra/                    # HYDRA 4.0 core
│   │   ├── mother_ai.py          # L1 orchestrator
│   │   ├── guardian.py           # Risk control
│   │   ├── tournament_manager.py # Rankings
│   │   ├── tournament_tracker.py # Performance tracking
│   │   ├── paper_trader.py       # Trade execution
│   │   ├── engine_portfolio.py   # Per-engine P&L
│   │   ├── engines/              # 4 gladiator engines
│   │   ├── data_feeds/           # Market data
│   │   ├── safety/               # Validators
│   │   └── cycles/               # Kill cycle, breeding
│   ├── data/                     # Data clients
│   │   ├── coinbase_client.py
│   │   └── coingecko_client.py
│   ├── db/                       # SQLAlchemy models
│   └── config/                   # Pydantic settings
├── data/
│   └── hydra/                    # HYDRA data directory
│       ├── hydra.db              # Main database
│       ├── paper_trades.jsonl    # Trade history
│       ├── tournament_*.jsonl    # Tournament data
│       └── lessons/              # Engine lessons learned
├── monitoring/                   # Grafana + Prometheus (coming)
│   ├── grafana/
│   └── prometheus/
├── .archive/                     # Archived systems
│   ├── v7/                       # V7 Ultimate (archived)
│   ├── dashboards/               # Old dashboards (ttyd, reflex)
│   └── v3_ultimate/              # V3 system (archived)
└── tests/
```

---

## Data Flow

```
Market Data (Coinbase API)
           |
    +------+------+
    |      |      |
  BTC    ETH    SOL
    |      |      |
    +------+------+
           |
    MOTHER AI (Orchestrator)
           |
    +------+------+------+
    |      |      |      |
 Engine  Engine Engine Engine
    A      B      C      D
    |      |      |      |
    +------+------+------+
           |
    Tournament Manager
    (Score & Rank)
           |
    Guardian (Risk Check)
           |
    Paper Trader
    (Execute Trade)
           |
    SQLite Database
           |
    +------+------+
    |             |
 Telegram      Grafana
(Alerts)    (Dashboard)
```

---

## Critical Concepts

### Tournament System

**Scoring Rules**:
- Correct prediction (vote matches trade outcome): +1 point
- Wrong prediction: 0 points
- HOLD vote: 0 points (neutral)

**Weight Distribution** (by ranking):
- #1: 40% weight
- #2: 30% weight
- #3: 20% weight
- #4: 10% weight

### Guardian (Risk Control)

**Hard Limits**:
- Daily loss: 4.5% max
- Total drawdown: 9% max
- Position sizing: Risk-based (1-2% per trade)

**Kill Switch**:
```bash
# Emergency stop
export KILL_SWITCH=true  # In .env
# Or via Guardian
sqlite3 data/hydra/hydra.db "UPDATE guardian_state SET kill_switch=1;"
```

### Paper Trading

**Trade Flow**:
1. Engine generates signal (BUY/SELL/HOLD)
2. Mother AI approves via tournament consensus
3. Guardian validates risk limits
4. Paper Trader records entry
5. Monitor for TP/SL hit
6. Record outcome and score

**Trade States**:
- OPEN: Position active
- CLOSED: Position exited (win/loss)
- EXPIRED: Timeout reached

---

## Environment Variables (.env)

**Required**:
```bash
# Data Provider
DATA_PROVIDER=coinbase
COINBASE_API_KEY_NAME=organizations/.../apiKeys/...
COINBASE_API_PRIVATE_KEY=-----BEGIN EC PRIVATE KEY-----...

# LLM APIs (for 4 engines)
DEEPSEEK_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
XAI_API_KEY=...
GOOGLE_API_KEY=...

# Premium APIs
COINGECKO_API_KEY=CG-...

# Safety
KILL_SWITCH=false
```

**Optional**:
```bash
# Telegram (notifications)
TELEGRAM_TOKEN=...
TELEGRAM_CHAT_ID=...
```

---

## Monitoring (Grafana)

**Status**: Coming Soon (In Development)

**Architecture**:
```
HYDRA Runtime
      |
      v
Prometheus Exporter (:9090/metrics)
      |
      v
Prometheus (scrape every 15s)
      |
      v
Grafana Dashboards (:3000)
      |
      v
Alertmanager -> Telegram
```

**Dashboards** (Planned):
1. **Overview**: P&L, win rate, prices, engine tournament
2. **Engine Performance**: Per-engine stats and trends
3. **Risk & Safety**: Drawdown, guardian status, limits
4. **Trades & Signals**: Trade history, recent signals

---

## Common Pitfalls

### 1. HYDRA Not Running
```bash
# Check status
ps aux | grep hydra_runtime | grep -v grep

# Check logs
tail -100 /tmp/hydra_runtime*.log

# Restart
pkill -f hydra_runtime.py
cd /root/crpbot && source .env
nohup .venv/bin/python apps/runtime/hydra_runtime.py --paper --assets BTC-USD ETH-USD SOL-USD > /tmp/hydra_runtime.log 2>&1 &
```

### 2. Database Locked
```bash
# SQLite lock issues
fuser -k data/hydra/hydra.db  # Kill processes holding lock
```

### 3. API Rate Limits
- Each engine has its own API key
- DeepSeek: ~$0.0003/call
- Claude: ~$0.002/call
- Grok: ~$0.001/call
- Gemini: Free tier available

### 4. Kill Switch Triggered
```bash
# Check Guardian state
sqlite3 data/hydra/hydra.db "SELECT * FROM guardian_state;"

# Reset kill switch
sqlite3 data/hydra/hydra.db "UPDATE guardian_state SET kill_switch=0;"
```

---

## Archived Systems

**Location**: `.archive/`

- `.archive/v7/` - V7 Ultimate (11 theories + DeepSeek LLM)
- `.archive/dashboards/ttyd/` - Terminal dashboard (Rich + ttyd)
- `.archive/dashboards/reflex/` - Reflex web dashboard
- `.archive/v3_ultimate/` - V3 system

These systems are preserved for reference but no longer in production.

---

## Key Documentation

- `CLAUDE.md` - This file (main reference)
- `HYDRA_3.0_IMPLEMENTATION_PLAN.md` - HYDRA design blueprint
- `README.md` - Project overview
- `Makefile` - Available commands

---

## Current Status (December 2025)

**Production System**: HYDRA 4.0
- Mother AI + 4 Independent Engines
- Paper trading on BTC-USD, ETH-USD, SOL-USD
- Tournament-based engine ranking
- Guardian risk control active

**Infrastructure**:
- Runtime: Cloud server (178.156.136.185)
- Database: SQLite (`data/hydra/hydra.db`)
- Monitoring: Grafana (Coming Soon)

**Next Milestones**:
1. Complete Grafana + Prometheus monitoring stack
2. Dashboard: P&L, engine tournament, risk metrics
3. Alerting: Telegram notifications for critical events

---

**Last Updated**: 2025-12-03
**Branch**: `feature/v7-ultimate`
