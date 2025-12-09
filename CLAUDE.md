# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Multi-Server Infrastructure

This project operates across **four servers**:

| Server | IP | OS | Role |
|--------|-----|-----|------|
| **US Production** | 178.156.136.185 | Linux | FTMO trading (low latency to Windows VPS) |
| **Finland Dev** | 77.42.23.42 | Linux | Grafana dashboard, development, git repo |
| **Windows VPS** | 45.82.167.195 | Windows | MT5 terminal, ZMQ executor |
| **Local Dev** | `/home/numan/crpbot` | Linux | QC Claude, testing |

### Server Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     FTMO Trading System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   ┌──────────────────┐     SSH/ZMQ     ┌──────────────────┐     │
│   │ US Production    │ ◄─────────────► │ Windows VPS      │     │
│   │ 178.156.136.185  │     tunnel      │ 45.82.167.195    │     │
│   │                  │                  │                  │     │
│   │ • FTMO Runner    │                  │ • MT5 Terminal   │     │
│   │ • 6 Trading Bots │                  │ • ZMQ Server     │     │
│   │ • Metalearning   │                  │ • Price Streamer │     │
│   └──────────────────┘                  └──────────────────┘     │
│           │                                                       │
│           │ rsync                                                 │
│           ▼                                                       │
│   ┌──────────────────┐                                           │
│   │ Finland Dev      │                                           │
│   │ 77.42.23.42      │                                           │
│   │                  │                                           │
│   │ • Grafana :3001  │                                           │
│   │ • Prometheus     │                                           │
│   │ • Git repo       │                                           │
│   └──────────────────┘                                           │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Credentials & Access

**US Production** (root@178.156.136.185):
```bash
ssh root@178.156.136.185
# FTMO runner location: /root/crpbot
```

**Finland Dev** (root@77.42.23.42):
```bash
ssh root@77.42.23.42
# Grafana: http://77.42.23.42:3001 (admin / hydra2024)
```

**Windows VPS** (trader@45.82.167.195):
```bash
# SSH access (requires sshpass)
sshpass -p '80B#^yOr2b5s' ssh trader@45.82.167.195
# MT5 ZMQ Server: C:\HYDRA\mt5_zmq_server.py
# Price Streamer: C:\HYDRA\mt5_price_streamer.py
```

### Sync Protocol

```bash
# On Finland (source of truth):
git pull origin main  # Before work
git push origin main  # After work

# Sync to US Production:
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='.venv' \
  /root/crpbot/libs/ root@178.156.136.185:/root/crpbot/libs/
rsync -avz /root/crpbot/apps/ root@178.156.136.185:/root/crpbot/apps/
```

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

## FTMO Live Trading System

### ZMQ Communication Architecture

```
Windows MT5 Terminal
        │
        ├──► mt5_zmq_server.py (Port 5555) ──► SSH Tunnel ──► Linux :15555
        │    (REQ/REP - Trade Execution)
        │
        └──► mt5_price_streamer.py (Port 5556) ──► SSH Tunnel ──► Linux :15556
             (PUB/SUB - Real-time Ticks)
```

**Port Reference**:
| Port | Protocol | Direction | Purpose |
|------|----------|-----------|---------|
| 5555 | ZMQ REQ/REP | Windows → Linux | Trade commands (PING, ACCOUNT, TRADE, CLOSE) |
| 5556 | ZMQ PUB/SUB | Windows → Linux | Price ticks (XAUUSD, EURUSD, etc.) |
| 15555 | Local | Linux | SSH tunnel to Windows:5555 |
| 15556 | Local | Linux | SSH tunnel to Windows:5556 |
| 9100 | HTTP | Linux | Prometheus metrics endpoint |
| 9101 | HTTP | Docker host | FTMO metrics (maps to 9100) |

### ZMQ Message Format

**IMPORTANT**: ZMQ server uses `cmd` field (not `command`):
```json
{"cmd": "PING"}
{"cmd": "ACCOUNT"}
{"cmd": "PRICE", "symbol": "XAUUSD"}
{"cmd": "TRADE", "symbol": "XAUUSD", "direction": "BUY", "volume": 0.01, "sl": 2650.0, "tp": 2700.0, "comment": "HYDRA_GoldLondon"}
{"cmd": "CLOSE", "ticket": 123456}
{"cmd": "POSITIONS"}
{"cmd": "CANDLES", "symbol": "XAUUSD", "timeframe": "M1", "count": 100}
```

### 6 FTMO Trading Bots

| Bot | Symbol | Strategy | Expected Daily P&L |
|-----|--------|----------|-------------------|
| Gold London Reversal | XAUUSD | Fade Asian trends at London open | $184 (61% WR) |
| EUR/USD Breakout | EURUSD | Daily S/R level breakouts | $172 |
| US30 ORB | US30 | Opening range breakout fade | $148 |
| NAS100 Gap | NAS100 | Gap fill at market open | $150 |
| Gold NY Reversion | XAUUSD | VWAP reversion during NY session | $148 |
| HF Scalper | MULTI | High-frequency momentum scalping | $66 (52% WR) |

### FTMO Bot Files

| File | Purpose |
|------|---------|
| `libs/hydra/ftmo_bots/base_ftmo_bot.py` | Abstract base class with ZMQ communication |
| `libs/hydra/ftmo_bots/orchestrator.py` | Master orchestrator, unified risk management |
| `libs/hydra/ftmo_bots/event_bus.py` | ZMQ SUB socket for real-time price streaming |
| `libs/hydra/ftmo_bots/event_bot_wrapper.py` | Event-driven wrapper with tick buffering |
| `libs/hydra/ftmo_bots/metrics.py` | Prometheus exporter (port 9100) |
| `libs/hydra/ftmo_bots/metalearning.py` | L1 adaptive sizing, L2 volatility detection |
| `libs/brokers/mt5_zmq_client.py` | ZMQ REQ/REP client for trade execution |

### FTMO Runner Commands

```bash
# Start ftmo-runner (Docker - Recommended)
docker compose up -d ftmo-runner
docker logs -f ftmo-runner

# Check FTMO metrics
curl http://localhost:9101/metrics | grep ftmo_

# Test ZMQ connection
python3 -c "
import zmq
ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.setsockopt(zmq.RCVTIMEO, 5000)
sock.connect('tcp://127.0.0.1:15555')
sock.send_json({'cmd': 'PING'})
print(sock.recv_json())
"

# Manual runner (non-Docker)
python apps/runtime/ftmo_event_runner.py --turbo --metrics-port 9100
```

### Windows VPS Setup (Required - Manual via RDP)

**CRITICAL**: MT5 requires interactive desktop session. Run via RDP, NOT SSH.

1. Connect via Remote Desktop: `45.82.167.195` / `trader` / `80B#^yOr2b5s`
2. Start MT5 Terminal, enable AutoTrading (green button)
3. Run ZMQ Server: `C:\HYDRA\start_zmq_server.bat`
4. Run Price Streamer: `C:\HYDRA\start_price_streamer.bat`

**Batch Files on Windows**:
```batch
# C:\HYDRA\start_zmq_server.bat
@echo off
cd /d C:\HYDRA
set FTMO_LOGIN=531025383
set FTMO_PASS=h9$K$FpY*1as
set FTMO_SERVER=FTMO-Server3
"C:\Program Files\Python311\python.exe" mt5_zmq_server.py

# C:\HYDRA\start_price_streamer.bat
@echo off
cd /d C:\HYDRA
set FTMO_LOGIN=531025383
set FTMO_PASS=h9$K$FpY*1as
set FTMO_SERVER=FTMO-Server3
"C:\Program Files\Python311\python.exe" mt5_price_streamer.py
```

### SSH Tunnel Setup (Linux Side)

```bash
# Create SSH tunnels for ZMQ communication
# Trade commands tunnel (REQ/REP)
ssh -N -L 15555:localhost:5555 trader@45.82.167.195 &

# Price streaming tunnel (PUB/SUB)
ssh -N -L 15556:localhost:5556 trader@45.82.167.195 &

# Or use autossh for auto-reconnect
autossh -M 0 -N -L 15555:localhost:5555 trader@45.82.167.195 &
autossh -M 0 -N -L 15556:localhost:5556 trader@45.82.167.195 &
```

### FTMO Challenge Rules

| Rule | Limit | Action |
|------|-------|--------|
| Daily Loss | 5% max (~$743 on $15k) | Stop trading |
| Total Drawdown | 10% max (~$1,488) | Kill switch |
| Profit Target | 10% (~$1,488) | Challenge passed |

**Built-in Risk Management**:
- 1.5% risk per trade (1% for HF scalper)
- Max 3 concurrent positions
- Position correlation limits (>0.7 prevented)
- Stop loss required on all trades

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
docker logs ftmo-runner 2>&1 | tail -100

# Restart after code changes
docker compose down hydra-runtime && docker compose build hydra-runtime && docker compose up -d hydra-runtime
docker compose down ftmo-runner && docker compose build ftmo-runner && docker compose up -d ftmo-runner

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
| `apps/runtime/ftmo_event_runner.py` | FTMO event-driven runner (22.5KB) |
| `apps/runtime/ftmo_runner.py` | FTMO simple orchestrator runner |
| `libs/hydra/` | Core library (mother_ai, guardian, tournament, paper_trader) |
| `libs/hydra/engines/` | 4 gladiator engines (DeepSeek, Claude, Grok, Gemini) |
| `libs/hydra/ftmo_bots/` | 6 FTMO trading bots + event bus + metalearning |
| `libs/hydra/cycles/` | Evolution cycles (kill, breeding, weight adjustment) |
| `libs/brokers/` | MT5/FTMO integration (mt5_zmq_client.py) |
| `libs/notifications/` | Multi-channel alerts (Telegram, SMS, Email) |
| `libs/data/` | Market data clients (Coinbase, CoinGecko) |
| `data/hydra/` | SQLite database, trade history, lessons |
| `data/hydra/ftmo/` | FTMO metalearning persistence |
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

### Metalearning System
- **L1**: Adaptive position sizing (20-trade rolling window)
- **L2**: Volatility regime detection (low/medium/high)
- Persistence: `/data/hydra/ftmo/trade_history.jsonl`

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

**FTMO Trading**:
```bash
# FTMO Account
FTMO_LOGIN=531025383
FTMO_PASS=h9$K$FpY*1as
FTMO_SERVER=FTMO-Server3

# Windows VPS
WINDOWS_VPS_IP=45.82.167.195
WINDOWS_VPS_USER=trader
WINDOWS_VPS_PASS=80B#^yOr2b5s

# Execution Mode
EXECUTION_MODE=paper     # paper, live, or shadow
```

**Optional**:
```bash
# Notifications
TELEGRAM_TOKEN=...
TELEGRAM_CHAT_ID=...
TWILIO_SID=...           # SMS (critical alerts only)
SMTP_HOST=...            # Email backup
```

---

## Common Issues

| Problem | Solution |
|---------|----------|
| HYDRA not running | `tail -100 /tmp/hydra_runtime*.log`, restart with commands above |
| Database locked | `fuser -k data/hydra/hydra.db` |
| Kill switch triggered | `sqlite3 data/hydra/hydra.db "UPDATE guardian_state SET kill_switch=0;"` |
| API rate limits | Each engine has own key; check quotas |
| ZMQ connection timeout | Check SSH tunnel: `ss -tlnp \| grep 15555` |
| "Unknown command" ZMQ error | Use `cmd` field not `command`: `{"cmd": "PING"}` |
| MT5 "investor mode" error | Use master password, not investor password |
| MT5 AutoTrading disabled | Click AutoTrading button in MT5 (must be green) |
| No ticks received | Restart price streamer via RDP on Windows |

---

## Monitoring Stack

**Ports**: Grafana `:3001`, Prometheus `:9090`, FTMO Metrics `:9101`, Alertmanager `:9093`

```bash
# Access (when running)
http://178.156.136.185:3001   # Grafana dashboards (admin/hydra4admin)
http://178.156.136.185:9090   # Prometheus queries
http://178.156.136.185:9101/metrics  # FTMO metrics

# Start monitoring stack
cd /root/crpbot/monitoring && docker compose up -d
```

**FTMO Prometheus Metrics**:
- `ftmo_connection_status` - MT5/Event bus status
- `ftmo_tick_rate` - Ticks/second by symbol
- `ftmo_bot_signals_total` - Signals by bot/direction
- `ftmo_bot_trades_total` - Trades by outcome
- `ftmo_bot_pnl_dollars` - Cumulative P&L
- `ftmo_account_balance_dollars` - Current balance
- `ftmo_daily_drawdown_percent` - Today's DD
- `ftmo_kill_switch_active` - Emergency stop status

---

## Troubleshooting FTMO

### ZMQ Connection Issues

```bash
# Test ZMQ server connectivity
python3 -c "
import zmq
ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.setsockopt(zmq.RCVTIMEO, 5000)
sock.connect('tcp://127.0.0.1:15555')
sock.send_json({'cmd': 'PING'})
print(sock.recv_json())
"

# Check SSH tunnel
ss -tlnp | grep 15555

# Restart SSH tunnel
pkill -f 'ssh -N -L 15555'
ssh -N -L 15555:localhost:5555 trader@45.82.167.195 &
```

### No Price Ticks

1. Check Windows VPS via RDP
2. Verify MT5 is connected (green icon bottom right)
3. Restart price streamer: `C:\HYDRA\start_price_streamer.bat`
4. Check ftmo-runner logs: `docker logs -f ftmo-runner`

### Emergency Stop

```bash
# Stop FTMO runner
docker compose stop ftmo-runner

# Or kill switch via env
export KILL_SWITCH=true

# Close all positions manually via MT5 on Windows
```

---

**Last Updated**: 2025-12-09 | **Branch**: `main` | **System**: HYDRA 4.0 + FTMO Live Trading
