# FTMO Live Trading System - Production Documentation

**Status**: VERIFIED WORKING
**Last Test**: 2025-12-09
**Test Result**: Successfully executed and closed 0.01 lot XAUUSD trade (+$0.80 CAD profit)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     HYDRA FTMO Trading System                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────┐         SSH Tunnel          ┌───────────┐ │
│  │  Linux Server       │         (port 15555)        │ Windows   │ │
│  │  77.42.23.42        │◄────────────────────────────►│ VPS       │ │
│  │                     │                              │           │ │
│  │  ┌───────────────┐  │                              │ MT5       │ │
│  │  │ ftmo-runner   │  │         ZMQ REQ/REP          │ Terminal  │ │
│  │  │ (Docker)      │──┼──────────────────────────────►│           │ │
│  │  └───────────────┘  │         (5555)               │ FTMO      │ │
│  │                     │                              │ Server3   │ │
│  │  ┌───────────────┐  │                              └───────────┘ │
│  │  │ Prometheus    │  │                                            │
│  │  │ Grafana       │  │                                            │
│  │  └───────────────┘  │                                            │
│  └─────────────────────┘                                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Critical Credentials

### FTMO Account
| Field | Value |
|-------|-------|
| Login | 531025383 |
| Master Password | h9$K$FpY*1as |
| Server | FTMO-Server3 |
| Account Type | Challenge (Hedge Mode) |
| Balance | ~$14,885 CAD |
| Leverage | 100:1 |

### Windows VPS (ForexVPS)
| Field | Value |
|-------|-------|
| IP | 45.82.167.195 |
| User | trader |
| Password | 80B#^yOr2b5s |

### Linux Server
| Field | Value |
|-------|-------|
| IP | 77.42.23.42 |
| SSH Tunnel | localhost:15555 → Windows:5555 |

---

## Startup Procedure

### Step 1: Windows VPS Setup (Manual - Required)

**IMPORTANT**: This must be done via RDP, NOT SSH. MT5 requires interactive desktop session.

1. Connect to Windows VPS via Remote Desktop:
   - IP: 45.82.167.195
   - User: trader
   - Password: 80B#^yOr2b5s

2. Ensure MT5 Terminal is running and connected to FTMO-Server3

3. **Enable AutoTrading** - Click the "AutoTrading" button in MT5 toolbar (must show green)

4. Run the ZMQ server batch file:
   ```
   Double-click: C:\HYDRA\start_zmq_server.bat
   ```

   Expected output:
   ```
   [INFO] MT5 initialized successfully
   [INFO] Logged in to account 531025383
   [INFO] ZMQ Server listening on tcp://0.0.0.0:5555
   ```

### Step 2: Linux Server - Start SSH Tunnel

```bash
# Create SSH tunnel (run in screen/tmux for persistence)
ssh -N -L 15555:localhost:5555 trader@45.82.167.195
```

Or use autossh for auto-reconnect:
```bash
autossh -M 0 -N -L 15555:localhost:5555 trader@45.82.167.195
```

### Step 3: Verify Connection

```bash
cd /root/crpbot

# Test ping
python3 -c "
import zmq
ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.setsockopt(zmq.RCVTIMEO, 5000)
sock.connect('tcp://127.0.0.1:15555')
sock.send_json({'command': 'PING'})
print(sock.recv_json())
"
```

Expected response:
```json
{"status": "ok", "mt5_connected": true, "account": 531025383}
```

### Step 4: Start FTMO Event Runner

```bash
cd /root/crpbot
docker compose up -d ftmo-runner
```

---

## ZMQ Commands Reference

The ZMQ server on Windows accepts these commands:

| Command | Description | Example |
|---------|-------------|---------|
| PING | Check connection | `{"command": "PING"}` |
| ACCOUNT | Get account info | `{"command": "ACCOUNT"}` |
| PRICE | Get current price | `{"command": "PRICE", "symbol": "XAUUSD"}` |
| TRADE | Execute trade | See below |
| CLOSE | Close position | `{"command": "CLOSE", "ticket": 123456}` |
| POSITIONS | List open positions | `{"command": "POSITIONS"}` |
| CANDLES | Get historical data | `{"command": "CANDLES", "symbol": "XAUUSD", "timeframe": "M5", "count": 100}` |

### Trade Command Example
```json
{
  "command": "TRADE",
  "symbol": "XAUUSD",
  "direction": "BUY",
  "lots": 0.01,
  "sl_pips": 100,
  "tp_pips": 200,
  "comment": "HYDRA_TEST"
}
```

---

## File Locations

### Windows VPS (C:\HYDRA\)
| File | Purpose |
|------|---------|
| start_zmq_server.bat | Startup script with FTMO credentials |
| mt5_zmq_server.py | ZMQ server handling trade commands |
| mt5_price_streamer.py | Price tick streamer (optional) |

### Linux Server (/root/crpbot/)
| File | Purpose |
|------|---------|
| apps/runtime/ftmo_event_runner.py | Main FTMO trading bot runner |
| libs/hydra/ftmo_bots/ | Bot strategies |
| docker-compose.yml | Container orchestration |
| .env | All credentials and configuration |

---

## FTMO Challenge Rules (Critical)

### Hard Limits - DO NOT EXCEED
| Rule | Limit | Current Status |
|------|-------|----------------|
| Daily Loss | 5% max ($743) | Within limits |
| Total Drawdown | 10% max ($1,488) | Within limits |
| Profit Target | 10% ($1,488) | In progress |

### Risk Management Built-In
- Max lot size per trade: Calculated by Guardian
- Position sizing: 1-2% risk per trade
- Stop loss required on all trades
- No overnight news trades

---

## Monitoring

### Grafana Dashboard
- URL: http://77.42.23.42:3001
- User: admin
- Password: hydra4admin

### Prometheus Metrics
- URL: http://77.42.23.42:9090
- FTMO metrics: http://77.42.23.42:9101/metrics

### Check Logs
```bash
# FTMO runner logs
docker logs -f ftmo-runner

# All container status
docker compose ps
```

---

## Troubleshooting

### Issue: ZMQ Connection Timeout
1. Check SSH tunnel is running: `ss -tlnp | grep 15555`
2. Check ZMQ server on Windows (should show "Listening on 5555")
3. Restart SSH tunnel if needed

### Issue: "Trading disabled - investor mode"
**Cause**: Wrong password (investor password instead of master password)
**Fix**: Use master password `h9$K$FpY*1as` in batch file

### Issue: "AutoTrading disabled by client"
**Cause**: AutoTrading button not enabled in MT5
**Fix**: Click AutoTrading button in MT5 toolbar (must be green)

### Issue: MT5 Disconnects When Running via SSH
**Cause**: SSH runs in Service session, can't access interactive desktop
**Fix**: Always run batch file via RDP, not SSH

### Issue: "Invalid account" Error
**Cause**: Wrong server or credentials
**Fix**: Verify FTMO-Server3 and credentials in batch file

---

## Emergency Procedures

### Stop All Trading
```bash
# Stop FTMO runner
docker compose stop ftmo-runner

# Or kill switch via env
export KILL_SWITCH=true
```

### Close All Positions Manually
1. Connect to Windows VPS via RDP
2. In MT5 Terminal, right-click each position → Close

### Full System Restart
```bash
# Linux side
docker compose down
docker compose up -d

# Windows side (via RDP)
1. Close ZMQ server window
2. Restart MT5 Terminal
3. Re-run start_zmq_server.bat
```

---

## Verified Test Results

**Date**: 2025-12-09
**Test Trade**: 0.01 lot XAUUSD BUY

| Step | Result |
|------|--------|
| PING | mt5_connected: True |
| ACCOUNT | Login 531025383, Balance $14,885.65 CAD |
| TRADE | Ticket 100486421, 0.01 lot XAUUSD |
| CLOSE | Closed with +$0.80 CAD profit |

**All systems verified operational.**

---

## Contact / Escalation

- Telegram Bot: Active (TELEGRAM_CHAT_ID in .env)
- SMS Alerts: Configured for critical events (Twilio)
- Email: Backup alerts configured

---

**Document Version**: 1.0
**Created**: 2025-12-09
**Author**: HYDRA Trading System
