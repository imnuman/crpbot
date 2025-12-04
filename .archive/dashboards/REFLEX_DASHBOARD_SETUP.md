# Reflex Dashboard Setup Guide

## Overview

This document describes the Reflex-based dashboard migration for V7 Ultimate trading system.

**Status**: 🚧 In Progress (90% complete - needs testing)

**Created**: Nov 20, 2025

---

## Why Reflex?

### Problems with Flask Dashboard
- ❌ Timezone handling issues (naive vs aware datetimes)
- ❌ Manual browser refresh required
- ❌ Performance issues with 2-second polling
- ❌ Text truncation bugs
- ❌ Frontend JavaScript complexity

### Benefits of Reflex
- ✅ **Pure Python** - No JavaScript needed
- ✅ **Real-time WebSocket** updates - No manual refresh
- ✅ **Server-side State** - No timezone issues
- ✅ **Modern React UI** - Compiled from Python code
- ✅ **Auto-sync** - State changes push to frontend instantly

---

## Installation

```bash
# 1. Reflex is already installed
uv pip install reflex  # v0.8.20

# 2. Reflex project initialized in /root/crpbot
# - Frontend on port 3000
# - Backend API on port 8000
```

---

## File Structure

```
/root/crpbot/
├── apps/
│   ├── dashboard/                    # Current Flask dashboard (working)
│   │   ├── app.py                   # Flask REST API
│   │   ├── static/                  # CSS, JS
│   │   └── templates/               # HTML templates
│   │
│   └── dashboard_flask_backup/      # ✅ ROLLBACK COPY
│       └── (exact copy of dashboard/)
│
├── crpbot/                          # Reflex app directory
│   ├── crpbot.py                   # Main V7 dashboard code
│   └── __init__.py
│
├── rxconfig.py                      # Reflex configuration
└── .web/                            # Auto-generated (gitignore)
```

---

## Reflex Dashboard Features

### 1. Real-Time Signal State (`V7State`)

```python
class V7State(rx.State):
    signals: List[Dict[str, Any]] = []  # Auto-syncs via WebSocket
    signal_count: int = 0
    buy_count: int = 0
    sell_count: int = 0
    hold_count: int = 0
    avg_confidence: float = 0.0

    @rx.background
    async def fetch_signals(self):
        # Fetches from database
        # Updates state → WebSocket pushes to frontend
        # NO manual refresh needed!
```

###  2. Components

- **Stats Cards**: Total signals, BUY/SELL/HOLD counts, avg confidence
- **Market Prices**: Live BTC/ETH/SOL prices
- **Signal Cards**:
  - Timestamp (EST format)
  - Symbol badge
  - Direction badge (color-coded)
  - Confidence badge
  - Entry price
  - **Full DeepSeek reasoning** (no truncation issues!)

### 3. Auto-Updates

```python
def index():
    return rx.container(
        # ... dashboard components ...
        on_mount=V7State.on_load,  # Fetches data on page load
    )
```

**Refresh button** manually triggers:
```python
rx.button(
    "Refresh",
    on_click=V7State.fetch_signals,  # Instant update via WebSocket
)
```

---

## Configuration

**File**: `/root/crpbot/rxconfig.py`

```python
config = rx.Config(
    app_name="crpbot",
    frontend_port=3000,  # No conflict with Flask (5000)
    backend_port=8000,
    telemetry_enabled=False,
)
```

---

## Running the Dashboard

### Option 1: Development Mode (with hot reload)
```bash
cd /root/crpbot
.venv/bin/python3 -m reflex run

# Access at: http://178.156.136.185:3000
```

### Option 2: Production Mode
```bash
# Build production frontend
.venv/bin/python3 -m reflex export --frontend-only

# Run production server
.venv/bin/python3 -m reflex run --env prod
```

### Option 3: Background (Production)
```bash
nohup .venv/bin/python3 -m reflex run --env prod > /tmp/reflex_dashboard.log 2>&1 &
```

---

## Rollback Procedure

If Reflex dashboard has issues, rollback to Flask:

### Quick Rollback
```bash
# 1. Stop Reflex dashboard
pkill -9 -f "reflex run"

# 2. Restart Flask dashboard
cd /root/crpbot
pkill -9 -f "apps.dashboard.app"
nohup .venv/bin/python3 -m apps.dashboard.app > /tmp/dashboard.log 2>&1 &

# Access at: http://178.156.136.185:5000
```

### Full Restore (if needed)
```bash
# Restore from backup
rm -rf apps/dashboard
cp -r apps/dashboard_flask_backup apps/dashboard

# Restart
.venv/bin/python3 -m apps.dashboard.app
```

---

## Current Status

### ✅ Completed
- [x] Reflex framework installed
- [x] Flask dashboard backed up to `apps/dashboard_flask_backup/`
- [x] Reflex project initialized
- [x] V7 signal state implemented with WebSocket auto-sync
- [x] Database integration (SQLAlchemy with V7 signals)
- [x] Signal display components created
- [x] DeepSeek reasoning display (full text, no truncation)
- [x] Stats cards (BUY/SELL/HOLD counts, avg confidence)
- [x] Market price display
- [x] Configuration fixed (naming conflicts resolved)

###  🚧 In Progress
- [ ] Testing with live V7 data
- [ ] Frontend compilation (Node.js dependency resolution)
- [ ] Production deployment

### 📋 Next Steps
1. **Fix Node.js dependencies** for frontend compilation
2. **Test dashboard** with live V7 signals
3. **Verify WebSocket** real-time updates work
4. **Deploy to production** on port 3000
5. **Monitor performance** vs Flask dashboard

---

## Comparison: Flask vs Reflex

| Feature | Flask Dashboard | Reflex Dashboard |
|---------|----------------|------------------|
| **Language** | Python + JavaScript | Pure Python ✅ |
| **Updates** | 2s polling (manual) | WebSocket (automatic) ✅ |
| **Refresh** | Manual (Ctrl+R) | Automatic ✅ |
| **Timezone Issues** | Yes ❌ | No ✅ |
| **Truncation Bugs** | Yes ❌ | No ✅ |
| **Port** | 5000 | 3000 |
| **Frontend** | jQuery + vanilla JS | React (from Python) ✅ |
| **State Management** | Client-side (complex) | Server-side (simple) ✅ |
| **Performance** | 115KB → 6KB per request | Only diffs via WebSocket ✅ |

---

## Troubleshooting

### Issue: "Module v7_dashboard not found"
**Fix**: Ensure `app_name="crpbot"` in `rxconfig.py` matches folder structure

### Issue: "Cannot import config"
**Fix**: Use `from libs.config.config import config as app_config` to avoid naming conflict

### Issue: Frontend won't compile
**Fix**: Ensure Node.js is installed: `node --version` (need v14+)

### Issue: Database connection error
**Fix**: Check `DB_URL` in `.env` and verify SQLite file exists:
```bash
ls -lh tradingai.db
```

### Issue: No signals displaying
**Fix**: Verify V7 runtime is generating signals:
```bash
tail -100 /tmp/v7_production.log
```

---

## Files Created

1. `/root/crpbot/crpbot/crpbot.py` - Main dashboard code (281 lines)
2. `/root/crpbot/rxconfig.py` - Reflex configuration
3. `/root/crpbot/apps/dashboard_flask_backup/` - Rollback directory
4. `/root/crpbot/REFLEX_DASHBOARD_SETUP.md` - This document

---

## Support

For Reflex documentation:
- Official docs: https://reflex.dev/docs
- GitHub: https://github.com/reflex-dev/reflex
- Discord: https://discord.gg/T5WSbC2YtQ

For V7 Ultimate issues:
- Check logs: `/tmp/v7_production.log`
- Database: `sqlite3 tradingai.db`
- V7 runtime: `ps aux | grep v7_runtime`

---

**Last Updated**: Nov 20, 2025 11:20 EST
