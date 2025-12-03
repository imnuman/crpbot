# Dashboard Quick Reference Card

## Current Status ✅

- **Frontend**: http://178.156.136.185:3000 (Port 3000 - Node.js)
- **Backend**: http://178.156.136.185:8000 (Port 8000 - Python/FastAPI)
- **Process**: Running (PIDs: 2564612, 2564644)
- **Log File**: `/tmp/reflex_clean_restart.log`

---

## Dashboard URLs

| Page | URL |
|------|-----|
| Main Dashboard | http://178.156.136.185:3000/ |
| Performance Tracking | http://178.156.136.185:3000/performance |
| **A/B Test Comparison** | http://178.156.136.185:3000/ab-test |

---

## Quick Commands

### Check Status
```bash
# Quick health check
lsof -i :3000 && lsof -i :8000

# View logs
tail -f /tmp/reflex_clean_restart.log
```

### Restart Dashboard
```bash
# Kill existing
pkill -9 -f "reflex"

# Start fresh
cd /root/crpbot/apps/dashboard_reflex
rm -rf /root/crpbot/**/__pycache__ 2>/dev/null
../../.venv/bin/python3 -m reflex run > /tmp/reflex_dashboard.log 2>&1 &
```

### Debug Issues
```bash
# Check backend errors
tail -100 /tmp/reflex_clean_restart.log | grep -i error

# Test database
sqlite3 /root/crpbot/tradingai.db "SELECT COUNT(*) FROM signals"

# Check ports
ss -tlnp | grep -E ':(3000|8000)'
```

---

## Common Issues

### "WebSocket Error"
**Fix**: Backend (port 8000) crashed
```bash
cd /root/crpbot/apps/dashboard_reflex
../../.venv/bin/python3 -m reflex run
```

### "No Data Showing"
**Fix**: Check V7 runtime is generating signals
```bash
tail -50 /tmp/v7_ab_testing_production.log
```

### "Port Already in Use"
**Fix**: Kill existing process
```bash
pkill -9 -f "reflex"
```

---

## Documentation

- **Full Backend Guide**: `/root/crpbot/REFLEX_DASHBOARD_BACKEND_GUIDE.md`
- **Dashboard Code**: `/root/crpbot/apps/dashboard_reflex/dashboard_reflex/dashboard_reflex.py`
- **A/B Testing Doc**: `/root/crpbot/AB_TEST_IMPLEMENTATION_STATUS.md`

---

## Support Checklist

Before reporting issues:
- [ ] Port 3000 listening? (`lsof -i :3000`)
- [ ] Port 8000 listening? (`lsof -i :8000`)
- [ ] Any errors in logs? (`tail -100 /tmp/reflex_clean_restart.log`)
- [ ] Database accessible? (`sqlite3 tradingai.db ".tables"`)
- [ ] V7 runtime running? (`ps aux | grep v7_runtime`)
