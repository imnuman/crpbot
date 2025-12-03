# Dashboard Troubleshooting - Port 8000 WebSocket Issue

**Date**: 2025-11-22
**Issue**: Cannot connect to server: websocket error at ws://178.156.136.185:8000/_event

---

## 🎯 Port Allocation for CRPBot

| Application | Port | Purpose |
|-------------|------|---------|
| **Reflex Frontend** | 3000 | Main dashboard UI |
| **Flask Dashboard** | 5000 | Legacy V6 dashboard (deprecated) |
| **Reflex Backend** | 8000 | WebSocket API for Reflex |

**Port 8001**: Reserved for other application (NOT crpbot)

---

## 🔍 Diagnosis Steps (Run on Cloud Server)

### 1. Check if Reflex Backend is Running

```bash
# SSH to cloud server
ssh root@178.156.136.185

# Check for Reflex processes
ps aux | grep reflex | grep -v grep

# Expected output: Should show reflex process running
# If nothing shown: Reflex is NOT running
```

### 2. Check Port 8000 Availability

```bash
# Check what's using port 8000
sudo lsof -i :8000
# OR
sudo netstat -tulpn | grep :8000
# OR
sudo ss -tulpn | grep :8000

# If port is in use by another process: Stop that process
# If port is free: Reflex backend is not running
```

### 3. Check for Port Conflicts

```bash
# Check all relevant ports
sudo lsof -i :3000  # Reflex frontend
sudo lsof -i :5000  # Flask dashboard
sudo lsof -i :8000  # Reflex backend
sudo lsof -i :8001  # Other application
```

---

## 🚀 Fix: Start/Restart Reflex Dashboard

### Option A: Quick Restart

```bash
# On cloud server (root@178.156.136.185)
cd /root/crpbot

# Pull latest config (port 8000 restored)
git pull origin feature/v7-ultimate

# Kill any existing Reflex processes
pkill -f reflex

# Wait 2 seconds
sleep 2

# Start Reflex dashboard
cd apps/dashboard_reflex
nohup reflex run --loglevel info > /tmp/reflex_dashboard.log 2>&1 &

# Verify it started
sleep 5
tail -50 /tmp/reflex_dashboard.log

# Check if port 8000 is now in use
sudo lsof -i :8000
```

### Option B: Debug Mode (Interactive)

```bash
# On cloud server
cd /root/crpbot/apps/dashboard_reflex

# Kill existing processes
pkill -f reflex

# Run in foreground to see errors
reflex run --loglevel debug

# Watch for errors:
# - Port already in use → Another app using port 8000
# - Module import errors → Missing dependencies
# - Database connection errors → Check DB_URL in .env
```

---

## 🐛 Common Issues & Fixes

### Issue 1: Port 8000 Already in Use

**Symptom**:
```
OSError: [Errno 98] Address already in use
```

**Fix**:
```bash
# Find what's using port 8000
sudo lsof -i :8000

# Kill the process (replace PID with actual PID)
sudo kill -9 <PID>

# Or if it's your other application on wrong port:
# Stop your other application
# Reconfigure it to use port 8001
# Start your other application on 8001
# Then start Reflex on 8000
```

### Issue 2: Reflex Not Starting

**Symptom**: No process found when checking `ps aux | grep reflex`

**Fix**:
```bash
# Check if reflex is installed
which reflex
reflex --version

# If not installed:
cd /root/crpbot
source .venv/bin/activate  # or use uv
pip install reflex
# OR
uv pip install reflex

# Try starting again
cd apps/dashboard_reflex
reflex run
```

### Issue 3: Database Connection Error

**Symptom**: Dashboard starts but shows no data

**Fix**:
```bash
# Verify database exists
ls -lh /root/crpbot/tradingai.db

# Check database has data
sqlite3 /root/crpbot/tradingai.db "SELECT COUNT(*) FROM signals;"

# Verify .env has correct DB_URL
grep DB_URL /root/crpbot/.env
# Should show: DB_URL=sqlite:///tradingai.db
```

### Issue 4: Firewall Blocking Port 8000

**Symptom**: Port 8000 open locally but not accessible from outside

**Fix**:
```bash
# Check firewall rules
sudo ufw status

# Allow port 8000 (if using UFW)
sudo ufw allow 8000/tcp

# OR for iptables:
sudo iptables -A INPUT -p tcp --dport 8000 -j ACCEPT
sudo iptables-save > /etc/iptables/rules.v4
```

---

## ✅ Verification Steps

After starting Reflex dashboard:

### 1. Check Process Running
```bash
ps aux | grep reflex | grep -v grep
# Should show: reflex process with PID
```

### 2. Check Ports Listening
```bash
sudo netstat -tulpn | grep -E ':(3000|8000)'
# Should show:
# tcp 0.0.0.0:3000 ... LISTEN ... reflex
# tcp 0.0.0.0:8000 ... LISTEN ... reflex
```

### 3. Test WebSocket Locally
```bash
# On cloud server
curl -I http://localhost:8000/_event
# Should return: Connection: Upgrade (WebSocket)

# Or test with websocat (if installed)
echo "test" | websocat ws://localhost:8000/_event
```

### 4. Test from Browser
```
Open: http://178.156.136.185:3000
Open browser console (F12)
Check Network tab for WebSocket connection to ws://178.156.136.185:8000/_event
Status should be: 101 Switching Protocols (successful)
```

---

## 📊 Expected Dashboard Status

After successful restart:

| Component | URL | Status |
|-----------|-----|--------|
| **Frontend** | http://178.156.136.185:3000 | ✅ Should load UI |
| **WebSocket** | ws://178.156.136.185:8000/_event | ✅ Should connect |
| **Backend API** | http://178.156.136.185:8000/ping | ✅ Should respond |

---

## 🔧 Quick Diagnostic Script

Run this on the cloud server to get full status:

```bash
#!/bin/bash
echo "=== CRPBot Dashboard Diagnostic ==="
echo ""

echo "1. Reflex Process:"
ps aux | grep reflex | grep -v grep || echo "   ❌ No Reflex process found"
echo ""

echo "2. Port Status:"
sudo lsof -i :3000 | grep LISTEN || echo "   ❌ Port 3000 not listening"
sudo lsof -i :8000 | grep LISTEN || echo "   ❌ Port 8000 not listening"
echo ""

echo "3. Database:"
if [ -f "/root/crpbot/tradingai.db" ]; then
    echo "   ✅ Database file exists"
    SIGNAL_COUNT=$(sqlite3 /root/crpbot/tradingai.db "SELECT COUNT(*) FROM signals;" 2>/dev/null)
    echo "   ✅ Signals in DB: $SIGNAL_COUNT"
else
    echo "   ❌ Database file not found"
fi
echo ""

echo "4. Recent Logs:"
if [ -f "/tmp/reflex_dashboard.log" ]; then
    echo "   Last 10 lines of reflex log:"
    tail -10 /tmp/reflex_dashboard.log
else
    echo "   ❌ No log file found at /tmp/reflex_dashboard.log"
fi
echo ""

echo "5. Config:"
grep -E "frontend_port|backend_port" /root/crpbot/rxconfig.py
```

Save as `diagnostic.sh`, then:
```bash
chmod +x diagnostic.sh
./diagnostic.sh
```

---

## 📝 Summary

**Root Cause**: Reflex backend not running on port 8000

**Solution**:
1. Ensure port 8000 is free (your other app moved to 8001)
2. Pull latest config (backend_port=8000)
3. Restart Reflex dashboard
4. Verify WebSocket connection

**Final Port Allocation**:
- 3000: Reflex frontend ✅
- 5000: Flask dashboard ✅
- 8000: Reflex backend ✅
- 8001: Your other application ✅

---

**Next Steps**: Run the restart commands on the cloud server and verify the dashboard loads.
