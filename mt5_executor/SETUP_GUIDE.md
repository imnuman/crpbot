# HYDRA MT5 Executor - ForexVPS Setup Guide

## Overview

This guide helps you set up the MT5 Executor on your ForexVPS Windows server for live FTMO trading.

```
Architecture:
┌─────────────────────────┐         ┌─────────────────────────┐
│  Linux Server           │   HTTP  │  ForexVPS (Windows)     │
│  (178.156.136.185)      │ ──────► │  (NY4 Equinix)          │
│                         │  <1ms   │                         │
│  HYDRA 4.0              │         │  MT5 Executor Service   │
│  - 4 AI Engines         │         │  - MT5 Terminal         │
│  - Signal Generation    │         │  - Trade Execution      │
│  - Paper Trading        │         │  - Position Monitor     │
└─────────────────────────┘         └─────────────────────────┘
                                              │
                                              │ <1ms
                                              ▼
                                    ┌─────────────────────────┐
                                    │  FTMO MT5 Servers       │
                                    │  (Same NY4 Building)    │
                                    └─────────────────────────┘
```

## Step 1: ForexVPS Setup

### 1.1 Order Your VPS

1. Go to: https://www.forexvps.net
2. Choose: **Basic Plan** ($29.99/mo) or higher
3. Location: **New York (NY4)** - CRITICAL for low latency
4. Wait for provisioning email (~10-30 minutes)

### 1.2 Connect via RDP

1. Open **Remote Desktop Connection** (Windows) or use **Remmina** (Linux)
2. Enter the IP address from ForexVPS email
3. Username: `Administrator`
4. Password: From ForexVPS email

## Step 2: Install MT5 Terminal

### 2.1 Download MT5

1. Log into FTMO Client Area: https://trader.ftmo.com
2. Go to: **Account Settings** → **Download Platform**
3. Download: **MetaTrader 5** installer
4. Run installer and complete setup

### 2.2 Login to MT5

1. Open MetaTrader 5
2. File → Login to Trade Account
3. Enter credentials:
   - **Login**: `531025383`
   - **Password**: `c*B@lWp41b784c`
   - **Server**: `FTMO-Server3`
4. Click **Login**
5. Verify connection (bottom right should show connection status)

## Step 3: Install Python Environment

### 3.1 Install Python

Open **PowerShell as Administrator** and run:

```powershell
# Download Python installer
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe" -OutFile "python-installer.exe"

# Install Python (silent, add to PATH)
.\python-installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0

# Refresh environment
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine")

# Verify installation
python --version
```

### 3.2 Install Required Packages

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install required packages
pip install MetaTrader5 flask loguru requests
```

## Step 4: Deploy Executor Service

### 4.1 Create Project Directory

```powershell
# Create directory
mkdir C:\HYDRA
cd C:\HYDRA

# Create executor script
notepad executor_service.py
```

### 4.2 Copy Executor Code

Copy the entire contents of `executor_service.py` from this folder into the notepad window, then save.

Or download directly:

```powershell
# If you have git:
git clone https://github.com/imnuman/crpbot.git
cd crpbot\mt5_executor
```

### 4.3 Create Environment File

Create `C:\HYDRA\.env`:

```powershell
notepad C:\HYDRA\.env
```

Add these lines:

```
FTMO_LOGIN=531025383
FTMO_PASS=c*B@lWp41b784c
FTMO_SERVER=FTMO-Server3
API_SECRET=hydra_secret_2024
```

### 4.4 Create Startup Script

Create `C:\HYDRA\start_executor.bat`:

```batch
@echo off
cd C:\HYDRA
set FTMO_LOGIN=531025383
set FTMO_PASS=c*B@lWp41b784c
set FTMO_SERVER=FTMO-Server3
set API_SECRET=hydra_secret_2024
python executor_service.py
pause
```

## Step 5: Configure Firewall

### 5.1 Allow Port 5000

```powershell
# Open PowerShell as Administrator
New-NetFirewallRule -DisplayName "HYDRA Executor" -Direction Inbound -Port 5000 -Protocol TCP -Action Allow
```

### 5.2 Verify Port is Open

From your Linux server, test connectivity:

```bash
# Replace YOUR_FOREXVPS_IP with actual IP
curl http://YOUR_FOREXVPS_IP:5000/health
```

## Step 6: Start the Executor

### 6.1 Manual Start (Testing)

1. Make sure MT5 Terminal is running and logged in
2. Double-click `start_executor.bat`
3. You should see:

```
============================================================
    HYDRA MT5 Executor Service
    ForexVPS Edition - Ultra Low Latency
============================================================

[CONFIG] MT5 Login: 531025383
[CONFIG] MT5 Server: FTMO-Server3
[CONFIG] API Port: 5000

[MT5] Connecting to FTMO-Server3...
[MT5] Connected successfully!
[MT5] Account: 531025383
[MT5] Balance: $15,000.00

[API] Starting server on http://0.0.0.0:5000
Ready to receive signals from HYDRA!
```

### 6.2 Auto-Start on Boot (Production)

1. Press `Win + R`, type `shell:startup`, press Enter
2. Create shortcut to `start_executor.bat` in this folder
3. Also create shortcut to MT5 terminal

## Step 7: Configure Linux Server

### 7.1 Set Environment Variables

On your Linux server (178.156.136.185):

```bash
# Edit .env file
nano /root/crpbot/.env

# Add these lines:
MT5_EXECUTOR_URL=http://YOUR_FOREXVPS_IP:5000
MT5_API_SECRET=hydra_secret_2024
EXECUTION_MODE=live
```

### 7.2 Test Connection

```bash
# Test health endpoint
curl -H "Authorization: Bearer hydra_secret_2024" \
     http://YOUR_FOREXVPS_IP:5000/health

# Test account endpoint
curl -H "Authorization: Bearer hydra_secret_2024" \
     http://YOUR_FOREXVPS_IP:5000/account
```

### 7.3 Test Trade Execution

```bash
# Send test signal (use small volume!)
curl -X POST \
     -H "Authorization: Bearer hydra_secret_2024" \
     -H "Content-Type: application/json" \
     -d '{
       "symbol": "BTCUSD",
       "direction": "BUY",
       "volume": 0.01,
       "stop_loss": 95000,
       "take_profit": 105000,
       "engine": "TEST",
       "trade_id": "TEST_001"
     }' \
     http://YOUR_FOREXVPS_IP:5000/execute
```

## Step 8: Go Live

### 8.1 Switch HYDRA to Live Mode

```bash
# Edit runtime configuration
export EXECUTION_MODE=live

# Or add to .env:
echo "EXECUTION_MODE=live" >> /root/crpbot/.env

# Restart HYDRA
docker compose down hydra-runtime
docker compose up -d hydra-runtime
```

### 8.2 Monitor Execution

On Windows VPS, watch the executor console for incoming signals.

On Linux, monitor HYDRA logs:

```bash
docker logs -f hydra-runtime 2>&1 | grep -E "MT5|LIVE|execute"
```

## Troubleshooting

### MT5 Connection Failed

1. Make sure MT5 Terminal is running
2. Check if you're logged in (bottom right status)
3. Try restarting MT5 terminal
4. Verify credentials in environment variables

### API Connection Refused

1. Check Windows Firewall allows port 5000
2. Verify ForexVPS IP hasn't changed
3. Test locally first: `curl http://localhost:5000/health`

### Orders Rejected

1. Check account balance and margin
2. Verify symbol name (BTCUSD vs BTC-USD)
3. Check lot size is within limits
4. Review MT5 journal for error messages

### Latency Issues

1. ForexVPS should be in NY4 - verify with support
2. Check network route: `tracert YOUR_LINUX_SERVER_IP`
3. Monitor execution times in logs

## Security Notes

1. **Change API_SECRET** to a strong random string
2. **Don't expose port 5000** to the public internet
3. Use **VPN or SSH tunnel** for production
4. **Monitor account** regularly for unauthorized trades
5. **Set up alerts** for large drawdowns

## Quick Reference

| Setting | Value |
|---------|-------|
| MT5 Login | 531025383 |
| MT5 Server | FTMO-Server3 |
| API Port | 5000 |
| API Secret | hydra_secret_2024 (CHANGE THIS!) |
| Max Daily Loss | 4.5% |
| Max Total Loss | 9% |

## Support

- FTMO Support: support@ftmo.com
- ForexVPS Support: Check your welcome email
- HYDRA Issues: https://github.com/imnuman/crpbot/issues
