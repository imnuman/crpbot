# V4 Monitoring - Build Checklist

**START HERE** - Follow this checklist to build V4 in 18-20 days

---

## ✅ COMPLETED (Foundation)

- [x] Project structure created
- [x] README.md - Architecture documentation
- [x] database/schema.sql - Database schema
- [x] config/config.yaml - Configuration template
- [x] requirements.txt - Python dependencies
- [x] QUICK_START.md - Developer guide

---

## 🔨 TO BUILD (In Order)

### Day 1: Setup (Today)

**1. Deploy Database Schema**
```bash
# Read password
DB_PASSWORD=$(cat /home/numan/crpbot/.db_password)

# Deploy schema
PGPASSWORD="$DB_PASSWORD" psql \
  -h crpbot-rds-postgres-db.cyjcoys82evx.us-east-1.rds.amazonaws.com \
  -p 5432 \
  -U crpbot_admin \
  -d crpbot \
  -f database/schema.sql

# Verify tables
PGPASSWORD="$DB_PASSWORD" psql \
  -h crpbot-rds-postgres-db.cyjcoys82evx.us-east-1.rds.amazonaws.com \
  -p 5432 \
  -U crpbot_admin \
  -d crpbot \
  -c "\dt"
```

**2. Install Dependencies**
```bash
cd /home/numan/crpbot/v4_monitoring
pip install -r requirements.txt
```

**3. Configure Settings**
```bash
# Edit config/config.yaml
# Update:
# - models.path (point to V3 models when trained)
# - bybit.api_key & api_secret
# - telegram.bot_token & chat_id
# - database.password (use ${DB_PASSWORD} from environment)
```

---

### Days 2-3: Component 1 - Signal Generator

**File**: `components/signal_generator.py`

**What it does**:
- Loads V3 ONNX models
- Fetches market data from Bybit
- Runs ML inference
- Filters signals (conf≥77%, vol≥2x, RR≥2.0)
- Calculates entry/SL/TP/position size
- Saves to `signals` table

**Key functions to implement**:
```python
def load_models():
    # Load ONNX models from config.models.path

def fetch_market_data(coin):
    # Get latest candles from Bybit

def run_inference(features):
    # ML prediction using ensemble + meta-learner

def filter_signals(predictions):
    # Apply quality gates

def calculate_trade_params(signal):
    # Calculate entry/SL/TP/size based on ATR

def save_to_database(signals):
    # Insert into signals table

if __name__ == "__main__":
    # Run every 15 minutes (cron job)
    main()
```

**Test**:
```bash
python components/signal_generator.py
# Should generate 0-3 signals and save to database
```

---

### Days 4-6: Component 2 - Performance Monitor

**File**: `components/performance_monitor.py`

**What it does**:
- Monitors Bybit open positions
- Detects TP/SL hits
- Records wins/losses to `trades` table
- Calculates rolling metrics
- Updates `performance_snapshots` table

**Key functions**:
```python
def monitor_open_positions():
    # Check Bybit for open positions

def check_if_closed(position):
    # Detect if TP or SL hit

def record_trade(position, exit_reason):
    # Save to trades table with win/loss

def calculate_metrics():
    # Win rate last 20, daily P&L, ECE, etc.

def update_performance_snapshot():
    # Save to performance_snapshots table

if __name__ == "__main__":
    # Run every 5 minutes
    main()
```

---

### Days 7-8: Component 3 - Trading Controller

**File**: `components/trading_controller.py`

**What it does**:
- Checks safety conditions
- Sets traffic light status (🟢🟡🔴)
- Updates `controller_status` table
- Triggers alerts

**Key logic**:
```python
def check_safety_conditions():
    # Get latest performance metrics
    wr_last_20 = get_win_rate_last_20()
    daily_loss = get_daily_pnl()
    consecutive_losses = get_consecutive_losses()
    ece = get_ece()

    # Traffic light logic
    if wr_last_20 < 0.65 or daily_loss > 0.04 or consecutive_losses >= 5:
        return 'red', 0.0  # Stop trading
    elif wr_last_20 < 0.70:
        return 'yellow', 0.5  # Reduce size
    else:
        return 'green', 1.0  # Trade normally

def update_controller_status(status, multiplier, message):
    # Save to controller_status table

def trigger_alerts(status_change):
    # If changed to yellow/red, send alerts
```

---

### Days 9-10: Component 4 - Alert System

**Files**:
- `alerts/telegram_bot.py`
- `alerts/email_sender.py`

**6 alert types**:
1. Daily status (8 AM)
2. Signal alerts (real-time)
3. Warnings (yellow)
4. Critical (red)
5. Weekly reports (Monday 8 AM)
6. Predictive warnings

**Telegram setup**:
```python
from telegram import Bot

def send_telegram(message):
    bot = Bot(token=config['telegram']['bot_token'])
    bot.send_message(
        chat_id=config['telegram']['chat_id'],
        text=message,
        parse_mode='HTML'
    )

def send_daily_status():
    status = get_current_status()
    wr = get_win_rate_last_20()
    signals = get_todays_signals()

    message = f"""
🌅 <b>Daily Status</b>
Status: {status}
Win Rate: {wr:.1%}
Signals Ready: {len(signals)}
    """
    send_telegram(message)
```

**Schedule alerts**:
```python
import schedule

schedule.every().day.at("08:00").do(send_daily_status)
schedule.every().monday.at("08:00").do(send_weekly_report)
```

---

### Days 11-15: Component 5 - Dashboard

**File**: `dashboard/app.py` (Streamlit)

**Layout**:
```python
import streamlit as st
import psycopg2

st.set_page_config(page_title="V4 Monitoring", layout="wide")

# Authentication
password = st.text_input("Password", type="password")
if password != config['dashboard']['password']:
    st.stop()

# Traffic Light Banner
status = get_current_status()
if status == 'green':
    st.success("🟢 GREEN - Trade Normally")
elif status == 'yellow':
    st.warning("🟡 YELLOW - Reduce Risk")
else:
    st.error("🔴 RED - Stop Trading")

# Performance Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Win Rate Last 20", f"{get_win_rate():.1%}")
col2.metric("Today's P&L", f"${get_daily_pnl():.2f}")
col3.metric("Sharpe Ratio", f"{get_sharpe():.2f}")
col4.metric("Current Streak", get_streak())

# Today's Signals
signals = get_todays_signals()
for signal in signals:
    with st.expander(f"{signal['coin']} {signal['direction']} - {signal['confidence']:.1%}"):
        st.write(f"Entry: {signal['entry_low']} - {signal['entry_high']}")
        st.write(f"Stop Loss: {signal['stop_loss']}")
        st.write(f"Take Profit: {signal['take_profit']}")
        st.write(f"Size: {signal['position_size']}")
        if st.button("Copy", key=signal['id']):
            st.code(format_signal_for_copy(signal))

# Auto-refresh
st_autorefresh(interval=60000)  # 60 seconds
```

**Run dashboard**:
```bash
streamlit run dashboard/app.py --server.port 8501
```

---

### Days 16-17: Component 6 - Execution Helper (Optional)

**Mode A: Copy-paste formatter** (Simple, recommend this)
```python
def format_signal_for_copy(signal):
    return f"""
{signal['coin']} {signal['direction']}
Entry: {signal['entry_low']} - {signal['entry_high']}
SL: {signal['stop_loss']}
TP: {signal['take_profit']}
Size: {signal['position_size']}
"""
```

**Mode B: Semi-automated** (Advanced, optional)
- Preview order via Bybit API
- User confirms
- Place limit entry + stop-market SL + limit TP

---

### Day 18: Integration Testing

**Test checklist**:
```bash
# 1. Database
python tests/test_database.py

# 2. Signal generation
python components/signal_generator.py
# Check signals table has new rows

# 3. Controller
python components/trading_controller.py
# Check controller_status table

# 4. Alerts
python alerts/test_telegram.py
# Should receive test message

# 5. Dashboard
streamlit run dashboard/app.py
# Check loads in <2 seconds
# Check traffic light displays
# Check signals show

# 6. End-to-end
# Generate signal → Execute on Bybit → Close position → Verify auto-tracked
```

---

### Days 19-20: Deployment

**VPS Setup**:
```bash
# 1. Setup server (DigitalOcean, Linode)
# - Ubuntu 22.04
# - 2GB RAM, 2 CPU

# 2. Install dependencies
sudo apt update
sudo apt install python3-pip postgresql-client nginx certbot

# 3. Clone code
scp -r v4_monitoring user@vps:/home/user/

# 4. Install Python packages
cd v4_monitoring
pip3 install -r requirements.txt

# 5. Setup services
# - Signal generator: cron every 15 min
# - Performance monitor: cron every 5 min
# - Dashboard: systemd service
# - Alerts: systemd service

# 6. Nginx reverse proxy
sudo nano /etc/nginx/sites-available/v4monitoring

# 7. SSL certificate
sudo certbot --nginx -d yourdomain.com

# 8. Start services
sudo systemctl start v4-dashboard
sudo systemctl enable v4-dashboard
```

---

## 📊 Progress Tracking

Update as you build:

- [ ] Day 1: Setup complete
- [ ] Days 2-3: Signal Generator working
- [ ] Days 4-6: Performance Monitor working
- [ ] Days 7-8: Trading Controller working
- [ ] Days 9-10: Alert System working
- [ ] Days 11-15: Dashboard working
- [ ] Days 16-17: Execution Helper (optional)
- [ ] Day 18: All tests passing
- [ ] Days 19-20: Deployed to VPS

---

## 🚀 When Complete

You'll have:
- ✅ Traffic light system (🟢🟡🔴)
- ✅ Dashboard at https://yourdomain.com:8501
- ✅ Telegram alerts (6 types)
- ✅ Auto win/loss tracking
- ✅ 15 min/day workflow

**Next**: Train V3 models, integrate with V4, start trading!

---

**Start building now!** Begin with Day 1 setup above.
