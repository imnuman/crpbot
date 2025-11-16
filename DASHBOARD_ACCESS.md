# V6 Enhanced Model Dashboard - Access Guide

**Status**: LIVE on Cloud Server
**Deployment Date**: 2025-11-16
**Public Access**: http://178.156.136.185:5000

---

## Quick Access

### Web Browser
Open your browser and navigate to:

**http://178.156.136.185:5000**

The dashboard will automatically refresh every 5 seconds to show real-time predictions and signals.

---

## Dashboard Features

### 1. System Overview
- **Mode**: Current runtime mode (LIVE/DRYRUN)
- **Confidence Threshold**: 65%
- **Average Accuracy**: 69.87% (BTC: 67.58%, ETH: 71.65%, SOL: 70.39%)
- **Architecture**: V6 Enhanced FNN (4-layer feedforward: 72→256→128→64→3)

### 2. Data Sources Monitor
Real-time status of all 3 data sources:
- **Coinbase Advanced Trade API**
  - Type: Primary OHLCV
  - Interval: 1-minute candles
  - Status: Active

- **Kraken API**
  - Type: OHLCV Backup + Multi-Timeframe
  - Intervals: 1m, 5m, 15m, 1h
  - Status: Active

- **CoinGecko Professional API**
  - Type: Fundamentals
  - Interval: Daily
  - Data: Market cap, ATH, sentiment
  - Status: Active

### 3. Live Predictions
Interactive doughnut charts showing 3-class probabilities for each symbol:
- **Down Probability** (Red)
- **Neutral Probability** (Gray)
- **Up Probability** (Green)

Each prediction card displays:
- Current direction (LONG/SHORT/NEUTRAL)
- Confidence percentage
- Tier classification (HIGH/MEDIUM/LOW)
- Timestamp of last prediction

### 4. Signal Statistics (24h)
- Total signals generated
- Average confidence
- Hourly signal rate
- Breakdown by:
  - Symbol (BTC-USD, ETH-USD, SOL-USD)
  - Direction (LONG, SHORT)
  - Tier (HIGH, MEDIUM, LOW)

### 5. Recent Signals
Table showing last 10 signals with:
- Timestamp
- Symbol
- Direction
- Confidence
- Tier

### 6. Analysis Process Visualization
Step-by-step view of how the model works:
1. **Data Collection**: Fetch OHLCV from Coinbase, Kraken, CoinGecko
2. **Feature Engineering**: Generate 72 Amazon Q features
3. **Model Inference**: V6 Enhanced FNN prediction
4. **Signal Generation**: Filter by 65% confidence threshold

---

## API Endpoints

You can also access the dashboard data programmatically:

### System Status
```bash
curl http://178.156.136.185:5000/api/status
```

Returns:
- System configuration
- Model architecture and accuracy
- Data source status

### Recent Signals
```bash
curl http://178.156.136.185:5000/api/signals/recent/24
```

Returns last 24 hours of signals in JSON format.

### Signal Statistics
```bash
curl http://178.156.136.185:5000/api/signals/stats/24
```

Returns aggregated statistics for last 24 hours.

### Live Predictions
```bash
curl http://178.156.136.185:5000/api/predictions/live
```

Returns latest prediction for each symbol.

---

## Monitoring Commands

### Check Dashboard Status
```bash
ssh root@178.156.136.185 "tail -f /tmp/dashboard.log"
```

### Check Runtime Status
```bash
ssh root@178.156.136.185 "tail -f /tmp/v6_65pct.log"
```

### Restart Dashboard
```bash
ssh root@178.156.136.185 "pkill -f 'apps/dashboard/app.py' && cd ~/crpbot && nohup .venv/bin/python3 apps/dashboard/app.py > /tmp/dashboard.log 2>&1 &"
```

---

## Understanding the Dashboard

### Signal Tiers
- **High (🔥)**: Confidence ≥75% - Strong conviction signals
- **Medium (⚡)**: Confidence ≥65% - Moderate conviction signals
- **Low (💡)**: Confidence ≥55% - Low conviction signals

### Model Confidence
The V6 Enhanced FNN outputs 3 probabilities (Down, Neutral, Up):
- **Up Probability** is used as confidence for LONG signals
- **Down Probability** is used as confidence for SHORT signals
- Only signals above 65% confidence are displayed

### Data Flow
```
Market Data (1m candles)
    ↓
Feature Engineering (72 indicators)
    ↓
V6 Enhanced FNN (4 layers)
    ↓
3-Class Output (Down, Neutral, Up)
    ↓
Signal (if confidence ≥65%)
    ↓
Dashboard + Database
```

---

## Technical Details

### Dashboard Technology
- **Backend**: Flask (Python web framework)
- **Frontend**: HTML5, CSS3, JavaScript
- **Charts**: Chart.js (doughnut charts)
- **Updates**: Auto-refresh every 5 seconds via AJAX

### Server Configuration
- **Host**: 0.0.0.0 (all interfaces)
- **Port**: 5000
- **Mode**: Development server (non-production)
- **Logs**: /tmp/dashboard.log

### Database
- **Type**: SQLite
- **Location**: ~/crpbot/tradingai.db
- **Tables**: Signal (stores all generated signals)

---

## Troubleshooting

### Dashboard Not Loading
1. Check if dashboard process is running:
   ```bash
   ssh root@178.156.136.185 "ps aux | grep dashboard"
   ```

2. Check dashboard logs:
   ```bash
   ssh root@178.156.136.185 "cat /tmp/dashboard.log"
   ```

3. Restart dashboard (see Monitoring Commands above)

### No Signals Showing
This is expected when:
- Market is ranging/choppy with no clear direction
- Model confidence is below 65% threshold
- Runtime recently restarted and hasn't completed first scan yet

Check runtime logs:
```bash
ssh root@178.156.136.185 "tail -50 /tmp/v6_65pct.log"
```

### Charts Not Updating
- Ensure JavaScript is enabled in your browser
- Check browser console for errors (F12)
- Try hard refresh (Ctrl+Shift+R or Cmd+Shift+R)

---

## Next Steps

1. **Monitor for 24-48 hours** to observe signal frequency and patterns
2. **Track accuracy** by comparing predictions vs actual price movements
3. **Adjust threshold** if needed (currently 65%)
4. **Enable Telegram notifications** for high-confidence signals
5. **Set up continuous retraining** pipeline for weekly model updates

---

## Quick Reference

| Item | Value |
|------|-------|
| **Dashboard URL** | http://178.156.136.185:5000 |
| **Confidence Threshold** | 65% |
| **Model Version** | V6 Enhanced FNN |
| **Average Accuracy** | 69.87% |
| **Features** | 72 (Amazon Q) |
| **Output Classes** | 3 (Down, Neutral, Up) |
| **Scan Interval** | 60 seconds |
| **Auto-Refresh** | 5 seconds |

---

**Dashboard is LIVE! 🚀**

Access it now at http://178.156.136.185:5000 to see V6 Enhanced models analyzing the market in real-time!
