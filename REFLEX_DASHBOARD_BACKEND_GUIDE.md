# Reflex Dashboard Backend Guide

**Last Updated**: 2025-11-21
**Dashboard Location**: `/root/crpbot/apps/dashboard_reflex/`

---

## Architecture Overview

The Reflex dashboard consists of **two separate processes**:

1. **Frontend (Node.js/React)** - Port 3000
   - Handles UI rendering
   - WebSocket client for real-time updates
   - Built from Python code using Reflex compiler

2. **Backend (Python/FastAPI)** - Port 8000
   - Serves API endpoints
   - WebSocket server for state management
   - Database queries via SQLAlchemy
   - Manages application state

---

## File Structure

```
apps/dashboard_reflex/
├── dashboard_reflex/
│   ├── __init__.py
│   └── dashboard_reflex.py  # Main dashboard code (950 lines)
├── rxconfig.py              # Reflex configuration
├── .web/                    # Generated frontend code (auto-generated)
└── assets/                  # Static assets
```

---

## Backend State Management

### State Class: `V7State`

Location: `apps/dashboard_reflex/dashboard_reflex/dashboard_reflex.py:48-349`

**Key Responsibilities**:
1. Database connection pooling
2. Data fetching from SQLite/PostgreSQL
3. State synchronization via WebSocket
4. Real-time updates to frontend

### State Variables

```python
class V7State(rx.State):
    # Signal data
    signals: List[Dict[str, Any]] = []
    signal_count: int = 0
    buy_count: int = 0
    sell_count: int = 0
    hold_count: int = 0
    avg_confidence: float = 0.0

    # Market prices
    btc_price: float = 0.0
    eth_price: float = 0.0
    sol_price: float = 0.0

    # Performance tracking
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    open_positions: List[Position] = []
    recent_trades: List[Trade] = []

    # A/B Test data
    full_math_total: int = 0
    full_math_wins: int = 0
    full_math_win_rate: float = 0.0
    full_math_trades: List[Trade] = []

    deepseek_only_total: int = 0
    deepseek_only_wins: int = 0
    deepseek_only_win_rate: float = 0.0
    deepseek_only_trades: List[Trade] = []
```

### Database Connection Pooling

**CRITICAL**: Use class-level connection pooling to prevent memory leaks.

```python
class V7State(rx.State):
    _engine = None
    _Session = None

    @classmethod
    def get_session(cls):
        """Get a database session with proper connection pooling"""
        if cls._engine is None:
            cls._engine = create_engine(
                str(config.db_url),
                pool_size=5,
                max_overflow=10,
                pool_recycle=3600,  # Recycle connections after 1 hour
                pool_pre_ping=True  # Verify connections before use
            )
            cls._Session = sessionmaker(bind=cls._engine)
        return cls._Session()
```

**Always close sessions**:
```python
def fetch_signals(self):
    session = self.get_session()
    try:
        # ... query database ...
    finally:
        session.close()  # CRITICAL - prevents connection leaks
```

---

## Backend Methods

### 1. Data Fetching Methods

#### `fetch_signals()` - Lines 101-173
```python
def fetch_signals(self):
    """Fetch latest V7 signals from database"""
    session = self.get_session()
    try:
        since = datetime.now() - timedelta(hours=2)
        signals_query = session.query(Signal).filter(
            Signal.timestamp >= since,
            Signal.model_version == 'v7_ultimate'
        ).order_by(desc(Signal.timestamp)).limit(30).all()

        # Process and update state
        self.signals = signals_data
        self.signal_count = len(signals_query)
    finally:
        session.close()
```

#### `fetch_market_prices()` - Lines 175-199
```python
def fetch_market_prices(self):
    """Fetch latest market prices"""
    session = self.get_session()
    try:
        for symbol in ['BTC-USD', 'ETH-USD', 'SOL-USD']:
            latest = session.query(Signal).filter(
                Signal.symbol == symbol
            ).order_by(desc(Signal.timestamp)).first()

            if latest and latest.entry_price:
                if symbol == 'BTC-USD':
                    self.btc_price = latest.entry_price
    finally:
        session.close()
```

#### `fetch_performance()` - Lines 201-254
```python
def fetch_performance(self):
    """Fetch performance tracking data"""
    tracker = PerformanceTracker()

    # Get performance stats
    stats = tracker.get_win_rate(days=30)
    self.total_trades = stats['total_trades']
    self.wins = stats['wins']
    self.win_rate = stats['win_rate']

    # Get open positions and recent trades
    self.open_positions = tracker.get_open_positions()
    self.recent_trades = tracker.get_recent_trades(limit=10)
```

#### `fetch_ab_test_data()` - Lines 273-342
```python
def fetch_ab_test_data(self):
    """Fetch A/B test comparison data"""
    tracker = PerformanceTracker()

    # Get data for v7_full_math strategy
    full_math_stats = tracker.get_win_rate(days=30, strategy="v7_full_math")
    self.full_math_total = full_math_stats['total_trades']
    self.full_math_trades = tracker.get_recent_trades(limit=10, strategy="v7_full_math")

    # Get data for v7_deepseek_only strategy
    deepseek_only_stats = tracker.get_win_rate(days=30, strategy="v7_deepseek_only")
    self.deepseek_only_total = deepseek_only_stats['total_trades']
    self.deepseek_only_trades = tracker.get_recent_trades(limit=10, strategy="v7_deepseek_only")
```

### 2. Lifecycle Methods

#### `on_load()` - Lines 344-348
```python
def on_load(self):
    """Called when page loads - fetch data"""
    self.fetch_signals()
    self.fetch_market_prices()
    self.fetch_performance()
```

---

## Routes and Pages

### Route Configuration

Location: `apps/dashboard_reflex/dashboard_reflex/dashboard_reflex.py:945-949`

```python
app = rx.App()
app.add_page(index, route="/", title="V7 Ultimate Dashboard")
app.add_page(performance, route="/performance", title="Performance Tracking")
app.add_page(ab_test_comparison, route="/ab-test", title="A/B Test Comparison")
```

### Page Functions

1. **`index()`** - Lines 389-412
   - Main dashboard page
   - Shows recent signals (last 2 hours)
   - Displays signal counts and average confidence
   - Market prices for BTC/ETH/SOL

2. **`performance()`** - Lines 558-596
   - Performance tracking page
   - Shows win rate, profit factor, avg win/loss
   - Displays open positions and recent trades

3. **`ab_test_comparison()`** - Lines 687-942
   - A/B test comparison page
   - Side-by-side view of v7_full_math vs v7_deepseek_only
   - Shows metrics and recent trades for each strategy

---

## Starting/Stopping the Backend

### Start Dashboard (Production)

```bash
cd /root/crpbot/apps/dashboard_reflex

# Clean start
rm -rf /root/crpbot/**/__pycache__ 2>/dev/null
nohup ../../.venv/bin/python3 -m reflex run > /tmp/reflex_dashboard.log 2>&1 &

# Monitor logs
tail -f /tmp/reflex_dashboard.log
```

### Check Status

```bash
# Check if both processes are running
lsof -i :3000  # Frontend (Node.js)
lsof -i :8000  # Backend (Python)

# Check process status
ps aux | grep "reflex run"
ps aux | grep "python.*reflex"
```

### Stop Dashboard

```bash
# Kill all Reflex processes
pkill -9 -f "reflex"
pkill -9 -f "bun"
pkill -9 -f "node.*react-router"

# Verify stopped
lsof -i :3000
lsof -i :8000
```

---

## Common Issues and Solutions

### Issue 1: WebSocket Connection Error

**Symptom**: "Cannot connect to server: websocket error"

**Cause**: Backend (port 8000) not running or crashed

**Solution**:
```bash
# Check backend logs
tail -100 /tmp/reflex_dashboard.log | grep -i error

# Restart backend
cd /root/crpbot/apps/dashboard_reflex
../../.venv/bin/python3 -m reflex run
```

### Issue 2: Database Connection Leaks

**Symptom**: Backend becomes unresponsive after some time

**Cause**: Sessions not being closed properly

**Solution**: Always use try/finally blocks:
```python
def fetch_data(self):
    session = self.get_session()
    try:
        # ... database operations ...
    finally:
        session.close()  # ALWAYS close
```

### Issue 3: Port Already in Use

**Symptom**: "Address already in use" error

**Solution**:
```bash
# Find process using port
lsof -i :3000
lsof -i :8000

# Kill specific process
kill -9 <PID>

# Or kill all Reflex processes
pkill -9 -f "reflex"
```

### Issue 4: Frontend Shows No Data

**Symptom**: Dashboard loads but shows "0 signals" or blank data

**Cause**: Backend not fetching data or database query issues

**Debug**:
```bash
# Check backend logs for SQL errors
tail -100 /tmp/reflex_dashboard.log

# Test database connection
sqlite3 /root/crpbot/tradingai.db "SELECT COUNT(*) FROM signals"

# Check if on_load() is being called
grep "on_load" /tmp/reflex_dashboard.log
```

---

## Performance Optimization

### 1. Connection Pooling

Use class-level engine and sessionmaker:
```python
_engine = create_engine(
    str(config.db_url),
    pool_size=5,           # Max 5 connections
    max_overflow=10,       # Allow 10 extra if needed
    pool_recycle=3600,     # Recycle after 1 hour
    pool_pre_ping=True     # Verify before use
)
```

### 2. Query Optimization

Limit query results:
```python
# Good
.limit(30).all()

# Bad
.all()  # Returns everything
```

Use indexes in database:
```sql
CREATE INDEX idx_signals_timestamp ON signals(timestamp);
CREATE INDEX idx_signals_symbol ON signals(symbol);
CREATE INDEX idx_signals_strategy ON signals(strategy);
```

### 3. State Updates

Only update changed data:
```python
# Good - only update if changed
if new_data != self.signals:
    self.signals = new_data

# Avoid unnecessary updates
```

---

## Debugging Tips

### Enable Verbose Logging

```bash
cd /root/crpbot/apps/dashboard_reflex
../../.venv/bin/python3 -m reflex run --loglevel debug
```

### Check Database Queries

```python
# Add logging to state methods
import logging
logger = logging.getLogger(__name__)

def fetch_signals(self):
    logger.info("Fetching signals...")
    # ...
    logger.info(f"Found {len(signals_query)} signals")
```

### Monitor Real-time

```bash
# Watch logs in real-time
tail -f /tmp/reflex_dashboard.log

# Filter for errors only
tail -f /tmp/reflex_dashboard.log | grep -i error

# Watch both ports
watch -n 1 'lsof -i :3000 && echo && lsof -i :8000'
```

---

## Environment Variables

```bash
# Database connection
DB_URL=sqlite:///tradingai.db
# or
DB_URL=postgresql+psycopg://user:pass@localhost:5432/tradingai

# Reflex settings (optional)
REFLEX_LOG_LEVEL=INFO
REFLEX_BACKEND_PORT=8000
REFLEX_FRONTEND_PORT=3000
```

---

## Monitoring Checklist

Before reporting issues, verify:

- [ ] Backend process is running (`ps aux | grep reflex`)
- [ ] Port 8000 is listening (`lsof -i :8000`)
- [ ] Port 3000 is listening (`lsof -i :3000`)
- [ ] No errors in logs (`tail -100 /tmp/reflex_dashboard.log`)
- [ ] Database is accessible (`sqlite3 tradingai.db ".schema"`)
- [ ] No connection leaks (check log file size)
- [ ] V7 runtime is generating signals (`tail /tmp/v7_ab_testing_production.log`)

---

## Quick Reference Commands

```bash
# Start dashboard
cd /root/crpbot/apps/dashboard_reflex && ../../.venv/bin/python3 -m reflex run

# Stop dashboard
pkill -9 -f "reflex"

# Check status
lsof -i :3000 && lsof -i :8000

# View logs
tail -f /tmp/reflex_dashboard.log

# Test backend
curl http://localhost:8000/_ping

# Clear cache and restart
rm -rf /root/crpbot/**/__pycache__ && cd /root/crpbot/apps/dashboard_reflex && ../../.venv/bin/python3 -m reflex run
```

---

## Support Resources

- **Reflex Docs**: https://reflex.dev/docs/getting-started/introduction/
- **Dashboard Code**: `/root/crpbot/apps/dashboard_reflex/dashboard_reflex/dashboard_reflex.py`
- **Performance Tracker**: `/root/crpbot/libs/tracking/performance_tracker.py`
- **Database Models**: `/root/crpbot/libs/db/models.py`
