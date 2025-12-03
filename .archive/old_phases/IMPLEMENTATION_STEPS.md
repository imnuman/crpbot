# V7 Ultimate - Implementation Steps (Action Plan)

**Last Updated**: 2025-11-20
**Current Status**: Phase 1 Complete → Starting Phase 2

---

## 🎯 THE PROBLEM

**Right now**: V7 generates 157 signals in 2 hours, but we have NO IDEA if they're any good.
- No win rate tracking
- No P&L calculation
- No theory performance measurement
- No way to improve

**This is unacceptable.** We need data NOW.

---

## ✅ STEP 1: Create Database Tables (5 minutes)

**Do this first:**

```bash
cd /root/crpbot && .venv/bin/python3 << 'EOF'
from sqlalchemy import create_engine, text

engine = create_engine('sqlite:///tradingai.db')

with engine.connect() as conn:
    # Table to track trade outcomes
    conn.execute(text('''
        CREATE TABLE IF NOT EXISTS signal_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id INTEGER NOT NULL REFERENCES signals(id),
            entry_price REAL NOT NULL,
            entry_timestamp TIMESTAMP NOT NULL,
            exit_price REAL,
            exit_timestamp TIMESTAMP,
            pnl_percent REAL,
            pnl_usd REAL,
            outcome TEXT CHECK(outcome IN ('win', 'loss', 'breakeven', 'open')),
            exit_reason TEXT,
            hold_duration_minutes INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(signal_id)
        )
    '''))

    # Table to track theory contributions
    conn.execute(text('''
        CREATE TABLE IF NOT EXISTS theory_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            theory_name TEXT NOT NULL,
            signal_id INTEGER NOT NULL REFERENCES signals(id),
            contribution_score REAL,
            was_correct BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    '''))

    conn.commit()
    print('✅ Performance tracking tables created')

    # Verify tables exist
    result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
    tables = [row[0] for row in result]
    print(f'✅ Tables in database: {tables}')
EOF
```

**Verification**: You should see both `signal_results` and `theory_performance` tables listed.

---

## ✅ STEP 2: Build Performance Tracker Class (15 minutes)

**Create the tracker:**

```bash
mkdir -p /root/crpbot/libs/tracking
cat > /root/crpbot/libs/tracking/__init__.py << 'EOF'
"""Performance tracking module for V7"""
EOF

cat > /root/crpbot/libs/tracking/performance_tracker.py << 'EOF'
"""
Performance Tracker for V7 Signals
Tracks trade outcomes, calculates P&L, measures theory performance
"""

from datetime import datetime
from typing import Optional, Dict, List
from sqlalchemy import create_engine, text, func
from sqlalchemy.orm import sessionmaker
import sys
sys.path.insert(0, '/root/crpbot')
from libs.db.models import Signal
from libs.config.config import Settings

config = Settings()


class PerformanceTracker:
    """Track and measure V7 signal performance"""

    def __init__(self):
        self.engine = create_engine(str(config.db_url))
        self.Session = sessionmaker(bind=self.engine)

    def record_entry(self, signal_id: int, entry_price: float, entry_timestamp: datetime) -> bool:
        """Record when we enter a trade based on a signal"""
        session = self.Session()
        try:
            session.execute(text('''
                INSERT INTO signal_results (signal_id, entry_price, entry_timestamp, outcome)
                VALUES (:signal_id, :entry_price, :entry_timestamp, 'open')
                ON CONFLICT(signal_id) DO UPDATE SET
                    entry_price = :entry_price,
                    entry_timestamp = :entry_timestamp
            '''), {
                'signal_id': signal_id,
                'entry_price': entry_price,
                'entry_timestamp': entry_timestamp
            })
            session.commit()
            print(f'✅ Recorded entry for signal {signal_id} @ ${entry_price:,.2f}')
            return True
        except Exception as e:
            print(f'❌ Failed to record entry: {e}')
            session.rollback()
            return False
        finally:
            session.close()

    def record_exit(
        self,
        signal_id: int,
        exit_price: float,
        exit_timestamp: datetime,
        exit_reason: str = 'manual'
    ) -> bool:
        """Record when we exit a trade"""
        session = self.Session()
        try:
            # Get entry data
            result = session.execute(text('''
                SELECT entry_price, entry_timestamp FROM signal_results
                WHERE signal_id = :signal_id
            '''), {'signal_id': signal_id}).fetchone()

            if not result:
                print(f'❌ No entry found for signal {signal_id}')
                return False

            entry_price, entry_timestamp = result

            # Calculate P&L
            pnl_percent = ((exit_price - entry_price) / entry_price) * 100
            pnl_usd = exit_price - entry_price

            # Determine outcome
            if pnl_percent > 0.5:
                outcome = 'win'
            elif pnl_percent < -0.5:
                outcome = 'loss'
            else:
                outcome = 'breakeven'

            # Calculate hold duration
            duration = (exit_timestamp - entry_timestamp).total_seconds() / 60

            # Update record
            session.execute(text('''
                UPDATE signal_results
                SET exit_price = :exit_price,
                    exit_timestamp = :exit_timestamp,
                    pnl_percent = :pnl_percent,
                    pnl_usd = :pnl_usd,
                    outcome = :outcome,
                    exit_reason = :exit_reason,
                    hold_duration_minutes = :duration
                WHERE signal_id = :signal_id
            '''), {
                'signal_id': signal_id,
                'exit_price': exit_price,
                'exit_timestamp': exit_timestamp,
                'pnl_percent': pnl_percent,
                'pnl_usd': pnl_usd,
                'outcome': outcome,
                'exit_reason': exit_reason,
                'duration': int(duration)
            })
            session.commit()

            print(f'✅ Recorded {outcome} for signal {signal_id}: {pnl_percent:+.2f}% (held {int(duration)}m)')
            return True

        except Exception as e:
            print(f'❌ Failed to record exit: {e}')
            session.rollback()
            return False
        finally:
            session.close()

    def get_win_rate(self, days: int = 30) -> Dict[str, float]:
        """Calculate win rate over last N days"""
        session = self.Session()
        try:
            result = session.execute(text('''
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
                    AVG(CASE WHEN outcome = 'win' THEN pnl_percent ELSE NULL END) as avg_win,
                    AVG(CASE WHEN outcome = 'loss' THEN pnl_percent ELSE NULL END) as avg_loss,
                    AVG(pnl_percent) as avg_pnl
                FROM signal_results
                WHERE outcome IN ('win', 'loss', 'breakeven')
                  AND exit_timestamp >= datetime('now', '-' || :days || ' days')
            '''), {'days': days}).fetchone()

            if not result or result[0] == 0:
                return {
                    'total_trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'win_rate': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'avg_pnl': 0.0,
                    'profit_factor': 0.0
                }

            total, wins, losses, avg_win, avg_loss, avg_pnl = result
            win_rate = (wins / total * 100) if total > 0 else 0.0

            # Calculate profit factor (gross profit / gross loss)
            profit_factor = 0.0
            if wins > 0 and losses > 0 and avg_loss < 0:
                profit_factor = abs((wins * avg_win) / (losses * avg_loss))

            return {
                'total_trades': total,
                'wins': wins or 0,
                'losses': losses or 0,
                'win_rate': win_rate,
                'avg_win': avg_win or 0.0,
                'avg_loss': avg_loss or 0.0,
                'avg_pnl': avg_pnl or 0.0,
                'profit_factor': profit_factor
            }
        finally:
            session.close()

    def get_open_positions(self) -> List[Dict]:
        """Get all currently open positions"""
        session = self.Session()
        try:
            results = session.execute(text('''
                SELECT sr.signal_id, s.symbol, s.direction, sr.entry_price, sr.entry_timestamp
                FROM signal_results sr
                JOIN signals s ON sr.signal_id = s.id
                WHERE sr.outcome = 'open'
                ORDER BY sr.entry_timestamp DESC
            ''')).fetchall()

            return [
                {
                    'signal_id': r[0],
                    'symbol': r[1],
                    'direction': r[2],
                    'entry_price': r[3],
                    'entry_timestamp': r[4]
                }
                for r in results
            ]
        finally:
            session.close()

    def get_recent_trades(self, limit: int = 20) -> List[Dict]:
        """Get recent closed trades"""
        session = self.Session()
        try:
            results = session.execute(text('''
                SELECT
                    sr.signal_id,
                    s.symbol,
                    s.direction,
                    sr.entry_price,
                    sr.exit_price,
                    sr.pnl_percent,
                    sr.outcome,
                    sr.exit_timestamp,
                    sr.hold_duration_minutes
                FROM signal_results sr
                JOIN signals s ON sr.signal_id = s.id
                WHERE sr.outcome IN ('win', 'loss', 'breakeven')
                ORDER BY sr.exit_timestamp DESC
                LIMIT :limit
            '''), {'limit': limit}).fetchall()

            return [
                {
                    'signal_id': r[0],
                    'symbol': r[1],
                    'direction': r[2],
                    'entry_price': r[3],
                    'exit_price': r[4],
                    'pnl_percent': r[5],
                    'outcome': r[6],
                    'exit_timestamp': r[7],
                    'hold_duration_minutes': r[8]
                }
                for r in results
            ]
        finally:
            session.close()

    def record_theory_contribution(
        self,
        signal_id: int,
        theory_name: str,
        contribution_score: float,
        was_correct: Optional[bool] = None
    ) -> bool:
        """Record how much a theory contributed to a signal"""
        session = self.Session()
        try:
            session.execute(text('''
                INSERT INTO theory_performance (signal_id, theory_name, contribution_score, was_correct)
                VALUES (:signal_id, :theory_name, :contribution_score, :was_correct)
            '''), {
                'signal_id': signal_id,
                'theory_name': theory_name,
                'contribution_score': contribution_score,
                'was_correct': was_correct
            })
            session.commit()
            return True
        except Exception as e:
            print(f'❌ Failed to record theory contribution: {e}')
            session.rollback()
            return False
        finally:
            session.close()
EOF

chmod +x /root/crpbot/libs/tracking/performance_tracker.py
echo '✅ Performance tracker created'
```

**Verification**: Test the tracker:

```bash
cd /root/crpbot && .venv/bin/python3 << 'EOF'
from libs.tracking.performance_tracker import PerformanceTracker
from datetime import datetime

tracker = PerformanceTracker()

# Test recording an entry
tracker.record_entry(
    signal_id=1,
    entry_price=50000.0,
    entry_timestamp=datetime.now()
)

# Get win rate (should show 0 since no exits yet)
stats = tracker.get_win_rate()
print(f'\n📊 Current Stats: {stats}')

# Get open positions
positions = tracker.get_open_positions()
print(f'\n📈 Open Positions: {len(positions)}')
print('✅ Performance tracker is working!')
EOF
```

---

## ✅ STEP 3: Add Manual Entry/Exit Interface (20 minutes)

**Create a simple CLI tool to manually record trades:**

```bash
cat > /root/crpbot/scripts/record_trade.py << 'EOF'
#!/usr/bin/env python3
"""
CLI tool to manually record trade entries/exits
Usage:
    python record_trade.py entry <signal_id> <price>
    python record_trade.py exit <signal_id> <price> [reason]
    python record_trade.py stats
"""

import sys
from datetime import datetime
sys.path.insert(0, '/root/crpbot')

from libs.tracking.performance_tracker import PerformanceTracker

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    tracker = PerformanceTracker()
    command = sys.argv[1].lower()

    if command == 'entry':
        if len(sys.argv) < 4:
            print("Usage: record_trade.py entry <signal_id> <price>")
            sys.exit(1)

        signal_id = int(sys.argv[2])
        price = float(sys.argv[3])
        tracker.record_entry(signal_id, price, datetime.now())

    elif command == 'exit':
        if len(sys.argv) < 4:
            print("Usage: record_trade.py exit <signal_id> <price> [reason]")
            sys.exit(1)

        signal_id = int(sys.argv[2])
        price = float(sys.argv[3])
        reason = sys.argv[4] if len(sys.argv) > 4 else 'manual'
        tracker.record_exit(signal_id, price, datetime.now(), reason)

    elif command == 'stats':
        stats = tracker.get_win_rate(days=30)
        print('\n📊 Performance Stats (Last 30 Days):')
        print(f'   Total Trades: {stats["total_trades"]}')
        print(f'   Wins: {stats["wins"]} | Losses: {stats["losses"]}')
        print(f'   Win Rate: {stats["win_rate"]:.1f}%')
        print(f'   Avg Win: {stats["avg_win"]:+.2f}%')
        print(f'   Avg Loss: {stats["avg_loss"]:+.2f}%')
        print(f'   Profit Factor: {stats["profit_factor"]:.2f}')

        print('\n📈 Open Positions:')
        positions = tracker.get_open_positions()
        for pos in positions:
            print(f'   Signal #{pos["signal_id"]}: {pos["symbol"]} {pos["direction"]} @ ${pos["entry_price"]:,.2f}')

        print('\n📜 Recent Trades:')
        trades = tracker.get_recent_trades(limit=10)
        for trade in trades:
            outcome_emoji = '✅' if trade['outcome'] == 'win' else '❌' if trade['outcome'] == 'loss' else '➖'
            print(f'   {outcome_emoji} {trade["symbol"]} {trade["direction"]}: {trade["pnl_percent"]:+.2f}% ({trade["hold_duration_minutes"]}m)')

    else:
        print(f'Unknown command: {command}')
        print(__doc__)
        sys.exit(1)

if __name__ == '__main__':
    main()
EOF

chmod +x /root/crpbot/scripts/record_trade.py
echo '✅ Trade recorder created'
```

**Test it:**

```bash
cd /root/crpbot
.venv/bin/python3 scripts/record_trade.py stats
```

---

## ✅ STEP 4: Add Performance Tab to Dashboard (30 minutes)

**This is for visualization - do this AFTER Steps 1-3 work**

I'll implement this once you confirm Steps 1-3 are working.

---

## ✅ STEP 5: Integrate with V7 Runtime (15 minutes)

**Automatically track theory contributions when signals are generated**

Modify `apps/runtime/v7_runtime.py` to call the performance tracker.

I'll do this once Steps 1-4 are confirmed working.

---

## 🎯 SUCCESS CRITERIA

After completing these steps, you should be able to:

1. **Manually record trades**:
   ```bash
   .venv/bin/python3 scripts/record_trade.py entry 123 50000
   .venv/bin/python3 scripts/record_trade.py exit 123 51000
   .venv/bin/python3 scripts/record_trade.py stats
   ```

2. **See win rate immediately** after recording 10+ trades

3. **Track which signals you actually traded** (vs. which you ignored)

4. **Measure if V7 is profitable** (this is the whole point!)

---

## ⚡ DO THIS NOW

Execute Steps 1, 2, and 3 in order. Should take 40 minutes total.

Once done, you'll have:
- Database ready to track performance
- Python API to record trades
- CLI tool to manage trade records
- Ability to see win rate, P&L, profit factor

**Then we can measure if V7 is actually working.**
