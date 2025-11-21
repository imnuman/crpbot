"""
Performance Tracker for V7 Signals
Tracks trade outcomes, calculates P&L, measures theory performance
"""

from datetime import datetime
from typing import Optional, Dict, List
from sqlalchemy import create_engine, text
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

            # Convert entry_timestamp string to datetime if needed
            if isinstance(entry_timestamp, str):
                from dateutil import parser
                entry_timestamp = parser.parse(entry_timestamp)

            # Get signal direction from database to calculate P&L correctly
            signal_result = session.execute(text('''
                SELECT direction FROM signals WHERE id = :signal_id
            '''), {'signal_id': signal_id}).fetchone()

            direction = signal_result[0] if signal_result else 'long'

            # Calculate P&L based on direction
            if direction.lower() == 'long':
                # LONG: profit when price goes up
                pnl_percent = ((exit_price - entry_price) / entry_price) * 100
                pnl_usd = exit_price - entry_price
            else:  # SHORT
                # SHORT: profit when price goes down
                pnl_percent = ((entry_price - exit_price) / entry_price) * 100
                pnl_usd = entry_price - exit_price

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

    def get_win_rate(self, days: int = 30, strategy: Optional[str] = None) -> Dict[str, float]:
        """Calculate win rate over last N days, optionally filtered by strategy"""
        session = self.Session()
        try:
            # Build query with optional strategy filter
            query = '''
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
                    AVG(CASE WHEN outcome = 'win' THEN pnl_percent ELSE NULL END) as avg_win,
                    AVG(CASE WHEN outcome = 'loss' THEN pnl_percent ELSE NULL END) as avg_loss,
                    AVG(pnl_percent) as avg_pnl
                FROM signal_results sr
                JOIN signals s ON sr.signal_id = s.id
                WHERE outcome IN ('win', 'loss', 'breakeven')
                  AND exit_timestamp >= datetime('now', '-' || :days || ' days')
            '''

            params = {'days': days}
            if strategy:
                query += ' AND s.strategy = :strategy'
                params['strategy'] = strategy

            result = session.execute(text(query), params).fetchone()

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

    def get_recent_trades(self, limit: int = 20, strategy: Optional[str] = None) -> List[Dict]:
        """Get recent closed trades, optionally filtered by strategy"""
        session = self.Session()
        try:
            query = '''
                SELECT
                    sr.signal_id,
                    s.symbol,
                    s.direction,
                    sr.entry_price,
                    sr.exit_price,
                    sr.pnl_percent,
                    sr.outcome,
                    sr.entry_timestamp,
                    sr.exit_timestamp,
                    sr.hold_duration_minutes,
                    s.strategy,
                    s.confidence
                FROM signal_results sr
                JOIN signals s ON sr.signal_id = s.id
                WHERE sr.outcome IN ('win', 'loss', 'breakeven')
            '''

            params = {'limit': limit}
            if strategy:
                query += ' AND s.strategy = :strategy'
                params['strategy'] = strategy

            query += ' ORDER BY sr.exit_timestamp DESC LIMIT :limit'

            results = session.execute(text(query), params).fetchall()

            return [
                {
                    'signal_id': r[0],
                    'symbol': r[1],
                    'direction': r[2],
                    'entry_price': r[3],
                    'exit_price': r[4],
                    'pnl_percent': r[5],
                    'outcome': r[6],
                    'entry_timestamp': r[7],
                    'exit_timestamp': r[8],
                    'hold_duration_minutes': r[9],
                    'strategy': r[10],
                    'confidence': r[11]
                }
                for r in results
            ]
        finally:
            session.close()

    def get_strategy_comparison(self, days: int = 30) -> Dict[str, Dict[str, float]]:
        """Compare performance between v7_full_math and v7_deepseek_only strategies"""
        full_math_stats = self.get_win_rate(days, strategy="v7_full_math")
        deepseek_only_stats = self.get_win_rate(days, strategy="v7_deepseek_only")

        return {
            "v7_full_math": full_math_stats,
            "v7_deepseek_only": deepseek_only_stats,
            "comparison": {
                "win_rate_diff": full_math_stats["win_rate"] - deepseek_only_stats["win_rate"],
                "pnl_diff": full_math_stats["avg_pnl"] - deepseek_only_stats["avg_pnl"],
                "profit_factor_diff": full_math_stats["profit_factor"] - deepseek_only_stats["profit_factor"]
            }
        }

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
