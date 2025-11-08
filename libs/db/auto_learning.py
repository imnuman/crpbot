"""Auto-learning system for pattern tracking and adjustment."""
import hashlib
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from libs.db.database import get_database
from libs.db.models import Pattern, RiskBookSnapshot


class AutoLearningSystem:
    """Auto-learning system for pattern tracking and confidence adjustment."""

    def __init__(self, pattern_sample_floor: int = 10):
        """
        Initialize auto-learning system.

        Args:
            pattern_sample_floor: Minimum samples for pattern influence
        """
        self.pattern_sample_floor = pattern_sample_floor
        self.db = get_database()
        logger.info(f"Auto-learning system initialized (sample floor: {pattern_sample_floor})")

    def _generate_pattern_hash(self, features: dict[str, Any]) -> str:
        """
        Generate pattern hash from features.

        Args:
            features: Feature dictionary

        Returns:
            Pattern hash string
        """
        # Create a deterministic hash from features
        feature_str = str(sorted(features.items()))
        pattern_hash = hashlib.sha256(feature_str.encode()).hexdigest()
        return pattern_hash

    def get_pattern_win_rate(
        self, features: dict[str, Any], pattern_name: str | None = None
    ) -> tuple[float | None, int]:
        """
        Get pattern win rate from database.

        Args:
            features: Feature dictionary
            pattern_name: Pattern name (optional)
            pattern_hash: Pattern hash (optional)

        Returns:
            Tuple of (win_rate, sample_count)
            - win_rate: Win rate (0.0-1.0) or None if pattern not found or sample count < floor
            - sample_count: Number of samples
        """
        pattern_hash = self._generate_pattern_hash(features)

        session = self.db.get_session_direct()
        try:
            pattern = session.query(Pattern).filter(Pattern.pattern_hash == pattern_hash).first()

            if pattern is None:
                return None, 0

            if pattern.total < self.pattern_sample_floor:
                logger.debug(
                    f"Pattern {pattern.name} has {pattern.total} samples < {self.pattern_sample_floor} floor"
                )
                return None, pattern.total

            return pattern.win_rate, pattern.total
        finally:
            session.close()

    def record_pattern_result(
        self, features: dict[str, Any], result: str, pattern_name: str | None = None
    ) -> None:
        """
        Record pattern result (win/loss).

        Args:
            features: Feature dictionary
            result: Result ('win' or 'loss')
            pattern_name: Pattern name (optional)
        """
        pattern_hash = self._generate_pattern_hash(features)

        if pattern_name is None:
            pattern_name = f"pattern_{pattern_hash[:8]}"

        session = self.db.get_session_direct()
        try:
            pattern = session.query(Pattern).filter(Pattern.pattern_hash == pattern_hash).first()

            if pattern is None:
                # Create new pattern
                pattern = Pattern(
                    name=pattern_name,
                    pattern_hash=pattern_hash,
                    wins=1 if result == "win" else 0,
                    total=1,
                    win_rate=1.0 if result == "win" else 0.0,
                )
                session.add(pattern)
                logger.debug(f"Created new pattern: {pattern_name} (hash: {pattern_hash[:8]})")
            else:
                # Update existing pattern
                pattern.total += 1
                if result == "win":
                    pattern.wins += 1
                pattern.win_rate = pattern.wins / pattern.total
                pattern.updated_at = datetime.now(timezone.utc)
                logger.debug(
                    f"Updated pattern: {pattern_name} (wins: {pattern.wins}/{pattern.total}, "
                    f"win_rate: {pattern.win_rate:.2%})"
                )

            session.commit()
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()

    def record_trade(
        self,
        signal_id: str,
        pair: str,
        tier: str,
        entry_time: datetime,
        entry_price: float,
        tp_price: float,
        sl_price: float,
        rr_expected: float,
        mode: str,
        spread_bps: float = 0.0,
        slippage_bps: float = 0.0,
        slippage_expected_bps: float = 0.0,
        latency_ms: float = 0.0,
    ) -> None:
        """
        Record a trade in the risk book.

        Args:
            signal_id: Unique signal ID
            pair: Trading pair
            tier: Confidence tier
            entry_time: Entry timestamp
            entry_price: Entry price
            tp_price: Take-profit price
            sl_price: Stop-loss price
            rr_expected: Expected risk:reward ratio
            mode: Mode ('dryrun' or 'live')
            spread_bps: Spread in basis points
            slippage_bps: Slippage in basis points
            slippage_expected_bps: Expected slippage in basis points
            latency_ms: Latency in milliseconds
        """
        session = self.db.get_session_direct()
        try:
            snapshot = RiskBookSnapshot(
                signal_id=signal_id,
                pair=pair,
                tier=tier,
                entry_time=entry_time,
                entry_price=entry_price,
                tp_price=tp_price,
                sl_price=sl_price,
                rr_expected=rr_expected,
                mode=mode,
                spread_bps=spread_bps,
                slippage_bps=slippage_bps,
                slippage_expected_bps=slippage_expected_bps,
                latency_ms=latency_ms,
            )
            session.add(snapshot)
            session.commit()
            logger.debug(f"Recorded trade: {signal_id} ({pair}, {tier}, {mode})")
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()

    def update_trade_result(
        self,
        signal_id: str,
        result: str,
        exit_time: datetime,
        exit_price: float,
        r_realized: float | None = None,
        time_to_tp_sl_seconds: int | None = None,
    ) -> None:
        """
        Update trade result (win/loss).

        Args:
            signal_id: Signal ID
            result: Result ('win' or 'loss')
            exit_time: Exit timestamp
            exit_price: Exit price
            r_realized: Realized R (risk:reward)
            time_to_tp_sl_seconds: Time to TP/SL in seconds
        """
        session = self.db.get_session_direct()
        try:
            snapshot = session.query(RiskBookSnapshot).filter(RiskBookSnapshot.signal_id == signal_id).first()

            if snapshot is None:
                logger.warning(f"Trade not found: {signal_id}")
                return

            snapshot.result = result
            snapshot.exit_time = exit_time
            snapshot.exit_price = exit_price
            snapshot.r_realized = r_realized
            snapshot.time_to_tp_sl_seconds = time_to_tp_sl_seconds

            session.commit()
            logger.debug(f"Updated trade result: {signal_id} ({result})")
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()

    def get_statistics(
        self, days: int = 30, mode: str | None = None, pair: str | None = None
    ) -> dict[str, Any]:
        """
        Get trading statistics.

        Args:
            days: Number of days to look back
            mode: Filter by mode ('dryrun' or 'live')
            pair: Filter by pair

        Returns:
            Statistics dictionary
        """
        from datetime import timedelta, timezone

        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        session = self.db.get_session_direct()
        try:
            query = session.query(RiskBookSnapshot).filter(RiskBookSnapshot.entry_time >= cutoff_date)

            if mode:
                query = query.filter(RiskBookSnapshot.mode == mode)
            if pair:
                query = query.filter(RiskBookSnapshot.pair == pair)

            trades = query.all()
        finally:
            session.close()

            if not trades:
                return {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "win_rate": 0.0,
                    "total_pnl": 0.0,
                }

            closed_trades = [t for t in trades if t.result is not None]
            winning_trades = [t for t in closed_trades if t.result == "win"]
            losing_trades = [t for t in closed_trades if t.result == "loss"]

            # Calculate PnL (simplified - would need actual position sizes)
            total_pnl = 0.0  # TODO: Calculate from entry/exit prices and positions

            return {
                "total_trades": len(closed_trades),
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "win_rate": len(winning_trades) / len(closed_trades) if closed_trades else 0.0,
                "total_pnl": total_pnl,
            }

