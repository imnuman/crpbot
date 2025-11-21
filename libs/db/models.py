"""Database models for trading system."""
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    Index,
    JSON,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class Signal(Base):
    """Trading signal record."""

    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    direction = Column(String(10), nullable=False)  # 'long' or 'short'
    confidence = Column(Float, nullable=False)
    tier = Column(String(10), nullable=False, index=True)  # 'high', 'medium', 'low'

    # Model predictions
    lstm_prediction = Column(Float)
    transformer_prediction = Column(Float)
    rl_prediction = Column(Float)
    ensemble_prediction = Column(Float, nullable=False)

    # Market context
    session = Column(String(20))  # 'tokyo', 'london', 'new_york'
    entry_price = Column(Float)
    tp_price = Column(Float)
    sl_price = Column(Float)

    # Execution
    executed = Column(Integer, default=0)  # 0=not executed, 1=executed
    execution_time = Column(DateTime)
    execution_price = Column(Float)

    # Result tracking
    result = Column(String(10))  # 'win', 'loss', 'pending', 'skipped'
    exit_time = Column(DateTime)
    exit_price = Column(Float)
    pnl = Column(Float)

    # Metadata
    latency_ms = Column(Float)
    model_version = Column(String(20))
    notes = Column(Text)

    # A/B Testing
    strategy = Column(String(50), default="v7_full_math")  # 'v7_full_math' or 'v7_deepseek_only'

    def __repr__(self) -> str:
        return (
            f"<Signal(id={self.id}, symbol={self.symbol}, "
            f"tier={self.tier}, confidence={self.confidence:.2f})>"
        )


class Pattern(Base):
    """Pattern tracking table for auto-learning."""

    __tablename__ = "patterns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, index=True)
    pattern_hash = Column(String(64), nullable=False, unique=True, index=True)
    wins = Column(Integer, default=0, nullable=False)
    total = Column(Integer, default=0, nullable=False)
    win_rate = Column(Float, default=0.0, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<Pattern(name={self.name}, win_rate={self.win_rate:.2%}, total={self.total})>"


class RiskBookSnapshot(Base):
    """Risk book snapshots table for trade tracking."""

    __tablename__ = "risk_book_snapshots"

    signal_id = Column(String(64), primary_key=True)
    pair = Column(String(20), nullable=False, index=True)
    tier = Column(String(10), nullable=False, index=True)
    entry_time = Column(DateTime, nullable=False, index=True)
    entry_price = Column(Float, nullable=False)
    tp_price = Column(Float, nullable=True)
    sl_price = Column(Float, nullable=True)
    rr_expected = Column(Float, nullable=True)
    result = Column(String(10), nullable=True, index=True)  # 'win', 'loss', None
    exit_time = Column(DateTime, nullable=True)
    exit_price = Column(Float, nullable=True)
    r_realized = Column(Float, nullable=True)
    time_to_tp_sl_seconds = Column(Integer, nullable=True)
    slippage_bps = Column(Float, nullable=True)
    slippage_expected_bps = Column(Float, nullable=True)
    spread_bps = Column(Float, nullable=True)
    latency_ms = Column(Float, nullable=True)
    mode = Column(String(10), nullable=False, index=True)  # 'dryrun' or 'live'
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    __table_args__ = (
        Index("idx_entry_time", "entry_time"),
        Index("idx_pair_mode", "pair", "mode"),
        Index("idx_result_mode", "result", "mode"),
    )

    def __repr__(self) -> str:
        return (
            f"<RiskBookSnapshot(signal_id={self.signal_id}, pair={self.pair}, "
            f"tier={self.tier}, result={self.result})>"
        )


class ModelDeployment(Base):
    """Model deployment tracking table."""

    __tablename__ = "model_deployments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String(50), nullable=False, unique=True, index=True)
    model_path = Column(String(500), nullable=False)
    model_type = Column(String(50), nullable=False)  # 'lstm', 'transformer', 'ensemble'
    deployed_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True
    )
    metrics_json = Column(JSON, nullable=True)
    rollback_reason = Column(Text, nullable=True)
    is_promoted = Column(Boolean, default=False, nullable=False, index=True)
    is_active = Column(Boolean, default=True, nullable=False, index=True)

    def __repr__(self) -> str:
        return (
            f"<ModelDeployment(version={self.version}, model_type={self.model_type}, "
            f"is_promoted={self.is_promoted})>"
        )


class SignalResult(Base):
    """Signal results tracking for performance analysis (paper trading)."""

    __tablename__ = "signal_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(Integer, nullable=False, unique=True, index=True)  # FK to signals.id

    # Entry tracking
    entry_price = Column(Float, nullable=False)
    entry_timestamp = Column(DateTime, nullable=False, index=True)

    # Exit tracking
    exit_price = Column(Float, nullable=True)
    exit_timestamp = Column(DateTime, nullable=True, index=True)
    exit_reason = Column(String(50), nullable=True)  # 'tp_hit', 'sl_hit', 'manual', 'timeout'

    # P&L tracking
    pnl_percent = Column(Float, nullable=True)
    pnl_usd = Column(Float, nullable=True)

    # Outcome
    outcome = Column(String(20), nullable=False, default='open', index=True)  # 'open', 'win', 'loss', 'breakeven'

    # Duration
    hold_duration_minutes = Column(Integer, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        Index("idx_signal_outcome", "signal_id", "outcome"),
        Index("idx_entry_timestamp", "entry_timestamp"),
        Index("idx_exit_timestamp", "exit_timestamp"),
    )

    def __repr__(self) -> str:
        return (
            f"<SignalResult(signal_id={self.signal_id}, outcome={self.outcome}, "
            f"pnl_percent={self.pnl_percent:.2f if self.pnl_percent else 0.0}%)>"
        )


class TheoryPerformance(Base):
    """Track individual theory contributions to signal performance."""

    __tablename__ = "theory_performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(Integer, nullable=False, index=True)  # FK to signals.id
    theory_name = Column(String(100), nullable=False, index=True)
    contribution_score = Column(Float, nullable=False)  # How strongly theory influenced signal
    was_correct = Column(Boolean, nullable=True)  # Whether theory's prediction was correct

    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    __table_args__ = (
        Index("idx_signal_theory", "signal_id", "theory_name"),
        Index("idx_theory_correct", "theory_name", "was_correct"),
    )

    def __repr__(self) -> str:
        return (
            f"<TheoryPerformance(signal_id={self.signal_id}, theory={self.theory_name}, "
            f"contribution={self.contribution_score:.2f})>"
        )


def create_tables(db_url: str = "sqlite:///tradingai.db") -> None:
    """Create all database tables."""

    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)


def get_session(db_url: str = "sqlite:///tradingai.db"):
    """Get a database session."""

    engine = create_engine(db_url, echo=False)
    Session = sessionmaker(bind=engine)
    return Session()
