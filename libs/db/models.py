"""SQLAlchemy database models for trading system."""
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
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

    def __repr__(self) -> str:
        return (
            f"<Signal(id={self.id}, symbol={self.symbol}, "
            f"tier={self.tier}, confidence={self.confidence:.2f})>"
        )


class Pattern(Base):
    """Pattern learning and tracking."""

    __tablename__ = "patterns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    discovered_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    pattern_type = Column(String(50), nullable=False)  # e.g., 'high_volatility_breakout'

    # Pattern characteristics
    session = Column(String(20))
    volatility_regime = Column(String(20))  # 'low', 'medium', 'high'
    timeframe = Column(String(10))

    # Performance metrics
    total_occurrences = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float)
    avg_pnl = Column(Float)
    sharpe_ratio = Column(Float)

    # Pattern definition
    features = Column(Text)  # JSON-encoded feature values
    threshold_values = Column(Text)  # JSON-encoded thresholds

    # Metadata
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    active = Column(Integer, default=1)  # 0=inactive, 1=active

    def __repr__(self) -> str:
        return f"<Pattern(id={self.id}, type={self.pattern_type}, win_rate={self.win_rate:.2%})>"


class RiskBookSnapshot(Base):
    """Risk book snapshots for tracking account state."""

    __tablename__ = "risk_book_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Account metrics
    balance = Column(Float, nullable=False)
    equity = Column(Float, nullable=False)
    daily_pnl = Column(Float, nullable=False)
    total_pnl = Column(Float, nullable=False)

    # Risk metrics
    daily_loss_limit = Column(Float, nullable=False)
    total_loss_limit = Column(Float, nullable=False)
    daily_loss_used_pct = Column(Float, nullable=False)
    total_loss_used_pct = Column(Float, nullable=False)

    # Position metrics
    open_positions = Column(Integer, default=0)
    total_exposure = Column(Float, default=0.0)

    # Signal rate limiting
    signals_last_hour = Column(Integer, default=0)
    high_tier_signals_last_hour = Column(Integer, default=0)

    # Metadata
    ftmo_account = Column(String(50))
    notes = Column(Text)

    def __repr__(self) -> str:
        return f"<RiskBook(balance=${self.balance:.2f}, daily_pnl=${self.daily_pnl:.2f})>"


def create_tables(db_url: str = "sqlite:///tradingai.db") -> None:
    """Create all database tables.

    Args:
        db_url: Database connection URL
    """
    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)


def get_session(db_url: str = "sqlite:///tradingai.db"):
    """Get a database session.

    Args:
        db_url: Database connection URL

    Returns:
        SQLAlchemy session
    """
    engine = create_engine(db_url, echo=False)
    Session = sessionmaker(bind=engine)
    return Session()
