"""
HYDRA 3.0 - Database Module

SQLite database for all HYDRA data:
- Regime history
- Evolved strategies (with genealogy)
- Tournament results
- Trades (paper + live)
- Consensus votes
- Explainability logs
- Lessons learned
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, timezone
from typing import Dict, List, Optional
from loguru import logger
import json

Base = declarative_base()


# ==================== REGIME HISTORY ====================

class RegimeHistory(Base):
    """Track market regime classifications over time."""
    __tablename__ = "regime_history"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    regime = Column(String(20), nullable=False)  # TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE, CHOPPY
    adx = Column(Float)
    atr = Column(Float)
    bb_width = Column(Float)
    confidence = Column(Float)  # 0.0-1.0


# ==================== STRATEGIES ====================

class Strategy(Base):
    """Evolved strategies with breeding genealogy."""
    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True)
    strategy_id = Column(String(50), unique=True, nullable=False, index=True)  # UUID
    name = Column(String(200), nullable=False)
    gladiator = Column(String(20), nullable=False)  # A, B, C, D
    regime = Column(String(20), nullable=False)  # Which regime optimized for
    parent_1_id = Column(String(50))  # Breeding parent 1
    parent_2_id = Column(String(50))  # Breeding parent 2
    generation = Column(Integer, nullable=False)  # 1 = original, 2+ = bred
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    status = Column(String(20), nullable=False, default="ACTIVE")  # ACTIVE, KILLED, CHAMPION

    # Strategy definition (JSON)
    logic = Column(Text, nullable=False)  # Full strategy as JSON
    parameters = Column(Text, nullable=False)  # Parameters as JSON

    # Performance metrics
    backtest_wr = Column(Float)
    backtest_sharpe = Column(Float)
    backtest_trades = Column(Integer)
    paper_wr = Column(Float)
    paper_trades = Column(Integer)
    live_wr = Column(Float)
    live_trades = Column(Integer)


# ==================== TOURNAMENT RESULTS ====================

class TournamentResult(Base):
    """Performance tracking for tournament cycles."""
    __tablename__ = "tournament_results"

    id = Column(Integer, primary_key=True)
    tournament_id = Column(String(50), nullable=False, index=True)  # UUID
    strategy_id = Column(String(50), ForeignKey("strategies.strategy_id"), nullable=False)
    regime = Column(String(20), nullable=False)  # Tournament regime
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)

    # Performance
    total_trades = Column(Integer, nullable=False)
    wins = Column(Integer, nullable=False)
    losses = Column(Integer, nullable=False)
    win_rate = Column(Float, nullable=False)
    pnl_percent = Column(Float, nullable=False)
    sharpe = Column(Float, nullable=False)
    max_drawdown = Column(Float, nullable=False)
    rank = Column(Integer)  # 1 = best, 4 = worst

    # Relationship
    strategy = relationship("Strategy")


# ==================== HYDRA TRADES ====================

class HydraTrade(Base):
    """All trades (paper + micro live)."""
    __tablename__ = "hydra_trades"

    id = Column(Integer, primary_key=True)
    trade_id = Column(String(50), unique=True, nullable=False, index=True)  # HYDRA-001
    timestamp = Column(DateTime, nullable=False, index=True)

    # Asset info
    symbol = Column(String(20), nullable=False, index=True)
    asset_type = Column(String(20), nullable=False)  # exotic_forex, meme_perp
    regime = Column(String(20), nullable=False)

    # Strategy & gladiator
    strategy_id = Column(String(50), ForeignKey("strategies.strategy_id"), nullable=False)
    gladiator = Column(String(20), nullable=False)

    # Consensus
    consensus_level = Column(Float, nullable=False)  # 0.5, 0.75, 1.0
    votes_buy = Column(Integer, nullable=False)
    votes_sell = Column(Integer, nullable=False)
    votes_hold = Column(Integer, nullable=False)

    # Trade details
    direction = Column(String(10), nullable=False)  # BUY, SELL
    entry_price = Column(Float, nullable=False)
    sl_price = Column(Float, nullable=False)
    tp_price = Column(Float, nullable=False)
    position_size_usd = Column(Float, nullable=False)
    risk_percent = Column(Float, nullable=False)

    # Execution
    entry_time = Column(DateTime)
    exit_time = Column(DateTime)
    exit_price = Column(Float)
    exit_reason = Column(String(50))  # TP, SL, TIME, MANUAL

    # Result
    pnl_usd = Column(Float)
    pnl_percent = Column(Float)
    outcome = Column(String(20))  # WIN, LOSS, BREAKEVEN

    # Mode
    mode = Column(String(20), nullable=False)  # PAPER, MICRO_LIVE

    # Relationship
    strategy = relationship("Strategy")


# ==================== CONSENSUS VOTES ====================

class ConsensusVote(Base):
    """Individual engine votes for each trade opportunity."""
    __tablename__ = "consensus_votes"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False)

    # Gladiator vote
    gladiator = Column(String(20), nullable=False)
    strategy_id = Column(String(50), ForeignKey("strategies.strategy_id"))
    vote = Column(String(10), nullable=False)  # BUY, SELL, HOLD
    confidence = Column(Float, nullable=False)
    reasoning = Column(Text)

    # What trade this vote was for (may not execute if no consensus)
    related_trade_id = Column(String(50))

    # Relationship
    strategy = relationship("Strategy")


# ==================== EXPLAINABILITY LOGS ====================

class ExplainabilityLog(Base):
    """Full explanation of why each trade happened (or didn't)."""
    __tablename__ = "explainability_logs"

    id = Column(Integer, primary_key=True)
    trade_id = Column(String(50), ForeignKey("hydra_trades.trade_id"), unique=True)
    timestamp = Column(DateTime, nullable=False)

    # Decision context
    symbol = Column(String(20), nullable=False)
    regime = Column(String(20), nullable=False)
    structural_edge = Column(String(200))  # Which edge was exploited

    # Filters (JSON array of results)
    filters_passed = Column(Text, nullable=False)  # JSON list

    # Consensus details
    gladiators_voted_buy = Column(Text)  # JSON list
    gladiators_voted_sell = Column(Text)
    gladiators_voted_hold = Column(Text)
    consensus_result = Column(String(50))

    # Guardian decision
    guardian_approved = Column(Boolean, nullable=False)
    guardian_reason = Column(Text)
    position_size_original = Column(Float)
    position_size_final = Column(Float)
    adjustment_reason = Column(String(200))

    # Full context (JSON)
    full_context = Column(Text)  # All data that went into decision

    # Relationship
    trade = relationship("HydraTrade")


# ==================== ENGINE HISTORY (STEP 33) ====================

class EngineHistory(Base):
    """Daily snapshots of engine performance for tracking improvement."""
    __tablename__ = "engine_history"

    id = Column(Integer, primary_key=True)
    engine_id = Column(String(10), nullable=False, index=True)  # A, B, C, D
    date = Column(String(10), nullable=False, index=True)  # YYYY-MM-DD
    rank = Column(Integer)
    weight = Column(Float)
    total_trades = Column(Integer)
    wins = Column(Integer)
    losses = Column(Integer)
    win_rate = Column(Float)
    total_pnl_usd = Column(Float)
    total_pnl_percent = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    specialty = Column(String(50))
    specialty_trades = Column(Integer)
    specialty_wins = Column(Integer)
    specialty_accuracy = Column(Float)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# ==================== EDGE GRAVEYARD (STEP 34) ====================

class EdgeGraveyard(Base):
    """Dead edges archived for potential resurrection."""
    __tablename__ = "edge_graveyard"

    id = Column(Integer, primary_key=True)
    edge_id = Column(String(50), unique=True, nullable=False, index=True)
    engine_id = Column(String(10), nullable=False, index=True)
    edge_type = Column(String(50), nullable=False)
    description = Column(Text)
    death_cause = Column(String(100), nullable=False)
    death_date = Column(DateTime, nullable=False)
    final_pnl_percent = Column(Float)
    total_trades = Column(Integer)
    win_rate = Column(Float)
    metadata_json = Column(Text)  # JSON storage
    resurrection_attempts = Column(Integer, default=0)
    last_resurrection_attempt = Column(DateTime)
    resurrect_eligible = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# ==================== TOURNAMENT SESSIONS (STEP 36) ====================

class TournamentSession(Base):
    """Tournament session records."""
    __tablename__ = "tournament_sessions"

    id = Column(Integer, primary_key=True)
    tournament_id = Column(String(50), unique=True, nullable=False, index=True)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    status = Column(String(20), default="active")  # active, completed
    cycles_completed = Column(Integer, default=0)
    total_trades = Column(Integer, default=0)
    winner_engine = Column(String(10))
    winner_pnl_usd = Column(Float)
    winner_win_rate = Column(Float)
    rankings_json = Column(Text)  # JSON storage
    breeding_events = Column(Integer, default=0)
    kill_events = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# ==================== LESSONS LEARNED ====================

class LessonLearned(Base):
    """Mistakes that resulted in new filters."""
    __tablename__ = "lessons_learned"

    id = Column(Integer, primary_key=True)
    lesson_id = Column(String(50), unique=True, nullable=False, index=True)  # L001, L002, etc.
    date = Column(DateTime, nullable=False)

    # What went wrong
    asset = Column(String(20), nullable=False)
    loss_amount_usd = Column(Float, nullable=False)
    loss_percent = Column(Float, nullable=False)
    loss_reason = Column(Text, nullable=False)

    # What we learned
    lesson = Column(Text, nullable=False)
    filter_added = Column(Text)  # Description of new filter

    # Status
    status = Column(String(20), nullable=False, default="ACTIVE")  # ACTIVE, DEPRECATED

    # Related trade
    related_trade_id = Column(String(50), ForeignKey("hydra_trades.trade_id"))

    # Relationship
    trade = relationship("HydraTrade")


# ==================== DATABASE CLASS ====================

class HydraDatabase:
    """
    Main database interface for HYDRA 3.0.
    """

    def __init__(self, db_path: str = "data/hydra/hydra.db"):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        logger.info(f"HYDRA database initialized: {db_path}")

    # ==================== REGIME METHODS ====================

    def store_regime(
        self,
        symbol: str,
        regime: str,
        adx: float,
        atr: float,
        bb_width: Optional[float] = None,
        confidence: Optional[float] = None
    ) -> RegimeHistory:
        """Store regime classification."""
        session = self.Session()
        try:
            regime_entry = RegimeHistory(
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                regime=regime,
                adx=adx,
                atr=atr,
                bb_width=bb_width,
                confidence=confidence
            )
            session.add(regime_entry)
            session.commit()
            logger.debug(f"Regime stored: {symbol} = {regime}")
            return regime_entry
        finally:
            session.close()

    def get_recent_regimes(self, symbol: str, hours: int = 24) -> List[RegimeHistory]:
        """Get recent regime history for a symbol."""
        session = self.Session()
        try:
            cutoff = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            return session.query(RegimeHistory).filter(
                RegimeHistory.symbol == symbol,
                RegimeHistory.timestamp >= cutoff
            ).order_by(RegimeHistory.timestamp.desc()).limit(hours).all()
        finally:
            session.close()

    # ==================== STRATEGY METHODS ====================

    def store_strategy(self, strategy_data: Dict) -> Strategy:
        """Store a new strategy."""
        session = self.Session()
        try:
            strategy = Strategy(
                strategy_id=strategy_data["strategy_id"],
                name=strategy_data["name"],
                gladiator=strategy_data["gladiator"],
                regime=strategy_data["regime"],
                parent_1_id=strategy_data.get("parent_1_id"),
                parent_2_id=strategy_data.get("parent_2_id"),
                generation=strategy_data["generation"],
                logic=json.dumps(strategy_data["logic"]),
                parameters=json.dumps(strategy_data["parameters"]),
                backtest_wr=strategy_data.get("backtest_wr"),
                backtest_sharpe=strategy_data.get("backtest_sharpe"),
                backtest_trades=strategy_data.get("backtest_trades")
            )
            session.add(strategy)
            session.commit()
            logger.info(f"Strategy stored: {strategy.name} ({strategy.strategy_id})")
            return strategy
        finally:
            session.close()

    def get_active_strategies(self, regime: Optional[str] = None) -> List[Strategy]:
        """Get all active strategies, optionally filtered by regime."""
        session = self.Session()
        try:
            query = session.query(Strategy).filter(Strategy.status == "ACTIVE")
            if regime:
                query = query.filter(Strategy.regime == regime)
            return query.all()
        finally:
            session.close()

    def update_strategy_performance(
        self,
        strategy_id: str,
        paper_wr: Optional[float] = None,
        paper_trades: Optional[int] = None,
        live_wr: Optional[float] = None,
        live_trades: Optional[int] = None
    ):
        """Update strategy performance metrics."""
        session = self.Session()
        try:
            strategy = session.query(Strategy).filter(
                Strategy.strategy_id == strategy_id
            ).first()

            if strategy:
                if paper_wr is not None:
                    strategy.paper_wr = paper_wr
                if paper_trades is not None:
                    strategy.paper_trades = paper_trades
                if live_wr is not None:
                    strategy.live_wr = live_wr
                if live_trades is not None:
                    strategy.live_trades = live_trades
                session.commit()
        finally:
            session.close()

    # ==================== TRADE METHODS ====================

    def store_trade(self, trade_data: Dict) -> HydraTrade:
        """Store a new trade."""
        session = self.Session()
        try:
            trade = HydraTrade(
                trade_id=trade_data["trade_id"],
                timestamp=datetime.now(timezone.utc),
                symbol=trade_data["symbol"],
                asset_type=trade_data["asset_type"],
                regime=trade_data["regime"],
                strategy_id=trade_data["strategy_id"],
                gladiator=trade_data["gladiator"],
                consensus_level=trade_data["consensus_level"],
                votes_buy=trade_data["votes_buy"],
                votes_sell=trade_data["votes_sell"],
                votes_hold=trade_data["votes_hold"],
                direction=trade_data["direction"],
                entry_price=trade_data["entry_price"],
                sl_price=trade_data["sl_price"],
                tp_price=trade_data["tp_price"],
                position_size_usd=trade_data["position_size_usd"],
                risk_percent=trade_data["risk_percent"],
                mode=trade_data["mode"]
            )
            session.add(trade)
            session.commit()
            logger.info(f"Trade stored: {trade.trade_id} ({trade.symbol} {trade.direction})")
            return trade
        finally:
            session.close()

    def update_trade_result(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
        pnl_usd: float,
        pnl_percent: float,
        outcome: str
    ):
        """Update trade with exit results."""
        session = self.Session()
        try:
            trade = session.query(HydraTrade).filter(
                HydraTrade.trade_id == trade_id
            ).first()

            if trade:
                trade.exit_time = datetime.now(timezone.utc)
                trade.exit_price = exit_price
                trade.exit_reason = exit_reason
                trade.pnl_usd = pnl_usd
                trade.pnl_percent = pnl_percent
                trade.outcome = outcome
                session.commit()
                logger.info(f"Trade result: {trade_id} = {outcome} ({pnl_percent:.2%})")
        finally:
            session.close()

    # ==================== EXPLAINABILITY METHODS ====================

    def store_explainability(self, explain_data: Dict) -> ExplainabilityLog:
        """Store explainability log for a trade decision."""
        session = self.Session()
        try:
            log = ExplainabilityLog(
                trade_id=explain_data["trade_id"],
                timestamp=datetime.now(timezone.utc),
                symbol=explain_data["symbol"],
                regime=explain_data["regime"],
                structural_edge=explain_data.get("structural_edge"),
                filters_passed=json.dumps(explain_data["filters_passed"]),
                gladiators_voted_buy=json.dumps(explain_data.get("gladiators_voted_buy", [])),
                gladiators_voted_sell=json.dumps(explain_data.get("gladiators_voted_sell", [])),
                gladiators_voted_hold=json.dumps(explain_data.get("gladiators_voted_hold", [])),
                consensus_result=explain_data["consensus_result"],
                guardian_approved=explain_data["guardian_approved"],
                guardian_reason=explain_data.get("guardian_reason"),
                position_size_original=explain_data.get("position_size_original"),
                position_size_final=explain_data.get("position_size_final"),
                adjustment_reason=explain_data.get("adjustment_reason"),
                full_context=json.dumps(explain_data.get("full_context", {}))
            )
            session.add(log)
            session.commit()
            return log
        finally:
            session.close()

    # ==================== LESSON METHODS ====================

    def store_lesson(self, lesson_data: Dict) -> LessonLearned:
        """Store a lesson learned from a losing trade."""
        session = self.Session()
        try:
            lesson = LessonLearned(
                lesson_id=lesson_data["lesson_id"],
                date=datetime.now(timezone.utc),
                asset=lesson_data["asset"],
                loss_amount_usd=lesson_data["loss_amount_usd"],
                loss_percent=lesson_data["loss_percent"],
                loss_reason=lesson_data["loss_reason"],
                lesson=lesson_data["lesson"],
                filter_added=lesson_data.get("filter_added"),
                related_trade_id=lesson_data.get("related_trade_id")
            )
            session.add(lesson)
            session.commit()
            logger.warning(f"Lesson learned: {lesson.lesson_id} - {lesson.lesson}")
            return lesson
        finally:
            session.close()

    def get_all_lessons(self, active_only: bool = True) -> List[LessonLearned]:
        """Get all lessons learned."""
        session = self.Session()
        try:
            query = session.query(LessonLearned)
            if active_only:
                query = query.filter(LessonLearned.status == "ACTIVE")
            return query.order_by(LessonLearned.date.desc()).all()
        finally:
            session.close()


# ==================== SINGLETON & INITIALIZATION ====================

_hydra_db = None

def init_hydra_db(db_path: str = "data/hydra/hydra.db") -> HydraDatabase:
    """Initialize and return singleton HYDRA database instance."""
    global _hydra_db
    if _hydra_db is None:
        _hydra_db = HydraDatabase(db_path=db_path)
    return _hydra_db


def get_hydra_session():
    """Get a new database session from the singleton database."""
    global _hydra_db
    if _hydra_db is None:
        _hydra_db = init_hydra_db()
    return _hydra_db.Session()

# Backwards compatibility alias
HydraSession = get_hydra_session
