"""
HYDRA 3.0 - Knowledge Transfer (Winner Teaches Losers)

After each trading cycle, the winner (#1) exports insights to all losers.

Flow:
1. Winner (#1) exports winning insights
2. All losers (ranks 2-4) receive insights
3. Losers choose response strategy:
   - IMPROVE: Adopt winner's approach
   - COUNTER: Build strategy to beat winner
   - INVENT: Create entirely new approach
4. Lessons stored in database for future reference

This creates competitive pressure and rapid evolution.

Phase 2, Week 2 - Step 14
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger
import json
import uuid
import random


@dataclass
class WinningInsight:
    """Single insight extracted from winner's performance"""
    insight_id: str
    source_engine: str
    insight_type: str  # entry, exit, timing, sizing, regime, risk
    description: str
    evidence: List[str]  # Trade IDs that demonstrate this
    confidence: float  # 0-1, based on sample size
    created_at: datetime


@dataclass
class LearningResponse:
    """How a loser responded to winner's insights"""
    response_id: str
    learner_engine: str
    teacher_engine: str
    strategy_chosen: str  # IMPROVE, COUNTER, INVENT
    insights_received: List[str]
    adaptation_plan: str
    timestamp: datetime


@dataclass
class TeachingSession:
    """Record of one knowledge transfer session"""
    session_id: str
    timestamp: datetime
    teacher_engine: str
    teacher_rank: int
    teacher_pnl: float
    teacher_win_rate: float
    learners: List[str]
    insights_shared: List[WinningInsight]
    responses: List[LearningResponse]
    cycle_number: int


class KnowledgeTransfer:
    """
    Knowledge Transfer System - Winner Teaches Losers.

    After each cycle:
    1. Extract insights from winner (#1)
    2. Broadcast to all losers (ranks 2-4)
    3. Each loser decides response strategy
    4. Store session for analysis

    MOD 10 - Anti-homogenization:
    - Engines CANNOT adopt insights that overlap with their specialty
    - Each engine must maintain its unique edge
    - COUNTER and INVENT strategies preferred for diversity
    """

    # Response strategy weights (for random selection)
    # MOD 10: Reduced IMPROVE weight to prevent homogenization
    RESPONSE_STRATEGIES = {
        "IMPROVE": 0.30,   # 30% chance - adopt winner's approach (reduced from 50%)
        "COUNTER": 0.40,   # 40% chance - build counter-strategy (increased from 30%)
        "INVENT": 0.30     # 30% chance - create new approach (increased from 20%)
    }

    # MOD 10: Engine specialties for anti-homogenization
    ENGINE_SPECIALTIES = {
        "A": "liquidation_cascade",
        "B": "funding_extreme",
        "C": "orderbook_imbalance",
        "D": "regime_transition",
    }

    def __init__(self, data_dir: Optional[Path] = None):
        # Auto-detect data directory based on environment
        if data_dir is None:
            import os
            if os.path.exists("/root/crpbot"):
                data_dir = Path("/root/crpbot/data/hydra")
            else:
                data_dir = Path.home() / "crpbot" / "data" / "hydra"

        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Session history
        self.session_history: List[TeachingSession] = []
        self.session_count = 0

        # Persistence files
        self.state_file = self.data_dir / "knowledge_transfer_state.json"
        self.sessions_file = self.data_dir / "teaching_sessions.jsonl"
        self.insights_file = self.data_dir / "shared_insights.jsonl"

        # Load state
        self._load_state()

        logger.info(f"[KnowledgeTransfer] Initialized. Sessions: {self.session_count}")

    def execute_knowledge_transfer(
        self,
        rankings: List[tuple],
        get_portfolio_fn,
        get_engine_fn
    ) -> Optional[TeachingSession]:
        """
        Execute knowledge transfer from winner to all losers.

        Args:
            rankings: List of (engine_name, stats) sorted by rank
            get_portfolio_fn: Function to get portfolio by engine name
            get_engine_fn: Function to get engine instance by name

        Returns:
            TeachingSession if executed, None otherwise
        """
        if not rankings or len(rankings) < 2:
            logger.debug("[KnowledgeTransfer] Not enough engines for transfer")
            return None

        self.session_count += 1
        now = datetime.now(timezone.utc)

        # Identify teacher (winner) and learners (losers)
        teacher_name, teacher_stats = rankings[0]
        learners = [(name, stats) for name, stats in rankings[1:]]

        logger.info(f"\n{'='*60}")
        logger.info(f"📚 KNOWLEDGE TRANSFER SESSION #{self.session_count} 📚")
        logger.info(f"{'='*60}")
        logger.info(f"Teacher: Engine {teacher_name} (Rank #1)")
        logger.info(f"  - P&L: ${teacher_stats.total_pnl_usd:+.2f}")
        logger.info(f"  - Win Rate: {teacher_stats.win_rate:.1%}")
        logger.info(f"Learners: {[l[0] for l in learners]}")
        logger.info(f"{'='*60}\n")

        # Step 1: Extract insights from winner
        teacher_portfolio = get_portfolio_fn(teacher_name)
        insights = self._extract_winning_insights(teacher_name, teacher_portfolio)

        if not insights:
            logger.warning("[KnowledgeTransfer] No insights extracted from winner")
            return None

        # Step 2: Share insights with each learner
        responses = []
        for learner_name, learner_stats in learners:
            learner_engine = get_engine_fn(learner_name)
            response = self._transfer_to_learner(
                teacher_name=teacher_name,
                learner_engine=learner_engine,
                learner_name=learner_name,
                learner_rank=learner_stats.current_rank,
                insights=insights
            )
            responses.append(response)

        # Step 3: Create session record
        session = TeachingSession(
            session_id=str(uuid.uuid4())[:8],
            timestamp=now,
            teacher_engine=teacher_name,
            teacher_rank=1,
            teacher_pnl=teacher_stats.total_pnl_usd,
            teacher_win_rate=teacher_stats.win_rate,
            learners=[l[0] for l in learners],
            insights_shared=insights,
            responses=responses,
            cycle_number=self.session_count
        )

        # Save session
        self.session_history.append(session)
        self._save_session(session)
        self._save_state()

        # Log summary
        logger.success(f"\n📚 Knowledge transfer complete:")
        logger.success(f"   Insights shared: {len(insights)}")
        for r in responses:
            logger.success(f"   Engine {r.learner_engine}: {r.strategy_chosen}")

        return session

    def _extract_winning_insights(
        self,
        engine: str,
        portfolio
    ) -> List[WinningInsight]:
        """
        Extract actionable insights from winner's trading history.

        Categories:
        - Entry timing
        - Exit discipline
        - Position sizing
        - Regime awareness
        - Risk management
        """
        insights = []
        now = datetime.now(timezone.utc)

        closed_trades = portfolio.get_closed_trades()
        winning_trades = [t for t in closed_trades if t.outcome == "win"]
        losing_trades = [t for t in closed_trades if t.outcome == "loss"]

        if not closed_trades:
            return insights

        # Insight 1: Win rate pattern
        win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
        if win_rate >= 0.55:
            insights.append(WinningInsight(
                insight_id=str(uuid.uuid4())[:8],
                source_engine=engine,
                insight_type="consistency",
                description=f"Maintains {win_rate:.1%} win rate across {len(closed_trades)} trades",
                evidence=[t.trade_id for t in winning_trades[:5]],
                confidence=min(0.9, len(closed_trades) / 50),
                created_at=now
            ))

        # Insight 2: Direction bias
        if winning_trades:
            long_wins = [t for t in winning_trades if t.direction == "BUY"]
            short_wins = [t for t in winning_trades if t.direction == "SELL"]

            if len(long_wins) > len(short_wins) * 1.5:
                insights.append(WinningInsight(
                    insight_id=str(uuid.uuid4())[:8],
                    source_engine=engine,
                    insight_type="direction",
                    description=f"LONG bias effective: {len(long_wins)}/{len(winning_trades)} wins are longs",
                    evidence=[t.trade_id for t in long_wins[:3]],
                    confidence=0.75,
                    created_at=now
                ))
            elif len(short_wins) > len(long_wins) * 1.5:
                insights.append(WinningInsight(
                    insight_id=str(uuid.uuid4())[:8],
                    source_engine=engine,
                    insight_type="direction",
                    description=f"SHORT bias effective: {len(short_wins)}/{len(winning_trades)} wins are shorts",
                    evidence=[t.trade_id for t in short_wins[:3]],
                    confidence=0.75,
                    created_at=now
                ))

        # Insight 3: Risk/Reward analysis
        rr_ratios = []
        for t in winning_trades:
            if t.entry_price > 0 and t.stop_loss > 0 and t.take_profit > 0:
                risk = abs(t.entry_price - t.stop_loss)
                reward = abs(t.take_profit - t.entry_price)
                if risk > 0:
                    rr_ratios.append(reward / risk)

        if rr_ratios:
            avg_rr = sum(rr_ratios) / len(rr_ratios)
            insights.append(WinningInsight(
                insight_id=str(uuid.uuid4())[:8],
                source_engine=engine,
                insight_type="risk_management",
                description=f"Uses {avg_rr:.1f}:1 reward/risk ratio on winning trades",
                evidence=[t.trade_id for t in winning_trades[:3]],
                confidence=0.80,
                created_at=now
            ))

        # Insight 4: Average win size
        if winning_trades:
            avg_win = sum(t.pnl_percent for t in winning_trades) / len(winning_trades)
            insights.append(WinningInsight(
                insight_id=str(uuid.uuid4())[:8],
                source_engine=engine,
                insight_type="sizing",
                description=f"Average winning trade: +{avg_win:.2f}%",
                evidence=[t.trade_id for t in winning_trades[:3]],
                confidence=0.70,
                created_at=now
            ))

        # Insight 5: Loss control
        if losing_trades:
            avg_loss = sum(t.pnl_percent for t in losing_trades) / len(losing_trades)
            if abs(avg_loss) < 2.0:  # Good loss control
                insights.append(WinningInsight(
                    insight_id=str(uuid.uuid4())[:8],
                    source_engine=engine,
                    insight_type="risk_management",
                    description=f"Excellent loss control: avg loss only {avg_loss:.2f}%",
                    evidence=[t.trade_id for t in losing_trades[:3]],
                    confidence=0.85,
                    created_at=now
                ))

        # Save insights to file
        self._save_insights(insights)

        logger.info(f"[KnowledgeTransfer] Extracted {len(insights)} insights from Engine {engine}")
        return insights

    def _transfer_to_learner(
        self,
        teacher_name: str,
        learner_engine,
        learner_name: str,
        learner_rank: int,
        insights: List[WinningInsight]
    ) -> LearningResponse:
        """
        Transfer insights to a learner and get their response strategy.

        Learner chooses:
        - IMPROVE: Adopt winner's patterns
        - COUNTER: Build strategy to exploit winner's weaknesses
        - INVENT: Create entirely new approach
        """
        now = datetime.now(timezone.utc)

        # Select response strategy (weighted random)
        strategy = self._select_response_strategy(learner_rank)

        # Create adaptation plan based on strategy
        adaptation_plan = self._create_adaptation_plan(
            strategy=strategy,
            teacher_name=teacher_name,
            learner_name=learner_name,
            insights=insights
        )

        # Inject learning into engine
        self._inject_learning(
            engine=learner_engine,
            teacher=teacher_name,
            strategy=strategy,
            insights=insights,
            adaptation_plan=adaptation_plan
        )

        response = LearningResponse(
            response_id=str(uuid.uuid4())[:8],
            learner_engine=learner_name,
            teacher_engine=teacher_name,
            strategy_chosen=strategy,
            insights_received=[i.insight_id for i in insights],
            adaptation_plan=adaptation_plan,
            timestamp=now
        )

        logger.info(
            f"[KnowledgeTransfer] Engine {learner_name} (Rank #{learner_rank}) "
            f"chose {strategy} strategy"
        )

        return response

    def _select_response_strategy(
        self,
        learner_rank: int,
        learner_name: str = None,
        teacher_name: str = None
    ) -> str:
        """
        Select response strategy based on rank and weighted probabilities.

        MOD 10 - Anti-homogenization rules:
        - If teacher and learner have DIFFERENT specialties: Normal selection
        - If teacher has specialty learner shouldn't copy: Force COUNTER or INVENT
        - Engines must maintain their unique edge

        Lower ranks are more likely to IMPROVE (adopt winner's approach).
        Higher ranks are more likely to COUNTER or INVENT.
        """
        # MOD 10: Check anti-homogenization
        block_improve = False
        if learner_name and teacher_name:
            learner_specialty = self.ENGINE_SPECIALTIES.get(learner_name)
            teacher_specialty = self.ENGINE_SPECIALTIES.get(teacher_name)

            if learner_specialty and teacher_specialty:
                # If different specialties, learner should NOT adopt teacher's approach
                # This prevents engines from becoming clones
                if learner_specialty != teacher_specialty:
                    block_improve = True
                    logger.debug(
                        f"[KnowledgeTransfer] Anti-homogenization: Engine {learner_name} "
                        f"({learner_specialty}) blocked from IMPROVE with Engine {teacher_name} "
                        f"({teacher_specialty})"
                    )

        # Adjust weights based on rank
        weights = self.RESPONSE_STRATEGIES.copy()

        if learner_rank == 4:  # Last place - more desperate, likely to IMPROVE
            weights["IMPROVE"] = 0.70
            weights["COUNTER"] = 0.20
            weights["INVENT"] = 0.10
        elif learner_rank == 3:  # Third place - balanced
            weights["IMPROVE"] = 0.50
            weights["COUNTER"] = 0.30
            weights["INVENT"] = 0.20
        else:  # Second place - more likely to COUNTER
            weights["IMPROVE"] = 0.30
            weights["COUNTER"] = 0.50
            weights["INVENT"] = 0.20

        # MOD 10: If IMPROVE is blocked, redistribute to COUNTER and INVENT
        if block_improve:
            improve_weight = weights["IMPROVE"]
            weights["IMPROVE"] = 0.0
            weights["COUNTER"] += improve_weight * 0.6  # 60% goes to COUNTER
            weights["INVENT"] += improve_weight * 0.4   # 40% goes to INVENT
            logger.info(
                f"[KnowledgeTransfer] Anti-homogenization active: "
                f"COUNTER={weights['COUNTER']:.0%}, INVENT={weights['INVENT']:.0%}"
            )

        # Weighted random selection
        r = random.random()
        cumulative = 0
        for strategy, weight in weights.items():
            cumulative += weight
            if r <= cumulative:
                return strategy

        return "COUNTER"  # Default (changed from IMPROVE for anti-homogenization)

    def _create_adaptation_plan(
        self,
        strategy: str,
        teacher_name: str,
        learner_name: str,
        insights: List[WinningInsight]
    ) -> str:
        """Create specific adaptation plan based on chosen strategy."""

        if strategy == "IMPROVE":
            plan = f"""
ADAPTATION STRATEGY: IMPROVE (Learn from Winner)

You are learning from Engine {teacher_name}'s success.

KEY INSIGHTS TO ADOPT:
"""
            for i, insight in enumerate(insights[:5], 1):
                plan += f"\n{i}. [{insight.insight_type}] {insight.description}"

            plan += f"""

ACTION PLAN:
- Study {teacher_name}'s entry patterns
- Mirror their risk management approach
- Adopt their win rate maintaining discipline
- Apply similar position sizing rules

GOAL: Match or exceed {teacher_name}'s performance by adopting proven methods.
"""

        elif strategy == "COUNTER":
            plan = f"""
ADAPTATION STRATEGY: COUNTER (Exploit Winner's Weaknesses)

You are building a counter-strategy against Engine {teacher_name}.

WINNER'S KNOWN PATTERNS:
"""
            for i, insight in enumerate(insights[:5], 1):
                plan += f"\n{i}. [{insight.insight_type}] {insight.description}"

            plan += f"""

COUNTER TACTICS:
- If {teacher_name} favors LONGs → look for SHORT opportunities they miss
- If {teacher_name} uses tight stops → use wider stops for same setups
- Find market conditions where {teacher_name}'s approach fails
- Exploit their blind spots

GOAL: Beat {teacher_name} by trading what they avoid.
"""

        else:  # INVENT
            plan = f"""
ADAPTATION STRATEGY: INVENT (Create New Approach)

You are creating an entirely new strategy, independent of Engine {teacher_name}.

LESSONS ACKNOWLEDGED (but not copied):
"""
            for i, insight in enumerate(insights[:3], 1):
                plan += f"\n{i}. [{insight.insight_type}] {insight.description}"

            plan += """

INVENTION GUIDELINES:
- DO NOT copy the winner's approach
- Find unexplored market inefficiencies
- Test unconventional entry/exit rules
- Embrace creative risk parameters
- Seek edges others haven't found

GOAL: Discover a completely novel profitable strategy.
"""

        return plan

    def _inject_learning(
        self,
        engine,
        teacher: str,
        strategy: str,
        insights: List[WinningInsight],
        adaptation_plan: str
    ):
        """Inject learning into engine's state."""

        # Store learning session
        if hasattr(engine, 'learning_history'):
            if not isinstance(engine.learning_history, list):
                engine.learning_history = []
            engine.learning_history.append({
                "session": self.session_count,
                "teacher": teacher,
                "strategy": strategy,
                "insights_count": len(insights),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

        # Store current adaptation plan
        if hasattr(engine, 'adaptation_plan'):
            engine.adaptation_plan = adaptation_plan

        # Update meta state
        if hasattr(engine, 'meta_state'):
            engine.meta_state['last_learning_session'] = self.session_count
            engine.meta_state['learning_strategy'] = strategy
            engine.meta_state['teacher'] = teacher

    def _save_insights(self, insights: List[WinningInsight]):
        """Save insights to file."""
        with open(self.insights_file, 'a') as f:
            for insight in insights:
                insight_dict = {
                    "insight_id": insight.insight_id,
                    "source_engine": insight.source_engine,
                    "insight_type": insight.insight_type,
                    "description": insight.description,
                    "evidence": insight.evidence,
                    "confidence": insight.confidence,
                    "created_at": insight.created_at.isoformat(),
                    "session": self.session_count
                }
                f.write(json.dumps(insight_dict) + '\n')

    def _save_session(self, session: TeachingSession):
        """Save teaching session to file."""
        with open(self.sessions_file, 'a') as f:
            session_dict = {
                "session_id": session.session_id,
                "timestamp": session.timestamp.isoformat(),
                "teacher_engine": session.teacher_engine,
                "teacher_rank": session.teacher_rank,
                "teacher_pnl": session.teacher_pnl,
                "teacher_win_rate": session.teacher_win_rate,
                "learners": session.learners,
                "insights_count": len(session.insights_shared),
                "responses": [
                    {
                        "learner": r.learner_engine,
                        "strategy": r.strategy_chosen
                    }
                    for r in session.responses
                ],
                "cycle_number": session.cycle_number
            }
            f.write(json.dumps(session_dict) + '\n')

    def _save_state(self):
        """Save knowledge transfer state."""
        state = {
            "session_count": self.session_count,
            "recent_sessions": [
                {
                    "session_id": s.session_id,
                    "timestamp": s.timestamp.isoformat(),
                    "teacher": s.teacher_engine,
                    "learners": s.learners
                }
                for s in self.session_history[-10:]
            ]
        }

        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load knowledge transfer state."""
        if not self.state_file.exists():
            return

        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            self.session_count = state.get("session_count", 0)
            logger.debug(f"[KnowledgeTransfer] Loaded state: {self.session_count} sessions")

        except Exception as e:
            logger.warning(f"[KnowledgeTransfer] Failed to load state: {e}")

    def get_transfer_summary(self) -> Dict[str, Any]:
        """Get summary of knowledge transfer history."""
        return {
            "total_sessions": self.session_count,
            "strategy_distribution": self._get_strategy_distribution(),
            "recent_sessions": [
                {
                    "session_id": s.session_id,
                    "teacher": s.teacher_engine,
                    "learners": s.learners,
                    "insights_shared": len(s.insights_shared)
                }
                for s in self.session_history[-5:]
            ]
        }

    def _get_strategy_distribution(self) -> Dict[str, int]:
        """Get distribution of chosen strategies across all sessions."""
        distribution = {"IMPROVE": 0, "COUNTER": 0, "INVENT": 0}

        for session in self.session_history:
            for response in session.responses:
                if response.strategy_chosen in distribution:
                    distribution[response.strategy_chosen] += 1

        return distribution


# ==================== SINGLETON PATTERN ====================

_knowledge_transfer: Optional[KnowledgeTransfer] = None

def get_knowledge_transfer() -> KnowledgeTransfer:
    """Get singleton instance of KnowledgeTransfer."""
    global _knowledge_transfer
    if _knowledge_transfer is None:
        _knowledge_transfer = KnowledgeTransfer()
    return _knowledge_transfer
