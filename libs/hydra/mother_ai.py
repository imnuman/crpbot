"""
HYDRA 3.0 - Mother AI (L1 Supervisor)

The Mother AI orchestrates all 4 gladiators in the tournament system.

Responsibilities:
1. Coordinate independent gladiator decisions
2. Manage tournament lifecycle (rankings, weights)
3. Execute breeding mechanism (every 4 days)
4. Implement "Winner Teaches Losers" system
5. Adjust weights based on performance (every 24 hours)
6. Monitor and update gladiator trades
7. Provide market context to all gladiators

Architecture:
- L1 (Mother AI): Strategic oversight, tournament management
- L2 (Gladiators A/B/C/D): Independent traders competing for #1 rank
"""

from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta
from loguru import logger
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from .engine_portfolio import get_tournament_manager
from .regime_detector import RegimeDetector
from .market_data_feeds import get_market_data_aggregator
from .orderbook_feed import get_orderbook_analyzer
from .internet_search import get_internet_search
from .cycles.kill_cycle import get_kill_cycle
from .cycles.breeding_cycle import get_breeding_cycle
from .cycles.knowledge_transfer import get_knowledge_transfer
from .cycles.stats_injector import get_stats_injector
from .cycles.weight_adjuster import get_weight_adjuster

from .engines.engine_a_deepseek import EngineA_DeepSeek
from .engines.engine_b_claude import EngineB_Claude
from .engines.engine_c_grok import EngineC_Grok
from .engines.engine_d_gemini import EngineD_Gemini


# MOD 7: Mother AI fallback state
class MotherAIFailure(Exception):
    """Raised when Mother AI encounters a critical failure."""
    pass


@dataclass
class MotherAIState:
    """Mother AI operational state for fallback handling."""
    is_healthy: bool = True
    is_frozen: bool = False
    failure_reason: Optional[str] = None
    frozen_at: Optional[datetime] = None
    consecutive_failures: int = 0
    max_failures_before_freeze: int = 3  # Freeze after 3 consecutive failures


@dataclass
class TradingCycle:
    """Represents one complete trading cycle."""
    cycle_number: int
    timestamp: datetime
    asset: str
    regime: str
    regime_confidence: float
    decisions_made: int
    trades_opened: int
    gladiators_active: List[str]


class MotherAI:
    """
    Mother AI (L1 Supervisor) - Orchestrates HYDRA 3.0 tournament.

    Core Functions:
    1. Run trading cycles (all 4 gladiators make independent decisions)
    2. Update all gladiator trades (SL/TP monitoring)
    3. Manage tournament rankings and weights
    4. Execute breeding (every 4 days)
    5. Implement Winner Teaches Losers (after breeding)
    6. Adjust weights based on performance (every 24 hours)
    """

    def __init__(self):
        logger.info("Initializing Mother AI (L1 Supervisor)...")

        # Tournament manager (singleton)
        self.tournament_manager = get_tournament_manager()

        # Initialize all 4 engines
        self.gladiators = {
            "A": EngineA_DeepSeek(),
            "B": EngineB_Claude(),
            "C": EngineC_Grok(),
            "D": EngineD_Gemini()
        }

        # Market infrastructure
        self.regime_detector = RegimeDetector()
        self.market_data = get_market_data_aggregator()
        self.orderbook = get_orderbook_analyzer()
        self.search = get_internet_search()

        # Evolution cycles
        self.kill_cycle = get_kill_cycle()
        self.breeding_cycle = get_breeding_cycle()
        self.knowledge_transfer = get_knowledge_transfer()
        self.stats_injector = get_stats_injector()
        self.weight_adjuster = get_weight_adjuster()

        # Tournament state
        self.tournament_start_time = datetime.now(timezone.utc)
        self.last_weight_adjustment = datetime.now(timezone.utc)
        self.last_breeding = datetime.now(timezone.utc)
        self.cycle_count = 0

        # Performance tracking
        self.cycles_history: List[TradingCycle] = []

        # MOD 7: Fallback state - if Mother AI fails, freeze all engines
        self.ai_state = MotherAIState()

        # Data persistence - use config for path
        from .config import MOTHER_AI_STATE_FILE
        self.state_file = MOTHER_AI_STATE_FILE

        logger.success("Mother AI initialized with 4 gladiators (A, B, C, D)")

    # ==================== MAIN ORCHESTRATION ====================

    def run_trading_cycle(self, asset: str, market_data: Dict) -> Optional[TradingCycle]:
        """
        Run one complete trading cycle.

        Flow:
        1. Check if Mother AI is frozen (MOD 7 - fail = freeze all)
        2. Detect market regime
        3. Gather market intelligence (orderbook, search, data feeds)
        4. All 4 gladiators make independent decisions
        5. Open trades for gladiators who decided to trade
        6. Update all existing trades (SL/TP monitoring)
        7. Update tournament rankings

        Args:
            asset: Trading symbol (e.g., "BTC-USD")
            market_data: Current market data (price, volume, etc.)

        Returns:
            TradingCycle summary (or None if frozen)
        """
        # MOD 7: Check if Mother AI is frozen - NO TRADING WITHOUT SUPERVISION
        if self.ai_state.is_frozen:
            logger.critical(
                f"MOTHER AI FROZEN - ALL ENGINES DISABLED | "
                f"Reason: {self.ai_state.failure_reason} | "
                f"Frozen since: {self.ai_state.frozen_at}"
            )
            return None

        self.cycle_count += 1
        cycle_start = datetime.now(timezone.utc)

        logger.info(f"\n{'='*80}")
        logger.info(f"MOTHER AI - CYCLE #{self.cycle_count} - {asset}")
        logger.info(f"{'='*80}\n")

        try:
            cycle = self._execute_cycle_internal(asset, market_data, cycle_start)

            # Cycle succeeded - reset failure counter
            self.ai_state.consecutive_failures = 0
            self.ai_state.is_healthy = True

            return cycle

        except Exception as e:
            # MOD 7: Track failures and freeze if too many consecutive failures
            self.ai_state.consecutive_failures += 1
            self.ai_state.is_healthy = False

            logger.error(
                f"MOTHER AI CYCLE FAILED (attempt {self.ai_state.consecutive_failures}/"
                f"{self.ai_state.max_failures_before_freeze}): {e}"
            )

            if self.ai_state.consecutive_failures >= self.ai_state.max_failures_before_freeze:
                self._freeze_all_engines(f"Critical failure: {e}")

            return None

    def _execute_cycle_internal(self, asset: str, market_data: Dict, cycle_start: datetime) -> TradingCycle:
        """
        Internal cycle execution - wrapped for failure handling.
        """
        # Step 1: Detect regime
        # TODO: Fix this to actually fetch candles and call regime detector properly
        # For now, use mock regime data to get system running
        try:
            # Would need to fetch candles first:
            # candles = self.coinbase.get_candles(asset, granularity="FIVE_MINUTE", limit=200)
            # regime_result = self.regime_detector.detect_regime(symbol=asset, candles=candles)

            # Mock regime for now
            regime = "CHOPPY"
            regime_confidence = 0.65
            logger.info(f"Regime: {regime} (confidence: {regime_confidence:.1%}) [MOCK]")
        except Exception as e:
            logger.warning(f"Regime detection failed, using mock: {e}")
            regime = "CHOPPY"
            regime_confidence = 0.5

        # Step 2: Gather market intelligence
        market_intelligence = self._gather_market_intelligence(asset, market_data)

        # Step 3: All gladiators make independent decisions
        decisions = self._collect_gladiator_decisions(
            asset=asset,
            regime=regime,
            regime_confidence=regime_confidence,
            market_data=market_data,
            market_intelligence=market_intelligence
        )

        # Step 4: Open trades for gladiators who decided to trade
        trades_opened = self._execute_gladiator_trades(decisions)

        # Step 5: Update all existing trades
        self._update_all_trades(market_data)

        # Step 6: Update tournament rankings
        self.tournament_manager.calculate_rankings()

        # Step 7: Check for tournament events (weight adjustment, breeding)
        self._check_tournament_events()

        # Create cycle summary
        cycle = TradingCycle(
            cycle_number=self.cycle_count,
            timestamp=cycle_start,
            asset=asset,
            regime=regime,
            regime_confidence=regime_confidence,
            decisions_made=len(decisions),
            trades_opened=trades_opened,
            gladiators_active=[g for g, d in decisions.items() if d is not None]
        )

        self.cycles_history.append(cycle)

        logger.success(f"Cycle #{self.cycle_count} complete: {trades_opened} trades opened")
        self._log_tournament_standings()

        # Save state to disk for dashboard
        self._save_state()

        return cycle

    def _gather_market_intelligence(self, asset: str, market_data: Dict) -> Dict:
        """
        Gather comprehensive market intelligence for gladiators.

        Returns:
            Dict with orderbook, market feeds, search results
        """
        logger.info("Gathering market intelligence...")

        intelligence = {
            "orderbook": None,
            "market_feeds": None,
            "search_results": None
        }

        # Orderbook analysis
        try:
            orderbook_data = self.orderbook.get_orderbook(asset)
            intelligence["orderbook"] = self.orderbook.analyze_orderbook(orderbook_data)
        except Exception as e:
            logger.warning(f"Orderbook analysis failed: {e}")

        # Market data feeds (funding, liquidations)
        try:
            intelligence["market_feeds"] = self.market_data.get_comprehensive_analysis(asset)
        except Exception as e:
            logger.warning(f"Market feeds failed: {e}")

        # Internet search for recent news (optional, can be slow)
        # Uncomment if you want live news integration:
        # try:
        #     intelligence["search_results"] = self.search.search_crypto_news(asset)
        # except Exception as e:
        #     logger.warning(f"Search failed: {e}")

        return intelligence

    def _collect_gladiator_decisions(
        self,
        asset: str,
        regime: str,
        regime_confidence: float,
        market_data: Dict,
        market_intelligence: Dict
    ) -> Dict[str, Optional[Dict]]:
        """
        Collect independent trading decisions from all 4 gladiators.

        Returns:
            Dict mapping gladiator name -> trade_params (or None if HOLD)
        """
        logger.info("Collecting gladiator decisions (all 4 in PARALLEL)...")

        decisions = {}

        # Calculate tournament stats for prompt injection
        rankings = self.tournament_manager.calculate_rankings()
        tournament_stats = self.stats_injector.calculate_stats(rankings)

        # Prepare base enhanced market data with intelligence
        base_enhanced_data = {
            **market_data,
            "orderbook_analysis": market_intelligence.get("orderbook"),
            "market_feeds": market_intelligence.get("market_feeds"),
            "search_results": market_intelligence.get("search_results")
        }

        # Create per-engine enhanced data with personalized stats
        engine_market_data = {}
        for name in ["A", "B", "C", "D"]:
            # Get personalized stats for this engine
            engine_stats = self.stats_injector.get_stats_for_engine(name, tournament_stats)
            compact_stats = self.stats_injector.format_compact(name, tournament_stats)
            emotion_prompt = self.stats_injector.format_emotion_prompt(name, tournament_stats)

            engine_market_data[name] = {
                **base_enhanced_data,
                "tournament_stats": engine_stats,
                "tournament_stats_compact": compact_stats,
                "tournament_emotion_prompt": emotion_prompt
            }

            logger.debug(f"[Engine {name}] Stats: {compact_stats}")

        # Helper function to get decision from a single gladiator
        def get_gladiator_decision(name: str, gladiator):
            try:
                # Use engine-specific market data with personalized tournament stats
                decision = gladiator.make_trade_decision(
                    asset=asset,
                    asset_type="crypto",  # TODO: Make this dynamic
                    regime=regime,
                    regime_confidence=regime_confidence,
                    market_data=engine_market_data[name]
                )

                if decision:
                    logger.info(
                        f"[Gladiator {name}] Decision: {decision['direction']} @ {decision['entry_price']} "
                        f"(confidence: {decision['confidence']:.1%}, size: {decision['position_size']:.2%})"
                    )
                else:
                    logger.info(f"[Gladiator {name}] Decision: HOLD")

                return name, decision

            except Exception as e:
                logger.error(f"[Gladiator {name}] Decision failed: {e}")
                return name, None

        # Execute all 4 gladiators in PARALLEL using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all 4 gladiator decisions
            futures = {
                executor.submit(get_gladiator_decision, name, gladiator): name
                for name, gladiator in self.gladiators.items()
            }

            # Collect results as they complete
            for future in as_completed(futures):
                name, decision = future.result()
                decisions[name] = decision

        return decisions

    def _execute_gladiator_trades(self, decisions: Dict[str, Optional[Dict]]) -> int:
        """
        Execute trades for gladiators who decided to trade.

        Returns:
            Number of trades successfully opened
        """
        trades_opened = 0

        for name, decision in decisions.items():
            if decision is None:
                continue

            try:
                gladiator = self.gladiators[name]
                trade_id = gladiator.open_trade(decision)

                if trade_id:
                    trades_opened += 1
                    logger.success(
                        f"[Gladiator {name}] Opened trade {trade_id}: "
                        f"{decision['direction']} {decision['asset']}"
                    )

            except Exception as e:
                logger.error(f"[Gladiator {name}] Failed to open trade: {e}")

        return trades_opened

    def _update_all_trades(self, market_data: Dict):
        """
        Update all open trades for all gladiators (check SL/TP).

        Args:
            market_data: Current market prices
        """
        # Extract current prices from market_data
        current_prices = {}

        # Get all unique assets from all portfolios
        for name, gladiator in self.gladiators.items():
            portfolio = gladiator.portfolio
            open_trades = portfolio.get_open_trades()

            for trade in open_trades:
                asset = trade["asset"]
                # Use current price from market_data if available
                if "close" in market_data:
                    current_prices[asset] = market_data["close"]

        # Update trades for all gladiators
        for name, gladiator in self.gladiators.items():
            try:
                gladiator.update_trades(current_prices)
            except Exception as e:
                logger.error(f"[Gladiator {name}] Trade update failed: {e}")

    # ==================== TOURNAMENT MANAGEMENT ====================

    def _check_tournament_events(self):
        """
        Check and execute tournament events:
        1. Knowledge transfer (every cycle) - winner teaches losers
        2. Kill cycle (every 24 hours) - eliminate weakest engine
        3. Weight adjustment (every 24 hours) - adjust influence weights
        4. Breeding (every 4 days) - combine winner DNA into offspring
        """
        now = datetime.now(timezone.utc)

        # Get current rankings
        rankings = self.tournament_manager.calculate_rankings()

        # Execute knowledge transfer (every cycle - winner teaches losers)
        teaching_session = self.knowledge_transfer.execute_knowledge_transfer(
            rankings=rankings,
            get_portfolio_fn=self.tournament_manager.get_portfolio,
            get_engine_fn=lambda name: self.gladiators[name]
        )

        if teaching_session:
            strategies = [r.strategy_chosen for r in teaching_session.responses]
            logger.info(
                f"📚 Knowledge transfer: Engine {teaching_session.teacher_engine} taught "
                f"{len(teaching_session.learners)} learners. Strategies: {strategies}"
            )

        # Check for kill cycle (every 24 hours)
        kill_event = self.kill_cycle.execute_kill(
            rankings=rankings,
            get_portfolio_fn=self.tournament_manager.get_portfolio,
            get_engine_fn=lambda name: self.gladiators[name]
        )

        if kill_event:
            logger.warning(
                f"💀 Kill cycle executed: Engine {kill_event.killed_engine} eliminated, "
                f"learned from Engine {kill_event.winner_engine}"
            )

        # Check for weight adjustment (every 24 hours) - using WeightAdjuster
        weight_result = self.weight_adjuster.adjust_weights(rankings)
        if weight_result and weight_result.adjustment_made:
            # Sync weights to tournament manager
            for engine, ew in weight_result.engine_weights.items():
                self.tournament_manager.update_weight(engine, ew.current_weight)
            logger.info(
                f"🎯 Weight adjustment complete: Strategy={weight_result.strategy.value}"
            )
            self.last_weight_adjustment = now

        # Check for breeding cycle (every 4 days)
        breeding_event = self.breeding_cycle.execute_breeding(
            rankings=rankings,
            get_portfolio_fn=self.tournament_manager.get_portfolio,
            get_engine_fn=lambda name: self.gladiators[name]
        )

        if breeding_event:
            logger.info(
                f"🧬 Breeding cycle executed: Offspring {breeding_event.offspring_id} "
                f"from Engine {breeding_event.parent1_engine} x {breeding_event.parent2_engine} "
                f"→ Engine {breeding_event.offspring_assigned_to}"
            )
            self.last_breeding = now

    def _adjust_weights(self):
        """
        Adjust gladiator weights based on performance.

        DEPRECATED: Now uses WeightAdjuster system with:
        - Multiple strategies (rank-based, Sharpe-based, hybrid)
        - Smooth transitions
        - Weight bounds (5% min, 50% max)
        - Momentum bonuses

        This method is kept for backwards compatibility.
        """
        rankings = self.tournament_manager.calculate_rankings()
        result = self.weight_adjuster.adjust_weights(rankings, force=True)

        if result and result.adjustment_made:
            for engine, ew in result.engine_weights.items():
                self.tournament_manager.update_weight(engine, ew.current_weight)
            logger.success(f"Weight adjustment complete using {result.strategy.value} strategy")

    def _execute_breeding(self):
        """
        Execute breeding mechanism (every 4 days).

        Flow:
        1. Identify top 2 gladiators
        2. Extract their successful patterns
        3. Create "offspring" strategy combining both
        4. Apply offspring insights to bottom 2 gladiators
        5. Winner Teaches Losers system activated
        """
        rankings = self.tournament_manager.get_tournament_summary()["rankings"]

        # Get top 2 and bottom 2
        winner_1 = rankings[0]
        winner_2 = rankings[1]
        loser_1 = rankings[2]
        loser_2 = rankings[3]

        logger.info("🧬 BREEDING MECHANISM ACTIVATED")
        logger.info(f"  Winners: Gladiator {winner_1['gladiator']} (#{winner_1['rank']}) + Gladiator {winner_2['gladiator']} (#{winner_2['rank']})")
        logger.info(f"  Learners: Gladiator {loser_1['gladiator']} (#{loser_1['rank']}) + Gladiator {loser_2['gladiator']} (#{loser_2['rank']})")

        # Extract successful patterns from winners
        winner_1_patterns = self._extract_successful_patterns(winner_1["gladiator"])
        winner_2_patterns = self._extract_successful_patterns(winner_2["gladiator"])

        # Combine patterns into offspring strategy
        offspring_strategy = self._combine_patterns(winner_1_patterns, winner_2_patterns)

        logger.info(f"  Offspring strategy created: {len(offspring_strategy)} key insights")

        # Winner Teaches Losers: Apply insights to bottom 2
        self._apply_winner_insights(loser_1["gladiator"], offspring_strategy)
        self._apply_winner_insights(loser_2["gladiator"], offspring_strategy)

        logger.success("🧬 Breeding complete - Winners' insights applied to learners")

    def _extract_successful_patterns(self, gladiator_name: str) -> List[str]:
        """
        Extract successful patterns from a gladiator's winning trades.

        Returns:
            List of pattern descriptions
        """
        gladiator = self.gladiators[gladiator_name]
        portfolio = gladiator.portfolio

        # Get closed trades with wins
        closed_trades = portfolio.get_closed_trades()
        winning_trades = [t for t in closed_trades if t["outcome"] == "win"]

        patterns = []

        # Extract patterns from winning trades
        for trade in winning_trades[-10:]:  # Last 10 wins
            patterns.append(f"Won with {trade['direction']} in {trade.get('regime', 'UNKNOWN')} regime")

        return patterns

    def _combine_patterns(self, patterns_1: List[str], patterns_2: List[str]) -> List[str]:
        """
        Combine patterns from two winners into offspring strategy.

        Returns:
            Combined insights
        """
        # Simple combination: Take best patterns from both
        combined = []

        # Add unique patterns from both
        all_patterns = set(patterns_1 + patterns_2)
        combined.extend(list(all_patterns))

        # Add meta-insight
        combined.append("Winners adapted to regime changes quickly")
        combined.append("Winners used disciplined position sizing")

        return combined[:10]  # Top 10 insights

    def _apply_winner_insights(self, gladiator_name: str, insights: List[str]):
        """
        Apply winner insights to a learning gladiator.

        NOTE: In a full implementation, this would update the gladiator's
        prompt or decision-making logic. For now, we log the insights.
        """
        logger.info(f"  [Gladiator {gladiator_name}] Learning from winners:")
        for insight in insights[:5]:  # Top 5 insights
            logger.info(f"    - {insight}")

        # TODO: In future, dynamically update gladiator prompts with these insights

    def _log_tournament_standings(self):
        """Log current tournament standings."""
        summary = self.tournament_manager.get_tournament_summary()

        logger.info("\n📊 TOURNAMENT STANDINGS:")
        for ranking in summary["rankings"]:
            logger.info(
                f"  #{ranking['rank']} - Gladiator {ranking['gladiator']} | "
                f"Weight: {ranking['weight']:.0%} | "
                f"P&L: ${ranking['total_pnl_usd']:+.2f} | "
                f"WR: {ranking['win_rate']:.1%} | "
                f"Trades: {ranking['total_trades']}"
            )

    # ==================== UTILITY METHODS ====================

    def _save_state(self):
        """
        Save current tournament state to disk for dashboard consumption.

        Saves complete snapshot of:
        - Gladiator stats (P&L, trades, win rate)
        - Tournament rankings and weights
        - Recent cycle history
        - Timestamp information
        """
        try:
            # Get stats for all 4 gladiators
            gladiator_stats = {}
            for name in ["A", "B", "C", "D"]:
                portfolio = self.tournament_manager.get_portfolio(name)
                stats = portfolio.get_stats()

                # Convert to serializable format
                gladiator_stats[name] = {
                    "total_trades": stats.total_trades,
                    "wins": stats.wins,
                    "losses": stats.losses,
                    "win_rate": stats.win_rate,
                    "total_pnl_usd": stats.total_pnl_usd,
                    "total_pnl_percent": stats.total_pnl_percent,
                    "sharpe_ratio": getattr(stats, 'sharpe_ratio', 0.0),
                    "max_drawdown": getattr(stats, 'max_drawdown', 0.0),
                    "open_trades": len(portfolio.get_open_trades()),
                    "closed_trades": len([t for t in portfolio.trades if t.status == "CLOSED"])
                }

            # Get tournament rankings
            tournament_summary = self.tournament_manager.get_tournament_summary()
            rankings = tournament_summary.get("rankings", [])

            # Create state snapshot
            state = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tournament_start": self.tournament_start_time.isoformat(),
                "cycle_count": self.cycle_count,
                "gladiators": gladiator_stats,
                "rankings": [
                    {
                        "rank": r["rank"],
                        "gladiator": r["gladiator"],
                        "weight": r["weight"],
                        "total_pnl_usd": r["total_pnl_usd"],
                        "win_rate": r["win_rate"],
                        "total_trades": r["total_trades"]
                    }
                    for r in rankings
                ],
                "last_weight_adjustment": self.last_weight_adjustment.isoformat(),
                "last_breeding": self.last_breeding.isoformat(),
                "recent_cycles": [
                    {
                        "cycle_number": c.cycle_number,
                        "timestamp": c.timestamp.isoformat(),
                        "asset": c.asset,
                        "regime": c.regime,
                        "decisions_made": c.decisions_made,
                        "trades_opened": c.trades_opened
                    }
                    for c in self.cycles_history[-10:]  # Last 10 cycles
                ]
            }

            # Write to file (atomic write with temp file)
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(state, f, indent=2)

            # Atomic rename (replaces existing file)
            temp_file.replace(self.state_file)

        except Exception as e:
            logger.warning(f"Failed to save Mother AI state: {e}")

    # ==================== HYDRA 4.0: BATCH GENERATION ====================

    def run_generation_cycle(
        self,
        strategies_per_engine: int = 1000,
        market_context: Optional[Dict] = None,
        use_mock: bool = False,
        parallel: bool = True,
        use_backtest: bool = True
    ) -> Dict:
        """
        HYDRA 4.0: Run a full 4-engine strategy generation cycle.

        This is the core of HYDRA 4.0 turbo mode:
        - Each engine generates 1000 strategies in parallel
        - All 4000 strategies are ranked in tournament
        - Votes counted by engine (top 100 strategies)
        - Winner teaches losers mechanism applied

        Args:
            strategies_per_engine: Strategies each engine generates (default 1000)
            market_context: Current market data
            use_mock: Use mock generation (no API costs)
            parallel: Run engines in parallel (faster)

        Returns:
            Dict with cycle results including winning engine and strategies
        """
        import time
        start_time = time.time()

        logger.info(f"[MotherAI] Starting HYDRA 4.0 generation cycle ({strategies_per_engine} per engine)")

        all_strategies = []
        engine_results = {}

        if parallel:
            # Run all 4 engines in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(
                        gladiator.generate_batch,
                        strategies_per_engine,
                        market_context,
                        use_mock
                    ): name
                    for name, gladiator in self.gladiators.items()
                }

                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        strategies = future.result()
                        all_strategies.extend(strategies)
                        engine_results[name] = {
                            "count": len(strategies),
                            "success": True
                        }
                        logger.info(f"[MotherAI] Engine {name} generated {len(strategies)} strategies")
                    except Exception as e:
                        logger.error(f"[MotherAI] Engine {name} failed: {e}")
                        engine_results[name] = {
                            "count": 0,
                            "success": False,
                            "error": str(e)
                        }
        else:
            # Sequential generation (for debugging)
            for name, gladiator in self.gladiators.items():
                try:
                    strategies = gladiator.generate_batch(
                        strategies_per_engine,
                        market_context,
                        use_mock
                    )
                    all_strategies.extend(strategies)
                    engine_results[name] = {"count": len(strategies), "success": True}
                except Exception as e:
                    logger.error(f"[MotherAI] Engine {name} failed: {e}")
                    engine_results[name] = {"count": 0, "success": False, "error": str(e)}

        # Rank all strategies using tournament
        logger.info(f"[MotherAI] Ranking {len(all_strategies)} strategies (backtest={use_backtest})...")
        ranked = self._rank_strategies(all_strategies, use_backtest=use_backtest)

        # Count votes by engine (top 100)
        vote_breakdown = self._count_votes(ranked[:100])

        # Determine winning engine
        winning_engine = max(vote_breakdown, key=vote_breakdown.get) if vote_breakdown else "A"

        # Get winning strategy
        winning_strategy = ranked[0] if ranked else None

        # Apply winner teaches loser
        if winning_strategy:
            self._apply_winner_teaches_loser(winning_engine, winning_strategy)

        cycle_time_ms = int((time.time() - start_time) * 1000)

        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_strategies": len(all_strategies),
            "engine_results": engine_results,
            "winning_engine": winning_engine,
            "winning_strategy_id": winning_strategy.get("strategy_id") if winning_strategy else None,
            "vote_breakdown": vote_breakdown,
            "cycle_time_ms": cycle_time_ms,
            "top_10": ranked[:10] if len(ranked) >= 10 else ranked
        }

        logger.info(f"[MotherAI] Generation cycle complete: {len(all_strategies)} strategies, "
                   f"winner: Engine {winning_engine}, time: {cycle_time_ms}ms")

        return result

    def _rank_strategies(self, strategies: List[Dict], use_backtest: bool = False) -> List[Dict]:
        """
        Rank strategies using tournament scoring.

        Args:
            strategies: List of strategy dicts
            use_backtest: If True, run full backtesting (slower but accurate)

        Returns:
            List of strategies sorted by rank score
        """
        if not strategies:
            return []

        if use_backtest:
            # Use TurboTournament for full backtesting
            try:
                from libs.hydra.turbo_tournament import get_turbo_tournament, BacktestResult
                from libs.hydra.turbo_generator import GeneratedStrategy, StrategyType

                tournament = get_turbo_tournament()

                # Convert dicts to GeneratedStrategy objects
                gen_strategies = []
                for s in strategies:
                    try:
                        specialty_str = s.get("specialty", "liquidation_cascade")
                        specialty = StrategyType(specialty_str) if specialty_str in [t.value for t in StrategyType] else StrategyType.LIQUIDATION_CASCADE

                        gen_strat = GeneratedStrategy(
                            strategy_id=s.get("strategy_id", f"GEN_{len(gen_strategies):06d}"),
                            name=s.get("reasoning", "Generated Strategy")[:50],
                            specialty=specialty,
                            regime=s.get("regime", "RANGING"),
                            asset_class=s.get("asset_class", "crypto"),
                            entry_rules={"rule": s.get("entry_rules", "")},
                            exit_rules={"rule": s.get("exit_rules", "")},
                            risk_per_trade=s.get("position_size_pct", 1.0) / 100,
                            stop_loss_atr_mult=s.get("stop_loss_pct", 2.0),
                            take_profit_atr_mult=s.get("take_profit_pct", 4.0),
                            min_confidence=s.get("confidence", 0.5),
                        )
                        gen_strategies.append(gen_strat)
                    except Exception as e:
                        logger.debug(f"[MotherAI] Error converting strategy: {e}")
                        continue

                if gen_strategies:
                    # Run tournament ranking
                    ranked_results = tournament.rank_batch(gen_strategies, max_workers=4)

                    # Convert back to dicts with scores
                    ranked = []
                    for strat, result in ranked_results:
                        strat_dict = strategies[gen_strategies.index(strat)] if strat in gen_strategies else {}
                        strat_dict["rank_score"] = result.rank_score
                        strat_dict["backtest_wr"] = result.win_rate
                        strat_dict["backtest_sharpe"] = result.sharpe_ratio
                        ranked.append(strat_dict)

                    return ranked

            except Exception as e:
                logger.warning(f"[MotherAI] Backtest ranking failed, using confidence: {e}")

        # Fallback: rank by confidence score
        # Add computed rank score based on multiple factors
        for s in strategies:
            confidence = s.get("confidence", 0.5)
            sl = s.get("stop_loss_pct", 2.0)
            tp = s.get("take_profit_pct", 4.0)
            rr_ratio = tp / sl if sl > 0 else 1.0

            # Composite score: confidence + risk/reward bonus
            s["rank_score"] = confidence * 0.6 + min(rr_ratio / 3, 0.4)

        ranked = sorted(strategies, key=lambda s: s.get("rank_score", 0), reverse=True)
        return ranked

    def _count_votes(self, top_strategies: List[Dict]) -> Dict[str, int]:
        """Count votes by engine in top strategies."""
        votes = {"A": 0, "B": 0, "C": 0, "D": 0}

        for strategy in top_strategies:
            engine = strategy.get("engine", "")
            if engine in votes:
                votes[engine] += 1
            elif strategy.get("strategy_id", "").startswith("A_"):
                votes["A"] += 1
            elif strategy.get("strategy_id", "").startswith("B_"):
                votes["B"] += 1
            elif strategy.get("strategy_id", "").startswith("C_"):
                votes["C"] += 1
            elif strategy.get("strategy_id", "").startswith("D_"):
                votes["D"] += 1

        return votes

    def _apply_winner_teaches_loser(self, winning_engine: str, winning_strategy: Dict):
        """Apply winner teaches loser mechanism."""
        logger.info(f"[MotherAI] Winner teaches loser: Engine {winning_engine} shares knowledge")

        for name, gladiator in self.gladiators.items():
            if name != winning_engine:
                try:
                    gladiator.learn_from_winner(winning_strategy)
                except Exception as e:
                    logger.warning(f"[MotherAI] Engine {name} failed to learn: {e}")

    def get_tournament_summary(self) -> Dict:
        """Get full tournament summary."""
        return {
            "tournament_start": self.tournament_start_time.isoformat(),
            "cycles_completed": self.cycle_count,
            "gladiators": list(self.gladiators.keys()),
            "rankings": self.tournament_manager.get_tournament_summary()["rankings"],
            "last_weight_adjustment": self.last_weight_adjustment.isoformat(),
            "last_breeding": self.last_breeding.isoformat()
        }

    def reset_tournament(self):
        """Reset tournament (clear all portfolios and history)."""
        logger.warning("Resetting tournament...")

        for name in self.gladiators.keys():
            portfolio = self.tournament_manager.get_portfolio(name)
            # TODO: Add reset method to portfolio

        self.tournament_start_time = datetime.now(timezone.utc)
        self.last_weight_adjustment = datetime.now(timezone.utc)
        self.last_breeding = datetime.now(timezone.utc)
        self.cycle_count = 0
        self.cycles_history = []

        logger.success("Tournament reset complete")

    # ==================== MOD 7: FALLBACK HANDLING ====================

    def _freeze_all_engines(self, reason: str):
        """
        CRITICAL: Freeze all engines when Mother AI fails.

        MOD 7 Rule: If Mother AI fails, freeze ALL engines, don't substitute.
        Engines should NEVER trade without Mother AI supervision.
        """
        self.ai_state.is_frozen = True
        self.ai_state.failure_reason = reason
        self.ai_state.frozen_at = datetime.now(timezone.utc)

        logger.critical("╔══════════════════════════════════════════════════════════════╗")
        logger.critical("║  MOTHER AI FROZEN - ALL ENGINES DISABLED                    ║")
        logger.critical(f"║  Reason: {reason[:50]}...")
        logger.critical("║  NO TRADING WILL OCCUR UNTIL MANUALLY UNFROZEN              ║")
        logger.critical("╚══════════════════════════════════════════════════════════════╝")

        # Persist frozen state
        self._save_frozen_state()

    def _save_frozen_state(self):
        """Save frozen state to disk for recovery."""
        try:
            frozen_file = self.state_file.parent / "mother_ai_frozen.json"
            state = {
                "is_frozen": self.ai_state.is_frozen,
                "failure_reason": self.ai_state.failure_reason,
                "frozen_at": self.ai_state.frozen_at.isoformat() if self.ai_state.frozen_at else None,
                "consecutive_failures": self.ai_state.consecutive_failures,
            }
            with open(frozen_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save frozen state: {e}")

    def unfreeze(self, operator_name: str = "manual"):
        """
        Manually unfreeze Mother AI after fixing the issue.

        Args:
            operator_name: Who is unfreezing (for logging)
        """
        if not self.ai_state.is_frozen:
            logger.info("Mother AI is not frozen, nothing to unfreeze")
            return

        logger.warning(f"MOTHER AI UNFROZEN by {operator_name}")
        logger.warning("All engines will resume trading on next cycle")

        self.ai_state.is_frozen = False
        self.ai_state.failure_reason = None
        self.ai_state.frozen_at = None
        self.ai_state.consecutive_failures = 0
        self.ai_state.is_healthy = True

        # Remove frozen state file
        try:
            frozen_file = self.state_file.parent / "mother_ai_frozen.json"
            if frozen_file.exists():
                frozen_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to remove frozen state file: {e}")

    def is_operational(self) -> bool:
        """Check if Mother AI is operational (not frozen)."""
        return not self.ai_state.is_frozen and self.ai_state.is_healthy

    def get_health_status(self) -> Dict:
        """Get Mother AI health status for monitoring."""
        return {
            "is_healthy": self.ai_state.is_healthy,
            "is_frozen": self.ai_state.is_frozen,
            "failure_reason": self.ai_state.failure_reason,
            "frozen_at": self.ai_state.frozen_at.isoformat() if self.ai_state.frozen_at else None,
            "consecutive_failures": self.ai_state.consecutive_failures,
            "max_failures_before_freeze": self.ai_state.max_failures_before_freeze,
            "cycles_completed": self.cycle_count,
        }


# ==================== SINGLETON PATTERN ====================

_mother_ai = None

def get_mother_ai() -> MotherAI:
    """Get singleton instance of Mother AI."""
    global _mother_ai
    if _mother_ai is None:
        _mother_ai = MotherAI()
    return _mother_ai
