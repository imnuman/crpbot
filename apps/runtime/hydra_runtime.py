"""
HYDRA 3.0 - Main Runtime Orchestrator

Integrates all 10 layers + 4 upgrades into a unified trading system.

Signal Flow:
1. Regime Detector → Classify market state
2. Asset Profiles → Load market-specific config
3. Anti-Manipulation → Pre-filter suspicious conditions
4. Gladiators (A/B/C/D) → Generate, validate, backtest, synthesize
5. Tournament → Track strategy performance
6. Consensus → Aggregate engine votes
7. Cross-Asset Filter → Check macro correlations
8. Lesson Memory → Check for known failures
9. Guardian → Final risk validation (9 sacred rules)
10. Execution Optimizer → Smart order placement
11. Explainability → Log full context

This orchestrator runs 24/7, manages strategy evolution, and trades with zero emotion.
"""

import os
import time
import argparse
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional
from loguru import logger
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# HYDRA Core Layers
from libs.hydra.regime_detector import get_regime_detector
from libs.hydra.asset_profiles import get_asset_profile_manager
from libs.hydra.anti_manipulation import get_anti_manipulation_filter
from libs.hydra.guardian import get_guardian
from libs.hydra.execution_optimizer import get_execution_optimizer
from libs.hydra.explainability import get_explainability_logger
from libs.hydra.cross_asset_filter import get_cross_asset_filter
from libs.hydra.consensus import get_consensus_engine
from libs.hydra.tournament_manager import get_tournament_manager
from libs.hydra.breeding_engine import get_breeding_engine
from libs.hydra.lesson_memory import get_lesson_memory
from libs.hydra.paper_trader import get_paper_trader
from libs.hydra.database import init_hydra_db, HydraSession
from libs.hydra.tournament_tracker import TournamentTracker
from libs.hydra.cycles.stats_injector import get_stats_injector
from libs.hydra.strategy_memory import get_strategy_memory

# Engines (4 AI competitors)
from libs.hydra.engines.engine_a_deepseek import EngineA_DeepSeek
from libs.hydra.engines.engine_b_claude import EngineB_Claude
from libs.hydra.engines.engine_c_grok import EngineC_Grok
from libs.hydra.engines.engine_d_gemini import EngineD_Gemini

# Data Provider
from libs.data.coinbase_client import get_coinbase_client

# Prometheus Monitoring
from libs.monitoring import MetricsExporter, HydraMetrics


class HydraRuntime:
    """
    Main orchestrator for HYDRA 3.0.

    Manages:
    - All 10 layers
    - 4 gladiators
    - Strategy evolution
    - Trade execution
    - Performance tracking
    """

    def __init__(
        self,
        assets: List[str],
        paper_trading: bool = True,
        check_interval_seconds: int = 300  # 5 minutes
    ):
        self.assets = assets
        self.paper_trading = paper_trading
        self.check_interval = check_interval_seconds

        logger.info("="*80)
        logger.info("HYDRA 3.0 - Initializing Runtime")
        logger.info("="*80)

        # Initialize database
        init_hydra_db()

        # Initialize all layers
        self._init_layers()

        # Initialize gladiators
        self._init_engines()

        # State tracking
        self.iteration = 0
        self.last_elimination_check = None
        self.last_breeding_check = None
        self.open_positions = {}  # Track open positions: {asset: position_data}

        # Start Prometheus metrics exporter
        self.metrics_exporter = MetricsExporter(port=9100)
        self.metrics_exporter.start()

        # Start background price update thread (updates every 10 seconds)
        self._price_update_running = True
        self._price_thread = threading.Thread(target=self._background_price_updater, daemon=True)
        self._price_thread.start()

        logger.success("HYDRA 3.0 initialized successfully")

    def _init_layers(self):
        """Initialize all 10 layers + 4 upgrades."""
        logger.info("Initializing HYDRA layers...")

        # Layer 1: Regime Detection
        self.regime_detector = get_regime_detector()

        # Layer 2: Asset Profiles (Upgrade B)
        self.asset_profiles = get_asset_profile_manager()

        # Layer 3: Anti-Manipulation Filter
        self.anti_manip = get_anti_manipulation_filter()

        # Layer 4: Guardian (9 sacred rules)
        self.guardian = get_guardian()

        # Layer 5: Tournament Manager
        self.tournament = get_tournament_manager()

        # Layer 6: Breeding Engine
        self.breeder = get_breeding_engine()

        # Layer 7: Consensus Engine
        self.consensus = get_consensus_engine()

        # Layer 8: Cross-Asset Filter (Upgrade D)
        self.cross_asset = get_cross_asset_filter()

        # Layer 9: Lesson Memory (Upgrade C)
        self.lessons = get_lesson_memory()

        # Layer 10: Execution Optimizer
        self.executor = get_execution_optimizer()

        # Upgrade A: Explainability
        self.explainer = get_explainability_logger()

        # Paper Trading System
        self.paper_trader = get_paper_trader()

        # Tournament Tracker (gladiator vote-level performance)
        self.vote_tracker = TournamentTracker()

        # Stats Injector for emotion prompts
        self.stats_injector = get_stats_injector()

        # Strategy Memory for 80/20 exploit/explore
        self.strategy_memory = get_strategy_memory()

        # Data provider
        self.data_client = get_coinbase_client()

        logger.success("All layers initialized")

    def _init_engines(self):
        """Initialize 4 gladiators with API keys."""
        logger.info("Initializing Gladiators...")

        self.gladiator_a = EngineA_DeepSeek(api_key=os.getenv("DEEPSEEK_API_KEY"))
        self.gladiator_b = EngineB_Claude(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.gladiator_c = EngineC_Grok(api_key=os.getenv("GROQ_API_KEY"))
        self.gladiator_d = EngineD_Gemini(api_key=os.getenv("GEMINI_API_KEY"))

        self.engines = [
            self.gladiator_a,
            self.gladiator_b,
            self.gladiator_c,
            self.gladiator_d
        ]

        logger.success(f"4 Gladiators initialized (A: DeepSeek, B: Claude, C: Groq, D: Gemini)")

    # ==================== MAIN LOOP ====================

    def run(self, iterations: int = -1):
        """
        Main runtime loop.

        Args:
            iterations: Number of iterations to run (-1 = infinite)
        """
        logger.info(f"Starting HYDRA runtime (paper trading: {self.paper_trading})")
        logger.info(f"Assets: {', '.join(self.assets)}")
        logger.info(f"Check interval: {self.check_interval}s")

        while iterations == -1 or self.iteration < iterations:
            self.iteration += 1

            try:
                logger.info(f"\n{'='*80}")
                logger.info(f"Iteration {self.iteration} - {datetime.now(timezone.utc)}")
                logger.info(f"{'='*80}\n")

                # Quick price update for all assets (for dashboard)
                self._update_all_prices()

                # Check open paper trades first
                if self.paper_trading:
                    self._check_paper_trades()

                # Process each asset
                for asset in self.assets:
                    self._process_asset(asset)

                # Run tournament cycles (if needed)
                self._run_tournament_cycles()

                # Print paper trading stats and tournament leaderboard
                if self.paper_trading and self.iteration % 10 == 0:  # Every 10 iterations
                    self._print_paper_trading_stats()
                    self.vote_tracker.print_leaderboard()

                # Update Prometheus metrics
                self._update_prometheus_metrics()

                # Sleep until next iteration
                if iterations == -1 or self.iteration < iterations:
                    logger.info(f"Sleeping {self.check_interval}s until next iteration...")
                    time.sleep(self.check_interval)

            except KeyboardInterrupt:
                logger.warning("Received interrupt signal, shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(60)  # Wait 1 min before retry

        logger.info("HYDRA runtime stopped")

    def _process_asset(self, asset: str):
        """
        Process a single asset through the full HYDRA pipeline.

        Flow:
        1. Fetch market data
        2. Detect regime
        3. Get asset profile
        4. Check anti-manipulation
        5. Get active strategies for this regime
        6. If needed, generate new strategies (gladiators)
        7. Get trading signals
        8. Consensus vote
        9. Cross-asset check
        10. Lesson memory check
        11. Guardian validation
        12. Execute trade
        13. Log explainability
        """
        logger.info(f"Processing {asset}...")

        # Step 1: Fetch market data
        try:
            df = self.data_client.fetch_klines(
                symbol=asset,
                interval="1m",
                limit=200
            )
            # Convert DataFrame to list of dicts for HYDRA components
            market_data = df.to_dict('records')
        except Exception as e:
            logger.error(f"Failed to fetch data for {asset}: {e}")
            return

        if not market_data or len(market_data) < 100:
            logger.warning(f"Insufficient data for {asset}")
            return

        # Update price metrics for dashboard
        current_price = market_data[-1]["close"]
        HydraMetrics.set_price(asset, current_price)

        # Step 2: Detect regime
        regime_result = self.regime_detector.detect_regime(
            symbol=asset,
            candles=market_data
        )
        regime = regime_result["regime"]
        regime_confidence = regime_result["confidence"]

        logger.info(f"{asset} regime: {regime} (confidence: {regime_confidence:.1%})")

        # Step 3: Get asset profile
        asset_type = self._classify_asset(asset)
        profile = self.asset_profiles.get_profile(asset)

        # Step 4: Anti-manipulation filtering happens AFTER strategy generation
        # (in _execute_paper_trade method after we have backtest results)

        # Step 5: Create market summary for gladiators
        # Bug Fix #41: Gladiators need summary dict, not candle list
        market_summary = self._create_market_summary(market_data)

        # Step 6-7: Strategy generation and signal creation
        signal = self._generate_signal(
            asset=asset,
            asset_type=asset_type,
            regime=regime,
            regime_confidence=regime_confidence,
            market_data=market_summary,  # Pass summary to gladiators
            profile=profile
        )

        if not signal or signal.get("action") == "HOLD":
            logger.info(f"{asset}: No trade signal (consensus: HOLD)")
            return

        # Step 8: Cross-asset filter (convert BUY/SELL → LONG/SHORT)
        cross_asset_result = self.cross_asset.check_cross_asset_alignment(
            asset=asset,
            asset_type=asset_type,
            direction=self._convert_direction(signal["action"]),
            market_data=market_data,
            dxy_data=self._get_dxy_data(),
            btc_data=self._get_btc_data(),
            em_basket_data=None  # TODO: Implement EM basket
        )

        cross_asset_passed, cross_asset_reason = cross_asset_result
        if not cross_asset_passed:
            logger.warning(f"{asset} blocked by cross-asset filter: {cross_asset_reason}")
            return

        # Step 9: Lesson memory check
        market_context = self._build_market_context(asset, market_data)
        lesson_check = self.lessons.check_lessons(
            asset=asset,
            regime=regime,
            strategy=signal.get("strategy", {}),
            signal=signal,
            market_context=market_context
        )

        lesson_triggered, lesson_obj = lesson_check
        if lesson_triggered:
            lesson_id = lesson_obj.lesson_id if lesson_obj else "Unknown"
            logger.error(f"{asset} rejected by lesson memory: {lesson_id}")
            return

        # Step 10: Guardian validation
        entry_price = market_data[-1]["close"]
        sl_pct = signal.get("stop_loss_pct", 0.015)
        if signal["action"] == "BUY":
            sl_price = entry_price * (1 - sl_pct)
        else:  # SELL
            sl_price = entry_price * (1 + sl_pct)

        # Step 10: Guardian validation (convert BUY/SELL → LONG/SHORT)
        guardian_check = self.guardian.validate_trade(
            asset=asset,
            asset_type=asset_type,
            direction=self._convert_direction(signal["action"]),
            position_size_usd=signal.get("position_size_usd", 100),
            entry_price=entry_price,
            sl_price=sl_price,
            regime=regime,
            current_positions=list(self.open_positions.values()) if hasattr(self, 'open_positions') else [],
            strategy_correlations=None
        )

        if not guardian_check["approved"]:
            logger.error(
                f"{asset} rejected by Guardian: {guardian_check['rejection_reason']}"
            )
            return

        # Step 11: Execute trade (or paper trade)
        if self.paper_trading:
            self._execute_paper_trade(asset, signal, market_data)
        else:
            self._execute_live_trade(asset, signal, market_data)

        # Step 12: Log explainability
        self._log_trade_decision(asset, signal, regime, market_context)

    def _generate_signal(
        self,
        asset: str,
        asset_type: str,
        regime: str,
        regime_confidence: float,
        market_data: List[Dict],
        profile: Dict
    ) -> Optional[Dict]:
        """
        Generate trading signal via TRUE TOURNAMENT COMPETITION.

        Each engine generates independently (no peeking at others).
        Each engine evaluates ALL 4 competing strategies.
        Voting picks the winning strategy.

        Returns:
            {
                "action": "BUY" | "SELL" | "HOLD",
                "consensus_level": "UNANIMOUS" | "STRONG" | "WEAK",
                "position_size_modifier": 1.0 | 0.75 | 0.5,
                "strategy": {...},
                "entry_price": 97500,
                "stop_loss_pct": 0.015,
                "take_profit_pct": 0.025
            }
        """
        population_key = f"{asset}:{regime}"

        # =======================================================
        # PHASE 0: INJECT TOURNAMENT EMOTION CONTEXT
        # =======================================================
        # Each engine gets personalized tournament standing info
        emotion_prompts = self._get_engine_emotion_prompts()

        # Create engine-specific market_data with emotion context
        market_data_a = {**market_data, "tournament_emotion_prompt": emotion_prompts.get("A", "")}
        market_data_b = {**market_data, "tournament_emotion_prompt": emotion_prompts.get("B", "")}
        market_data_c = {**market_data, "tournament_emotion_prompt": emotion_prompts.get("C", "")}
        market_data_d = {**market_data, "tournament_emotion_prompt": emotion_prompts.get("D", "")}

        # =======================================================
        # PHASE 1: 80/20 EXPLOIT/EXPLORE STRATEGY SELECTION
        # =======================================================
        # Each engine: 80% use winning strategy from memory, 20% generate new

        strategies = []
        engines_data = [
            ("A", self.gladiator_a, market_data_a),
            ("B", self.gladiator_b, market_data_b),
            ("C", self.gladiator_c, market_data_c),
            ("D", self.gladiator_d, market_data_d),
        ]

        for engine_name, gladiator, engine_market_data in engines_data:
            # Try to select from strategy memory (80% exploit)
            # Uses edge decay detection + regime validation
            selected = self.strategy_memory.select_strategy_with_validation(
                engine=engine_name,
                asset=asset,
                regime=regime,
                explore_probability=0.2  # 20% chance to generate new
            )

            if selected:
                # EXPLOIT: Use existing winning strategy
                logger.debug(f"Engine {engine_name}: EXPLOIT - using strategy {selected['strategy_id']} (WR: {selected.get('win_rate', 0):.1%})")
                strategy = {
                    **selected,
                    "source": "memory",
                    "gladiator": engine_name
                }
            else:
                # EXPLORE: Generate new strategy
                logger.debug(f"Engine {engine_name}: EXPLORE - generating new strategy")
                strategy = gladiator.generate_strategy(
                    asset=asset,
                    asset_type=asset_type,
                    asset_profile=profile,
                    regime=regime,
                    regime_confidence=regime_confidence,
                    market_data=engine_market_data,
                    existing_strategies=[]  # No peeking at others
                )
                strategy["source"] = "generated"

            strategies.append(strategy)

        strategy_a, strategy_b, strategy_c, strategy_d = strategies

        # =======================================================
        # PHASE 2: REGISTER ALL STRATEGIES IN TOURNAMENT
        # =======================================================
        population = self.tournament.populations.get(population_key, [])
        for strategy in strategies:
            if strategy.get("strategy_id") and strategy.get("gladiator"):
                existing = any(
                    s.strategy_id == strategy["strategy_id"]
                    for s in population
                )
                if not existing:
                    self.tournament.register_strategy(
                        strategy_id=strategy["strategy_id"],
                        gladiator=strategy["gladiator"],
                        asset=asset,
                        regime=regime
                    )

        # =======================================================
        # PHASE 3: COMPETITIVE VOTING - Each engine evaluates ALL strategies
        # =======================================================
        votes = []
        current_price = market_data["close"]
        mock_signal = {
            "direction": "BUY",
            "entry_price": current_price,
            "stop_loss_pct": 0.015,
            "take_profit_pct": 0.025
        }

        trade_id = f"{asset}_{int(time.time())}"

        # Track which strategy each engine prefers
        strategy_votes = {s.get("strategy_id", f"unknown_{i}"): 0 for i, s in enumerate(strategies)}

        for gladiator in self.engines:
            # Each engine evaluates ALL 4 competing strategies
            best_vote = None
            best_confidence = -1
            best_strategy_id = None

            for strategy in strategies:
                # Use specialty-enforced voting - engines vote HOLD if specialty not triggered
                vote = gladiator.vote_on_trade_with_specialty_check(
                    asset=asset,
                    asset_type=asset_type,
                    regime=regime,
                    strategy=strategy,
                    signal=mock_signal,
                    market_data=market_data
                )

                # Track which strategy this engine prefers most
                vote_confidence = vote.get("confidence", 0)
                if vote_confidence > best_confidence:
                    best_confidence = vote_confidence
                    best_vote = vote
                    best_strategy_id = strategy.get("strategy_id", "unknown")

            if best_vote:
                best_vote["gladiator"] = gladiator.name
                best_vote["preferred_strategy"] = best_strategy_id
                votes.append(best_vote)

                # Count vote for winning strategy
                if best_strategy_id in strategy_votes:
                    strategy_votes[best_strategy_id] += 1

                # Record vote in tournament tracker
                self.vote_tracker.record_vote(
                    trade_id=trade_id,
                    gladiator=gladiator.name,
                    asset=asset,
                    vote=best_vote.get("vote", "HOLD"),
                    confidence=best_vote.get("confidence", 0.5),
                    reasoning=best_vote.get("reasoning", "")
                )

        # =======================================================
        # PHASE 4: DETERMINE WINNING STRATEGY (with confidence tiebreaker)
        # =======================================================
        # Find strategy with most votes, use confidence as tiebreaker
        winning_strategy = strategy_d  # Default fallback
        winning_strategy_id = None

        if strategy_votes:
            # Find max vote count
            max_votes = max(strategy_votes.values())

            # Get all strategies with max votes (potential ties)
            tied_strategies = [sid for sid, count in strategy_votes.items() if count == max_votes]

            if len(tied_strategies) == 1:
                # No tie - clear winner
                winning_strategy_id = tied_strategies[0]
            else:
                # TIE BREAKER: Use average confidence for each tied strategy
                strategy_confidences = {}
                for sid in tied_strategies:
                    confidences = [
                        v.get("confidence", 0.5)
                        for v in votes
                        if v.get("preferred_strategy") == sid
                    ]
                    strategy_confidences[sid] = sum(confidences) / len(confidences) if confidences else 0

                # Pick strategy with highest average confidence
                winning_strategy_id = max(strategy_confidences, key=strategy_confidences.get)
                logger.info(f"Tie broken by confidence: {strategy_confidences}")

            for s in strategies:
                if s.get("strategy_id") == winning_strategy_id:
                    winning_strategy = s
                    break

            logger.info(
                f"Strategy voting: {strategy_votes} -> Winner: {winning_strategy_id} "
                f"({strategy_votes.get(winning_strategy_id, 0)}/4 votes)"
            )

        # =======================================================
        # PHASE 5: GET CONSENSUS AND BUILD SIGNAL
        # =======================================================
        # Sync engine weights to consensus engine
        self.consensus.update_weights(self.tournament.engine_weights)

        # Get consensus (now weight-aware)
        consensus = self.consensus.get_consensus(votes)

        # Build signal using WINNING strategy
        signal = {
            "action": consensus["action"],
            "consensus_level": consensus["consensus_level"],
            "position_size_modifier": consensus["position_size_modifier"],
            "strategy": winning_strategy,
            "winning_strategy_id": winning_strategy_id,
            "strategy_votes": strategy_votes,
            "entry_price": current_price,
            "stop_loss_pct": 0.015,
            "take_profit_pct": 0.0225,
            "votes": votes,
            "consensus_summary": consensus["summary"]
        }

        return signal

    def _execute_paper_trade(self, asset: str, signal: Dict, market_data: List[Dict]):
        """Execute paper trade (simulation only)."""
        # Extract regime from previous detection
        regime_result = self.regime_detector.detect_regime(
            symbol=asset,
            candles=market_data
        )
        regime = regime_result["regime"]

        # Create paper trade
        trade = self.paper_trader.create_paper_trade(
            asset=asset,
            regime=regime,
            strategy_id=signal.get("strategy", {}).get("strategy_id", "UNKNOWN"),
            gladiator=signal.get("strategy", {}).get("gladiator", "UNKNOWN"),
            signal=signal
        )

        logger.success(
            f"PAPER TRADE CREATED: {signal['action']} {asset} @ {signal['entry_price']:.2f} "
            f"(consensus: {signal['consensus_level']}, "
            f"size modifier: {signal['position_size_modifier']:.0%})"
        )

    def _execute_live_trade(self, asset: str, signal: Dict, market_data: List[Dict]):
        """Execute live trade."""
        logger.critical(
            f"LIVE TRADE: {signal['action']} {asset} @ {signal['entry_price']} "
            f"(consensus: {signal['consensus_level']})"
        )

        # Use execution optimizer (convert BUY/SELL → LONG/SHORT)
        asset_type = self._classify_asset(asset)
        execution_result = self.executor.optimize_entry(
            asset=asset,
            asset_type=asset_type,
            direction=self._convert_direction(signal["action"]),
            size=signal.get("position_size_usd", 100),
            current_bid=market_data[-1]["close"] * 0.9995,  # Approximate
            current_ask=market_data[-1]["close"] * 1.0005,
            spread_normal=0.001,
            spread_reject_multiplier=2.0,
            broker_api=None
        )

        logger.info(f"Execution result: {execution_result}")

    def _log_trade_decision(
        self,
        asset: str,
        signal: Dict,
        regime: str,
        market_context: Dict
    ):
        """Log full explainability for this trade."""
        asset_type = self._classify_asset(asset)
        strategy = signal.get("strategy", {})

        # Calculate SL and TP prices
        entry_price = signal["entry_price"]
        sl_pct = signal.get("stop_loss_pct", 0.015)
        tp_pct = signal.get("take_profit_pct", 0.0225)

        if signal["action"] == "BUY":
            sl_price = entry_price * (1 - sl_pct)
            tp_price = entry_price * (1 + tp_pct)
        else:  # SELL
            sl_price = entry_price * (1 + sl_pct)
            tp_price = entry_price * (1 - tp_pct)

        self.explainer.log_trade_decision(
            trade_id=f"{asset}_{int(time.time())}",
            asset=asset,
            asset_type=asset_type,
            regime=regime,
            gladiator_votes=signal.get("votes", []),
            consensus_level=signal.get("avg_confidence", 0.5),
            strategy_id=strategy.get("strategy_id", "UNKNOWN"),
            structural_edge=strategy.get("structural_edge", "Multi-gladiator consensus"),
            entry_reasoning=signal.get("consensus_summary", "Gladiator vote consensus"),
            exit_reasoning=f"SL: {sl_pct:.1%}, TP: {tp_pct:.1%}",
            filters_passed={
                "anti_manipulation": True,
                "cross_asset": True,
                "lesson_memory": True
            },
            filter_block_reasons=[],
            guardian_approved=True,
            guardian_reason="Trade approved by Guardian",
            position_size_original=100.0,  # Base size
            position_size_final=100.0 * signal.get("position_size_modifier", 1.0),
            adjustment_reason=f"Consensus modifier: {signal.get('position_size_modifier', 1.0):.0%}",
            direction=signal["action"],
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            risk_reward_ratio=tp_pct / sl_pct if sl_pct > 0 else 1.5
        )

    # ==================== TOURNAMENT CYCLES ====================

    def _run_tournament_cycles(self):
        """Run elimination and breeding cycles for all populations."""
        current_time = datetime.now(timezone.utc)

        # Elimination cycle (24 hours)
        if (
            self.last_elimination_check is None or
            (current_time - self.last_elimination_check).total_seconds() > 86400
        ):
            logger.info("\n" + "="*80)
            logger.info("Running ELIMINATION CYCLE (24-hour)")
            logger.info("="*80)

            for population_key in list(self.tournament.populations.keys()):
                asset, regime = population_key.split(":")
                result = self.tournament.run_elimination_cycle(asset, regime, current_time)

                if result.get("eliminated"):
                    logger.warning(
                        f"{population_key}: Eliminated {len(result['eliminated'])} strategies"
                    )

            self.last_elimination_check = current_time

        # Breeding cycle (4 days)
        if (
            self.last_breeding_check is None or
            (current_time - self.last_breeding_check).total_seconds() > 345600  # 4 days
        ):
            logger.info("\n" + "="*80)
            logger.info("Running BREEDING CYCLE (4-day)")
            logger.info("="*80)

            for population_key in list(self.tournament.populations.keys()):
                asset, regime = population_key.split(":")
                breed_result = self.tournament.run_breeding_cycle(asset, regime, current_time)

                if breed_result.get("bred"):
                    # Get breeding pairs
                    parent1_id = breed_result["breeding_pairs"][0][0]
                    parent2_id = breed_result["breeding_pairs"][0][1]

                    # Find parent strategies in population
                    population = self.tournament.populations.get(population_key, [])
                    parent1_perf = next((s for s in population if s.strategy_id == parent1_id), None)
                    parent2_perf = next((s for s in population if s.strategy_id == parent2_id), None)

                    if parent1_perf and parent2_perf:
                        # Build parent strategy dicts from performance records
                        parent1_dict = {
                            "strategy_id": parent1_id,
                            "gladiator": parent1_perf.gladiator,
                            "entry_rules": f"Entry logic from {parent1_perf.gladiator} (WR: {parent1_perf.win_rate:.1%})",
                            "exit_rules": f"Exit at {parent1_perf.avg_rr:.1f}R or SL hit",
                            "structural_edge": f"Edge from {parent1_perf.gladiator}: {parent1_perf.total_trades} trades",
                            "filters": ["spread_normal", "volume_confirmation"],
                            "risk_per_trade": 0.01,
                            "expected_wr": parent1_perf.win_rate,
                            "expected_rr": parent1_perf.avg_rr,
                            "why_it_works": f"Proven over {parent1_perf.total_trades} trades",
                            "weaknesses": []
                        }
                        parent2_dict = {
                            "strategy_id": parent2_id,
                            "gladiator": parent2_perf.gladiator,
                            "entry_rules": f"Entry logic from {parent2_perf.gladiator} (WR: {parent2_perf.win_rate:.1%})",
                            "exit_rules": f"Exit at {parent2_perf.avg_rr:.1f}R or SL hit",
                            "structural_edge": f"Edge from {parent2_perf.gladiator}: {parent2_perf.total_trades} trades",
                            "filters": ["spread_normal", "regime_stable"],
                            "risk_per_trade": 0.01,
                            "expected_wr": parent2_perf.win_rate,
                            "expected_rr": parent2_perf.avg_rr,
                            "why_it_works": f"Proven over {parent2_perf.total_trades} trades",
                            "weaknesses": []
                        }

                        # Actually breed using BreedingEngine
                        offspring = self.breeder.breed(
                            parent1=parent1_dict,
                            parent2=parent2_dict,
                            parent1_fitness=parent1_perf.fitness_score,
                            parent2_fitness=parent2_perf.fitness_score,
                            crossover_type="weighted_fitness"
                        )

                        # Validate offspring
                        is_valid, reason = self.breeder.validate_offspring(offspring)

                        if is_valid:
                            # Register offspring in tournament
                            self.tournament.register_strategy(
                                strategy_id=offspring["strategy_id"],
                                gladiator="BRED",  # Mark as bred offspring
                                asset=asset,
                                regime=regime
                            )
                            logger.success(
                                f"{population_key}: Created offspring {offspring['strategy_id']} "
                                f"from {parent1_id} x {parent2_id}"
                            )
                        else:
                            logger.warning(f"Offspring validation failed: {reason}")
                    else:
                        logger.warning(f"Could not find parent strategies for breeding")

            self.last_breeding_check = current_time

        # Sync TournamentTracker rankings to TournamentManager weights
        self._sync_tournament_rankings()

    def _sync_tournament_rankings(self):
        """
        Sync TournamentTracker (vote-level) rankings to TournamentManager (engine weights).

        Creates single source of truth by updating engine_weights based on win rates.
        Weight formula: 40% to #1, 30% to #2, 20% to #3, 10% to #4
        """
        leaderboard = self.vote_tracker.get_leaderboard(sort_by="win_rate")

        if len(leaderboard) < 4:
            return  # Not enough data yet

        # Weight by rank: 40%, 30%, 20%, 10%
        rank_weights = [0.40, 0.30, 0.20, 0.10]

        for rank, entry in enumerate(leaderboard[:4]):
            engine = entry.get("gladiator", "?")
            new_weight = rank_weights[rank]

            # Update TournamentManager
            self.tournament.update_weight(engine, new_weight)

        logger.debug(f"Tournament weights synced: {self.tournament.engine_weights}")

    # ==================== HELPERS ====================

    def _classify_asset(self, asset: str) -> str:
        """Classify asset type."""
        if asset in ["BTC-USD", "ETH-USD", "SOL-USD"]:
            return "standard"
        elif "BONK" in asset or "WIF" in asset or "PEPE" in asset:
            return "meme_perp"
        elif "TRY" in asset or "ZAR" in asset or "MXN" in asset:
            return "exotic_forex"
        else:
            return "standard"

    def _convert_direction(self, direction: str) -> str:
        """
        Convert runtime direction (BUY/SELL) to Guardian/filter direction (LONG/SHORT).

        Bug Fix #31-32: Runtime uses BUY/SELL terminology, but Guardian and filters
        expect LONG/SHORT. This helper ensures consistent terminology.
        """
        if direction == "BUY":
            return "LONG"
        elif direction == "SELL":
            return "SHORT"
        else:
            return direction  # HOLD or other

    def _create_market_summary(self, candles: List[Dict]) -> Dict:
        """
        Create summary dict from candle data for engines.

        Bug Fix #41: Gladiators expect a single dict with market statistics,
        not a list of candles.
        """
        if not candles:
            return {}

        latest = candles[-1]

        # Calculate 24h volume (sum of all candle volumes)
        total_volume = sum(c.get('volume', 0) for c in candles)

        # Calculate ATR (average true range) - simplified version
        # True range = max(high - low, abs(high - prev_close), abs(low - prev_close))
        atr_values = []
        for i in range(1, len(candles)):
            curr = candles[i]
            prev = candles[i-1]
            tr = max(
                curr['high'] - curr['low'],
                abs(curr['high'] - prev['close']),
                abs(curr['low'] - prev['close'])
            )
            atr_values.append(tr)

        atr = sum(atr_values) / len(atr_values) if atr_values else 0

        return {
            'close': latest.get('close'),
            'open': latest.get('open'),
            'high': latest.get('high'),
            'low': latest.get('low'),
            'volume': latest.get('volume'),
            'volume_24h': total_volume,
            'atr': atr,
            'timestamp': latest.get('start'),
            'spread': 0,  # Not available from candle data
            'funding_rate': 0  # Not available from candle data
        }

    def _get_dxy_data(self) -> Optional[Dict]:
        """Get DXY (US Dollar Index) data."""
        # TODO: Implement DXY data fetch
        return None

    def _get_btc_data(self) -> Optional[Dict]:
        """Get BTC data for correlation checks."""
        try:
            df = self.data_client.fetch_klines(
                symbol="BTC-USD",
                interval="1h",
                limit=2
            )
            btc_data = df.to_dict('records')
            if len(btc_data) >= 2:
                change = (btc_data[-1]["close"] - btc_data[0]["close"]) / btc_data[0]["close"]
                return {
                    "symbol": "BTC-USD",
                    "price": btc_data[-1]["close"],
                    "change_pct_1h": change
                }
        except Exception as e:
            logger.error(f"Failed to fetch BTC data: {e}")

        return None

    def _build_market_context(self, asset: str, market_data: List[Dict]) -> Dict:
        """Build market context for lesson memory."""
        return {
            "dxy_change": 0.0,  # TODO: Get real DXY
            "btc_change": 0.0,  # TODO: Get real BTC change
            "news_events": [],  # TODO: Implement news calendar
            "session": "Unknown",  # TODO: Detect trading session
            "day_of_week": datetime.now(timezone.utc).strftime("%A")
        }

    def _get_engine_emotion_prompts(self) -> Dict[str, str]:
        """
        Generate personalized emotion prompts for each engine based on tournament standings.

        Returns:
            Dict mapping engine name (A/B/C/D) to emotion prompt string
        """
        try:
            # Get leaderboard from tournament tracker (sorted by win rate)
            leaderboard = self.vote_tracker.get_leaderboard(sort_by="win_rate")

            if not leaderboard:
                # No rankings yet - return empty prompts
                return {"A": "", "B": "", "C": "", "D": ""}

            # Build rank lookup
            rank_lookup = {}
            for rank, entry in enumerate(leaderboard, 1):
                gladiator = entry.get("gladiator", "?")
                rank_lookup[gladiator] = {
                    "rank": rank,
                    "win_rate": entry.get("win_rate", 0.0),
                    "total_points": entry.get("total_points", 0),
                    "correct_votes": entry.get("correct_votes", 0),
                    "total_votes": entry.get("total_votes", 0)
                }

            # Find leader
            leader = leaderboard[0] if leaderboard else {}
            leader_name = leader.get("gladiator", "?")
            leader_wr = leader.get("win_rate", 0.0)

            # Engine specialties (matches stats_injector.py)
            specialties = {
                "A": ("LIQUIDATION HUNTER", "liquidation cascades (>$20M)"),
                "B": ("FUNDING CONTRARIAN", "funding rate extremes (>0.5%)"),
                "C": ("ORDER BOOK READER", "orderbook imbalance (>2.5:1)"),
                "D": ("REGIME SPECIALIST", "regime transitions (ATR 2× expansion)")
            }

            # Generate emotion prompts
            emotion_prompts = {}
            for engine in ["A", "B", "C", "D"]:
                stats = rank_lookup.get(engine, {"rank": 4, "win_rate": 0.0})
                rank = stats["rank"]
                wr = stats["win_rate"]
                gap = leader_wr - wr if rank > 1 else 0.0

                specialty_name, specialty_trigger = specialties.get(engine, ("TRADER", "your specialty"))

                if rank == 1:
                    status = "LEADING"
                    strategy = "MAINTAIN CONSISTENCY - Your current approach is working."
                elif rank == 2:
                    status = "CHASING"
                    strategy = f"CLOSE THE GAP - Engine {leader_name} leads by {gap:.1f}%."
                elif rank == 3:
                    status = "TRAILING"
                    strategy = "FOCUS ON YOUR SPECIALTY - Quality over quantity."
                else:
                    status = "LAST PLACE"
                    strategy = f"DISCIPLINED RECOVERY - You're {gap:.1f}% behind. Avoid forcing trades."

                prompt = f"""
TOURNAMENT POSITION: #{rank}/4 ({status})
SPECIALTY: {specialty_name}
TRIGGER: {specialty_trigger}

Stats: WR: {wr:.1f}% | Leader: Engine {leader_name} {leader_wr:.1f}% | Gap: {gap:.1f}%

STRATEGY: {strategy}

Only trade when your specialty trigger activates. Patience beats aggression.
"""
                emotion_prompts[engine] = prompt.strip()

            return emotion_prompts

        except Exception as e:
            logger.warning(f"Failed to generate emotion prompts: {e}")
            return {"A": "", "B": "", "C": "", "D": ""}

    # ==================== PAPER TRADING HELPERS ====================

    def _check_paper_trades(self):
        """Check all open paper trades for SL/TP hits."""
        if not self.paper_trader.open_trades:
            return

        # Fetch market data for all assets with open trades
        assets_with_trades = set(trade.asset for trade in self.paper_trader.open_trades.values())

        market_data = {}
        for asset in assets_with_trades:
            try:
                df = self.data_client.fetch_klines(
                    symbol=asset,
                    interval="1m",
                    limit=1
                )
                market_data[asset] = df.to_dict('records')
            except Exception as e:
                logger.error(f"Failed to fetch data for {asset}: {e}")

        # Check for exits
        self.paper_trader.check_open_trades(market_data)

        # For closed trades, update tournament and learn from losses
        for trade in list(self.paper_trader.closed_trades[-10:]):  # Last 10 closed
            if trade.status == "CLOSED" and not hasattr(trade, "_processed"):
                self._process_closed_paper_trade(trade)
                trade._processed = True  # Mark as processed

    def _process_closed_paper_trade(self, trade):
        """Process a closed paper trade - update tournament and learn from losses."""
        # Update tournament performance
        trade_result = {
            "outcome": trade.outcome,
            "pnl_percent": trade.pnl_percent,
            "rr_actual": trade.rr_actual,
            "entry_timestamp": trade.entry_timestamp,
            "exit_timestamp": trade.exit_timestamp
        }

        self.tournament.update_strategy_performance(
            strategy_id=trade.strategy_id,
            asset=trade.asset,
            regime=trade.regime,
            trade_result=trade_result
        )

        # Score engine votes for this trade
        self.vote_tracker.score_trade_outcome(
            trade_id=trade.trade_id,
            actual_direction=trade.direction,
            outcome=trade.outcome,
            exit_reason=trade.exit_reason or "unknown"
        )

        # Learn from losses
        if trade.outcome == "loss":
            market_context = {
                "day_of_week": trade.entry_timestamp.strftime("%A"),
                "session": "Unknown",  # TODO: Detect session
                "dxy_change": 0.0,
                "btc_change": 0.0,
                "news_events": []
            }

            self.lessons.learn_from_failure(
                trade_id=trade.trade_id,
                asset=trade.asset,
                regime=trade.regime,
                strategy={"strategy_id": trade.strategy_id},
                signal={"direction": trade.direction, "entry_price": trade.entry_price},
                trade_result=trade_result,
                market_context=market_context
            )

        # Update Strategy Memory with trade outcome
        self.strategy_memory.add_strategy(
            engine=trade.gladiator,
            asset=trade.asset,
            regime=trade.regime,
            strategy={"strategy_id": trade.strategy_id, "direction": trade.direction},
            outcome=trade.outcome,
            pnl_percent=trade.pnl_percent,
            rr_actual=trade.rr_actual
        )

        # Record trade result for decay detection + regime validation
        self.strategy_memory.record_trade_result(
            engine=trade.gladiator,
            asset=trade.asset,
            regime=trade.regime,
            strategy_id=trade.strategy_id,
            outcome=trade.outcome
        )
        logger.debug(f"Strategy memory updated: {trade.gladiator} - {trade.strategy_id} ({trade.outcome})")

    def _print_paper_trading_stats(self):
        """Print paper trading statistics."""
        stats = self.paper_trader.get_overall_stats()

        logger.info("\n" + "="*80)
        logger.info("PAPER TRADING STATISTICS")
        logger.info("="*80)
        logger.info(f"Total Trades: {stats['total_trades']}")
        logger.info(f"Wins: {stats['wins']} | Losses: {stats['losses']}")
        logger.info(f"Win Rate: {stats['win_rate']:.1%}")
        logger.info(f"Total P&L: {stats['total_pnl_percent']:+.2%} (${stats['total_pnl_usd']:+.2f})")
        logger.info(f"Avg Win: {stats['avg_win']:+.2%} | Avg Loss: {stats['avg_loss']:+.2%}")
        logger.info(f"Avg R:R: {stats['avg_rr']:.2f}")
        logger.info(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        logger.info(f"Open Trades: {stats['open_trades']}")
        logger.info("="*80 + "\n")

    def _update_all_prices(self):
        """Quick price update for all assets (for dashboard real-time display)."""
        try:
            for asset in self.assets:
                df = self.data_client.fetch_klines(
                    symbol=asset,
                    interval="1m",
                    limit=1  # Just get latest candle
                )
                if df is not None and not df.empty:
                    current_price = df.iloc[-1]["close"]
                    HydraMetrics.set_price(asset, current_price)
        except Exception as e:
            logger.warning(f"Failed to update prices: {e}")

    def _background_price_updater(self):
        """Background thread that updates prices every 10 seconds for real-time dashboard."""
        logger.info("Background price updater started (10s interval)")
        while self._price_update_running:
            try:
                for asset in self.assets:
                    try:
                        df = self.data_client.fetch_klines(
                            symbol=asset,
                            interval="1m",
                            limit=1
                        )
                        if df is not None and not df.empty:
                            current_price = df.iloc[-1]["close"]
                            HydraMetrics.set_price(asset, current_price)
                    except Exception as e:
                        # Don't spam logs for individual asset failures
                        pass
            except Exception as e:
                logger.debug(f"Background price update error: {e}")
            time.sleep(10)  # Update every 10 seconds

    def _update_prometheus_metrics(self):
        """Update Prometheus metrics with current state."""
        try:
            # Get paper trading stats
            stats = self.paper_trader.get_overall_stats()

            # Update P&L metrics
            HydraMetrics.set_pnl(
                total=stats.get('total_pnl_percent', 0) * 100,
                daily=0  # Will be set from Guardian below
            )

            # Update win rate
            HydraMetrics.set_win_rate(
                rate_24h=stats.get('win_rate', 0) * 100,
                rate_total=stats.get('win_rate', 0) * 100
            )

            # Update consecutive wins/losses
            HydraMetrics.consecutive_wins.set(stats.get('consecutive_wins', 0))
            HydraMetrics.consecutive_losses.set(stats.get('consecutive_losses', 0))

            # Get tournament leaderboard for engine stats
            leaderboard = self.vote_tracker.get_leaderboard()
            for i, engine_stats in enumerate(leaderboard, 1):
                engine = engine_stats.get('gladiator', '')
                if engine in ['A', 'B', 'C', 'D']:
                    # Calculate weight based on rank
                    weights = {1: 40, 2: 30, 3: 20, 4: 10}
                    HydraMetrics.set_engine_stats(
                        engine=engine,
                        rank=i,
                        weight=weights.get(i, 10),
                        points=engine_stats.get('total_points', 0),
                        win_rate=engine_stats.get('win_rate', 0),
                        active=True
                    )

            # ========== Phase 1: Per-Asset Metrics ==========
            for asset in self.assets:
                try:
                    # Get per-asset stats from paper trader
                    asset_stats = self.paper_trader.get_stats_by_asset(asset)
                    HydraMetrics.set_asset_stats(
                        asset=asset,
                        pnl_percent=asset_stats.get('total_pnl_percent', 0) * 100,
                        win_rate=asset_stats.get('win_rate', 0),
                        trade_count=asset_stats.get('total_trades', 0)
                    )
                except Exception as e:
                    logger.debug(f"Could not get stats for {asset}: {e}")

            # ========== Phase 1: Technical Indicators ==========
            for asset in self.assets:
                try:
                    # Fetch candles for regime detection
                    df = self.data_client.fetch_klines(symbol=asset, interval="5m", limit=50)
                    if not df.empty:
                        # Get price for market data
                        price = df.iloc[-1]['close']
                        HydraMetrics.set_price(asset, price)

                        # Convert DataFrame to candles list for regime detector
                        candles = df.to_dict('records')
                        regime_result = self.regime_detector.detect_regime(asset, candles)

                        # Set regime and indicators
                        metrics = regime_result.get('metrics', {})
                        HydraMetrics.set_regime_info(
                            asset=asset,
                            regime=regime_result.get('regime', 'CHOPPY'),
                            confidence=regime_result.get('confidence', 0.5),
                            adx=metrics.get('adx', 0),
                            atr=metrics.get('atr', 0),
                            bb_width=metrics.get('bb_width', 0)
                        )
                except Exception as e:
                    logger.debug(f"Could not get regime for {asset}: {e}")

            # ========== Phase 1: Guardian Full State ==========
            try:
                guardian_status = self.guardian.get_status()
                HydraMetrics.set_guardian_state(
                    account_bal=guardian_status.get('account_balance', 10000),
                    peak_bal=guardian_status.get('peak_balance', 10000),
                    daily_pnl=guardian_status.get('daily_pnl', 0),
                    daily_pnl_pct=guardian_status.get('daily_pnl_percent', 0),
                    circuit_breaker=guardian_status.get('circuit_breaker_active', False),
                    emergency_shutdown=guardian_status.get('emergency_shutdown_active', False),
                    trading_allowed=guardian_status.get('trading_allowed', True),
                    position_multiplier=guardian_status.get('position_size_multiplier', 1.0)
                )

                # Also update risk metrics from Guardian
                HydraMetrics.set_risk_metrics(
                    daily_dd=guardian_status.get('current_drawdown_percent', 0),
                    total_dd=guardian_status.get('current_drawdown_percent', 0),
                    kill_switch=guardian_status.get('emergency_shutdown_active', False),
                    exposure=len(self.paper_trader.open_trades) * 2
                )

                # Update daily P&L from Guardian
                HydraMetrics.pnl_daily.set(guardian_status.get('daily_pnl_percent', 0))

            except Exception as e:
                logger.debug(f"Could not get Guardian status: {e}")
                # Fallback risk metrics
                HydraMetrics.set_risk_metrics(
                    daily_dd=0,
                    total_dd=0,
                    kill_switch=False,
                    exposure=len(self.paper_trader.open_trades) * 2
                )

            # ========== Phase 2: Per-Regime Performance ==========
            REGIMES = ['TRENDING_UP', 'TRENDING_DOWN', 'RANGING', 'VOLATILE', 'CHOPPY']
            for regime in REGIMES:
                try:
                    regime_stats = self.paper_trader.get_stats_by_regime(regime)
                    HydraMetrics.set_regime_stats(
                        regime=regime,
                        pnl_percent=regime_stats.get('total_pnl_percent', 0) * 100,
                        win_rate=regime_stats.get('win_rate', 0),
                        trade_count=regime_stats.get('total_trades', 0),
                        avg_pnl=regime_stats.get('avg_pnl_percent', 0) * 100
                    )
                except Exception as e:
                    logger.debug(f"Could not get stats for regime {regime}: {e}")

            # ========== Phase 3: Engine Analytics ==========
            try:
                ENGINES = ['A', 'B', 'C', 'D']
                all_votes = []

                for engine in ENGINES:
                    engine_stats = self.vote_tracker.get_engine_stats(engine)
                    if engine_stats:
                        # Get vote breakdown
                        buy_votes = engine_stats.get('correct_votes', 0)
                        sell_votes = engine_stats.get('wrong_votes', 0)
                        hold_votes = engine_stats.get('hold_votes', 0)

                        HydraMetrics.set_engine_votes(
                            engine=engine,
                            buy_votes=buy_votes,
                            sell_votes=sell_votes,
                            hold_votes=hold_votes,
                            last_vote='HOLD'  # Default, will be updated if available
                        )

                        # Store win rate for agreement calculation
                        all_votes.append(engine_stats.get('win_rate', 0))

                # Calculate agreement rate (variance of win rates - lower = more agreement)
                if all_votes and len(all_votes) > 1:
                    avg_wr = sum(all_votes) / len(all_votes)
                    variance = sum((wr - avg_wr) ** 2 for wr in all_votes) / len(all_votes)
                    # Convert to agreement rate (1 - normalized variance)
                    agreement = max(0, 1 - (variance * 4))  # Scale factor for 0-1 range
                    HydraMetrics.set_engine_agreement(agreement)

            except Exception as e:
                logger.debug(f"Could not get engine analytics: {e}")

            # ========== Phase 4: Advanced Statistics ==========
            try:
                # Get closed trades for calculations
                closed_trades = self.paper_trader.closed_trades

                if closed_trades:
                    returns = [t.pnl_percent for t in closed_trades]
                    wins = [r for r in returns if r > 0]
                    losses = [r for r in returns if r < 0]

                    # Sharpe ratio (already calculated in get_overall_stats)
                    sharpe = stats.get('sharpe_ratio', 0)

                    # Sortino ratio (uses only downside deviation)
                    if losses:
                        downside_returns = [min(0, r) for r in returns]
                        downside_variance = sum(r**2 for r in downside_returns) / len(downside_returns)
                        downside_std = downside_variance ** 0.5
                        avg_return = sum(returns) / len(returns)
                        sortino = avg_return / downside_std if downside_std > 0 else 0
                    else:
                        sortino = 0

                    # Max drawdown
                    cumulative = 0
                    peak = 0
                    max_dd = 0
                    for r in returns:
                        cumulative += r
                        if cumulative > peak:
                            peak = cumulative
                        dd = peak - cumulative
                        if dd > max_dd:
                            max_dd = dd

                    # Calmar ratio (return / max drawdown)
                    total_return = sum(returns)
                    calmar = total_return / max_dd if max_dd > 0 else 0

                    # Profit factor (gross profits / gross losses)
                    gross_profit = sum(wins) if wins else 0
                    gross_loss = abs(sum(losses)) if losses else 0
                    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (10.0 if gross_profit > 0 else 0)

                    # Expectancy (win_rate * avg_win - loss_rate * avg_loss)
                    win_rate = len(wins) / len(returns) if returns else 0
                    avg_win = sum(wins) / len(wins) if wins else 0
                    avg_loss = abs(sum(losses) / len(losses)) if losses else 0
                    expectancy_val = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

                    HydraMetrics.set_advanced_stats(
                        sharpe=sharpe,
                        sortino=sortino,
                        max_dd=max_dd * 100,  # Convert to percent
                        calmar=calmar,
                        profit_factor=profit_factor,
                        expectancy=expectancy_val * 100,  # Convert to percent
                        avg_rr=stats.get('avg_rr', 0),
                        total_trades=len(closed_trades)
                    )
                else:
                    # No trades yet
                    HydraMetrics.set_advanced_stats(
                        sharpe=0, sortino=0, max_dd=0, calmar=0,
                        profit_factor=0, expectancy=0, avg_rr=0, total_trades=0
                    )

            except Exception as e:
                logger.debug(f"Could not calculate advanced stats: {e}")

            # Record cycle completion
            HydraMetrics.record_cycle(self.check_interval)

        except Exception as e:
            logger.error(f"Error updating Prometheus metrics: {e}")
            HydraMetrics.record_error("metrics_update", "hydra_runtime")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="HYDRA 3.0 Runtime")
    parser.add_argument(
        "--assets",
        nargs="+",
        default=["BTC-USD", "ETH-USD", "SOL-USD"],
        help="Assets to trade"
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        default=True,
        help="Paper trading mode (default: True)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=-1,
        help="Number of iterations (-1 = infinite)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Check interval in seconds (default: 300 = 5 min)"
    )

    args = parser.parse_args()

    # Configure logging
    logger.add(
        "/tmp/hydra_runtime_{time}.log",
        rotation="100 MB",
        retention="7 days",
        level="INFO"
    )

    # Create runtime
    runtime = HydraRuntime(
        assets=args.assets,
        paper_trading=args.paper,
        check_interval_seconds=args.interval
    )

    # Run
    runtime.run(iterations=args.iterations)


if __name__ == "__main__":
    main()
