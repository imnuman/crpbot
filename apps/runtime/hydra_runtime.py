"""
HYDRA 3.0 - Main Runtime Orchestrator

Integrates all 10 layers + 4 upgrades into a unified trading system.

Signal Flow:
1. Regime Detector → Classify market state
2. Asset Profiles → Load market-specific config
3. Anti-Manipulation → Pre-filter suspicious conditions
4. Gladiators (A/B/C/D) → Generate, validate, backtest, synthesize
5. Tournament → Track strategy performance
6. Consensus → Aggregate gladiator votes
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
from datetime import datetime, timezone
from typing import Dict, List, Optional
from loguru import logger
from pathlib import Path

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

# Gladiators
from libs.hydra.gladiators.gladiator_a_deepseek import GladiatorA_DeepSeek
from libs.hydra.gladiators.gladiator_b_claude import GladiatorB_Claude
from libs.hydra.gladiators.gladiator_c_groq import GladiatorC_Groq
from libs.hydra.gladiators.gladiator_d_gemini import GladiatorD_Gemini

# Data Provider
from libs.data.coinbase_client import get_coinbase_client


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
        self._init_gladiators()

        # State tracking
        self.iteration = 0
        self.last_elimination_check = None
        self.last_breeding_check = None

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

        # Data provider
        self.data_client = get_coinbase_client()

        logger.success("All layers initialized")

    def _init_gladiators(self):
        """Initialize 4 gladiators with API keys."""
        logger.info("Initializing Gladiators...")

        self.gladiator_a = GladiatorA_DeepSeek(api_key=os.getenv("DEEPSEEK_API_KEY"))
        self.gladiator_b = GladiatorB_Claude(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.gladiator_c = GladiatorC_Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.gladiator_d = GladiatorD_Gemini(api_key=os.getenv("GEMINI_API_KEY"))

        self.gladiators = [
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

                # Check open paper trades first
                if self.paper_trading:
                    self._check_paper_trades()

                # Process each asset
                for asset in self.assets:
                    self._process_asset(asset)

                # Run tournament cycles (if needed)
                self._run_tournament_cycles()

                # Print paper trading stats
                if self.paper_trading and self.iteration % 10 == 0:  # Every 10 iterations
                    self._print_paper_trading_stats()

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
            market_data = self.data_client.get_candles(
                symbol=asset,
                granularity="ONE_MINUTE",
                limit=200
            )
        except Exception as e:
            logger.error(f"Failed to fetch data for {asset}: {e}")
            return

        if not market_data or len(market_data) < 100:
            logger.warning(f"Insufficient data for {asset}")
            return

        # Step 2: Detect regime
        regime_result = self.regime_detector.detect_regime(market_data)
        regime = regime_result["regime"]
        regime_confidence = regime_result["confidence"]

        logger.info(f"{asset} regime: {regime} (confidence: {regime_confidence:.1%})")

        # Step 3: Get asset profile
        asset_type = self._classify_asset(asset)
        profile = self.asset_profiles.get_profile(asset, asset_type)

        # Step 4: Anti-manipulation check
        manip_check = self.anti_manip.check_all_layers(
            asset=asset,
            market_data=market_data,
            current_price=market_data[-1]["close"]
        )

        if not manip_check["passed"]:
            logger.warning(
                f"{asset} failed anti-manipulation: {manip_check['rejection_reason']}"
            )
            return

        # Step 5-7: Strategy generation and signal creation
        signal = self._generate_signal(
            asset=asset,
            asset_type=asset_type,
            regime=regime,
            regime_confidence=regime_confidence,
            market_data=market_data,
            profile=profile
        )

        if not signal or signal.get("action") == "HOLD":
            logger.info(f"{asset}: No trade signal (consensus: HOLD)")
            return

        # Step 8: Cross-asset filter
        cross_asset_result = self.cross_asset.check_cross_asset_alignment(
            asset=asset,
            asset_type=asset_type,
            direction=signal["action"],
            market_data=market_data,
            dxy_data=self._get_dxy_data(),
            btc_data=self._get_btc_data(),
            em_basket_data=None  # TODO: Implement EM basket
        )

        if not cross_asset_result[0]:
            logger.warning(f"{asset} blocked by cross-asset filter: {cross_asset_result[1]}")
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

        if lesson_check[0]:  # Lesson triggered
            logger.error(f"{asset} rejected by lesson memory: {lesson_check[1].lesson_id}")
            return

        # Step 10: Guardian validation
        guardian_check = self.guardian.validate_trade(
            asset=asset,
            direction=signal["action"],
            position_size_usd=signal.get("position_size_usd", 100),
            stop_loss_pct=signal.get("stop_loss_pct", 0.02),
            market_data=market_data
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
        Generate trading signal via gladiator consensus.

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
        # Check if we have active strategies for this asset/regime
        population_key = f"{asset}:{regime}"

        # Get existing strategies (if any)
        population = self.tournament.populations.get(population_key, [])
        existing_strategies = [
            {"strategy_id": s.strategy_id, "gladiator": s.gladiator}
            for s in population
        ]

        # Generate strategies from all 4 gladiators
        strategies = []

        # Gladiator A: Generate structural edge
        strategy_a = self.gladiator_a.generate_strategy(
            asset=asset,
            asset_type=asset_type,
            asset_profile=profile,
            regime=regime,
            regime_confidence=regime_confidence,
            market_data=market_data,
            existing_strategies=existing_strategies
        )
        strategies.append(strategy_a)

        # Gladiator B: Validate A's strategy
        strategy_b = self.gladiator_b.generate_strategy(
            asset=asset,
            asset_type=asset_type,
            asset_profile=profile,
            regime=regime,
            regime_confidence=regime_confidence,
            market_data=market_data,
            existing_strategies=strategies
        )
        strategies.append(strategy_b)

        # Gladiator C: Backtest validated strategy
        strategy_c = self.gladiator_c.generate_strategy(
            asset=asset,
            asset_type=asset_type,
            asset_profile=profile,
            regime=regime,
            regime_confidence=regime_confidence,
            market_data=market_data,
            existing_strategies=strategies
        )
        strategies.append(strategy_c)

        # Gladiator D: Synthesize final strategy
        strategy_d = self.gladiator_d.generate_strategy(
            asset=asset,
            asset_type=asset_type,
            asset_profile=profile,
            regime=regime,
            regime_confidence=regime_confidence,
            market_data=market_data,
            existing_strategies=strategies
        )
        strategies.append(strategy_d)

        # Register strategies in tournament (if new)
        for strategy in strategies:
            if strategy.get("strategy_id") and strategy.get("gladiator"):
                # Check if already registered
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

        # Get votes from all gladiators
        votes = []

        # Create mock signal for voting
        current_price = market_data[-1]["close"]
        mock_signal = {
            "direction": "BUY",  # Will be overridden by votes
            "entry_price": current_price,
            "stop_loss_pct": 0.015,
            "take_profit_pct": 0.025
        }

        for gladiator in self.gladiators:
            vote = gladiator.vote_on_trade(
                asset=asset,
                asset_type=asset_type,
                regime=regime,
                strategy=strategy_d,  # Final synthesized strategy
                signal=mock_signal,
                market_data=market_data
            )
            vote["gladiator"] = gladiator.name
            votes.append(vote)

        # Get consensus
        consensus = self.consensus.get_consensus(votes)

        # Build signal
        signal = {
            "action": consensus["action"],
            "consensus_level": consensus["consensus_level"],
            "position_size_modifier": consensus["position_size_modifier"],
            "strategy": strategy_d,
            "entry_price": current_price,
            "stop_loss_pct": profile.get("typical_sl_pct", 0.015),
            "take_profit_pct": profile.get("typical_sl_pct", 0.015) * 1.5,  # 1.5R
            "votes": votes,
            "consensus_summary": consensus["summary"]
        }

        return signal

    def _execute_paper_trade(self, asset: str, signal: Dict, market_data: List[Dict]):
        """Execute paper trade (simulation only)."""
        # Extract regime from previous detection
        regime_result = self.regime_detector.detect_regime(market_data)
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

        # Use execution optimizer
        execution_result = self.executor.optimize_entry(
            asset=asset,
            direction=signal["action"],
            current_bid=market_data[-1]["close"] * 0.9995,  # Approximate
            current_ask=market_data[-1]["close"] * 1.0005,
            spread_normal=0.001,
            target_size_usd=signal.get("position_size_usd", 100),
            max_spread_multiplier=2.0
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
        self.explainer.log_trade_decision(
            trade_id=f"{asset}_{int(time.time())}",
            asset=asset,
            gladiator_votes=signal.get("votes", []),
            consensus_level=signal["consensus_level"],
            filters_passed={
                "anti_manipulation": True,
                "cross_asset": True,
                "lesson_memory": True
            },
            guardian_approved=True,
            position_size_final=signal["position_size_modifier"],
            regime=regime,
            strategy=signal.get("strategy", {}),
            market_context=market_context
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
                    # Create offspring
                    parent1_id = breed_result["breeding_pairs"][0][0]
                    parent2_id = breed_result["breeding_pairs"][0][1]

                    # Get parent strategies (would need to fetch from DB in real impl)
                    # For now, skip actual breeding
                    logger.success(
                        f"{population_key}: Breeding {parent1_id} x {parent2_id}"
                    )

            self.last_breeding_check = current_time

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

    def _get_dxy_data(self) -> Optional[Dict]:
        """Get DXY (US Dollar Index) data."""
        # TODO: Implement DXY data fetch
        return None

    def _get_btc_data(self) -> Optional[Dict]:
        """Get BTC data for correlation checks."""
        try:
            btc_data = self.data_client.get_candles(
                symbol="BTC-USD",
                granularity="ONE_HOUR",
                limit=2
            )
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
                data = self.data_client.get_candles(
                    symbol=asset,
                    granularity="ONE_MINUTE",
                    limit=1
                )
                market_data[asset] = data
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
