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

from .engines.engine_a_deepseek import EngineA_DeepSeek
from .engines.engine_b_claude import EngineB_Claude
from .engines.engine_c_grok import EngineC_Grok
from .engines.engine_d_gemini import EngineD_Gemini


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

        # Tournament state
        self.tournament_start_time = datetime.now(timezone.utc)
        self.last_weight_adjustment = datetime.now(timezone.utc)
        self.last_breeding = datetime.now(timezone.utc)
        self.cycle_count = 0

        # Performance tracking
        self.cycles_history: List[TradingCycle] = []

        # Data persistence
        self.state_file = Path("/root/crpbot/data/hydra/mother_ai_state.json")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        logger.success("Mother AI initialized with 4 gladiators (A, B, C, D)")

    # ==================== MAIN ORCHESTRATION ====================

    def run_trading_cycle(self, asset: str, market_data: Dict) -> TradingCycle:
        """
        Run one complete trading cycle.

        Flow:
        1. Detect market regime
        2. Gather market intelligence (orderbook, search, data feeds)
        3. All 4 gladiators make independent decisions
        4. Open trades for gladiators who decided to trade
        5. Update all existing trades (SL/TP monitoring)
        6. Update tournament rankings

        Args:
            asset: Trading symbol (e.g., "BTC-USD")
            market_data: Current market data (price, volume, etc.)

        Returns:
            TradingCycle summary
        """
        self.cycle_count += 1
        cycle_start = datetime.now(timezone.utc)

        logger.info(f"\n{'='*80}")
        logger.info(f"MOTHER AI - CYCLE #{self.cycle_count} - {asset}")
        logger.info(f"{'='*80}\n")

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

        # Prepare enhanced market data with intelligence
        enhanced_market_data = {
            **market_data,
            "orderbook_analysis": market_intelligence.get("orderbook"),
            "market_feeds": market_intelligence.get("market_feeds"),
            "search_results": market_intelligence.get("search_results")
        }

        # Helper function to get decision from a single gladiator
        def get_gladiator_decision(name: str, gladiator):
            try:
                decision = gladiator.make_trade_decision(
                    asset=asset,
                    asset_type="crypto",  # TODO: Make this dynamic
                    regime=regime,
                    regime_confidence=regime_confidence,
                    market_data=enhanced_market_data
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

        # Check for weight adjustment (every 24 hours)
        time_since_adjustment = now - self.last_weight_adjustment
        if time_since_adjustment >= timedelta(hours=24):
            logger.info("🎯 24-hour mark reached - Adjusting weights based on performance")
            self._adjust_weights()
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
        Adjust gladiator weights based on performance (every 24 hours).

        Weight distribution:
        - Rank #1: 40% weight
        - Rank #2: 30% weight
        - Rank #3: 20% weight
        - Rank #4: 10% weight

        NOTE: No killing - all gladiators continue competing.
        """
        rankings = self.tournament_manager.get_tournament_summary()["rankings"]

        # Define weight distribution
        weights = {
            1: 0.40,  # 40% for rank #1
            2: 0.30,  # 30% for rank #2
            3: 0.20,  # 20% for rank #3
            4: 0.10   # 10% for rank #4
        }

        logger.info("Adjusting weights based on performance:")
        for ranking in rankings:
            gladiator_name = ranking["gladiator"]
            rank = ranking["rank"]
            new_weight = weights[rank]

            # Update weight in tournament manager
            self.tournament_manager.update_weight(gladiator_name, new_weight)

            logger.info(
                f"  [Gladiator {gladiator_name}] Rank #{rank} → Weight: {new_weight:.0%} "
                f"(P&L: ${ranking['total_pnl_usd']:+.2f}, WR: {ranking['win_rate']:.1%})"
            )

        logger.success("Weight adjustment complete - all gladiators continue competing")

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


# ==================== SINGLETON PATTERN ====================

_mother_ai = None

def get_mother_ai() -> MotherAI:
    """Get singleton instance of Mother AI."""
    global _mother_ai
    if _mother_ai is None:
        _mother_ai = MotherAI()
    return _mother_ai
