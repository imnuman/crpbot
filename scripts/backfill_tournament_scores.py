#!/usr/bin/env python3
"""
Backfill Tournament Scores from Historical Paper Trades

This script processes all historical paper trades and scores them
for the tournament tracker.

Since HYDRA doesn't currently record individual gladiator votes,
we'll infer voting from the final consensus decision:
- If gladiator made the final call (gladiator field), they "voted" for that direction
- All gladiators are assumed to have participated in consensus

This is a simplified model until vote recording is integrated into runtime.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from libs.hydra.tournament_tracker import TournamentTracker


def load_paper_trades(file_path: str) -> List[Dict]:
    """Load all paper trades from JSONL file."""
    trades = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    trade = json.loads(line)
                    trades.append(trade)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line: {e}")
    return trades


def main():
    logger.info("Starting tournament score backfill...")

    # Initialize tracker
    tracker = TournamentTracker()

    # Load paper trades
    trades_file = "data/hydra/paper_trades.jsonl"
    if not Path(trades_file).exists():
        logger.error(f"Paper trades file not found: {trades_file}")
        return

    trades = load_paper_trades(trades_file)
    logger.info(f"Loaded {len(trades)} paper trades")

    # Filter to only closed trades with outcomes
    closed_trades = [
        t for t in trades
        if t.get('status') == 'CLOSED' and t.get('outcome') in ['win', 'loss']
    ]
    logger.info(f"Found {len(closed_trades)} closed trades with outcomes")

    # Process each trade
    for trade in closed_trades:
        trade_id = trade['trade_id']
        gladiator = trade.get('gladiator', 'D')  # Default to D (final synthesizer)
        asset = trade['asset']
        direction = trade['direction']
        outcome = trade['outcome']
        exit_reason = trade.get('exit_reason', 'unknown')

        # Record simplified vote (gladiator who made final call)
        # Note: This is a simplified model - real implementation should record all 4 votes
        tracker.record_vote(
            trade_id=trade_id,
            gladiator=gladiator,
            asset=asset,
            vote=direction,
            confidence=0.75,  # We don't have historical confidence data
            reasoning=f"Final decision maker (backfilled from paper trade)"
        )

        # Score the outcome
        tracker.score_trade_outcome(
            trade_id=trade_id,
            actual_direction=direction,
            outcome=outcome,
            exit_reason=exit_reason
        )

    logger.success(f"Backfilled scores for {len(closed_trades)} trades")

    # Print leaderboard
    tracker.print_leaderboard()

    # Print detailed stats
    print("\n" + "=" * 80)
    print("DETAILED GLADIATOR STATS")
    print("=" * 80)

    for gladiator in ['A', 'B', 'C', 'D']:
        stats = tracker.get_gladiator_stats(gladiator)
        if stats:
            print(f"\nGladiator {gladiator}:")
            print(f"  Total Points: {stats['total_points']}")
            print(f"  Total Votes: {stats['total_votes']}")
            print(f"  Win Rate: {stats['win_rate']:.1f}%")
            print(f"  Correct: {stats['correct_votes']}, Wrong: {stats['wrong_votes']}, HOLD: {stats['hold_votes']}")
            print(f"  Best Asset: {stats.get('best_asset', 'N/A')}")
            print(f"  Worst Asset: {stats.get('worst_asset', 'N/A')}")

    print("\n" + "=" * 80)
    logger.info("Backfill complete!")


if __name__ == "__main__":
    main()
