#!/usr/bin/env python3
"""
HYDRA Tournament Leaderboard CLI

Quick utility to view current tournament standings.

Usage:
    python scripts/show_leaderboard.py                  # Show full leaderboard
    python scripts/show_leaderboard.py --gladiator A    # Show Gladiator A stats
    python scripts/show_leaderboard.py --recent 24      # Show last 24h performance
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from libs.hydra.tournament_tracker import TournamentTracker


def main():
    parser = argparse.ArgumentParser(description="View HYDRA tournament standings")
    parser.add_argument('--gladiator', choices=['A', 'B', 'C', 'D'], help="Show specific gladiator stats")
    parser.add_argument('--recent', type=int, metavar='HOURS', help="Show recent performance (hours)")
    parser.add_argument('--sort', choices=['points', 'win_rate', 'votes'], default='points',
                        help="Sort leaderboard by (default: points)")
    args = parser.parse_args()

    tracker = TournamentTracker()

    if args.gladiator:
        # Show specific gladiator stats
        stats = tracker.get_gladiator_stats(args.gladiator)
        if not stats:
            print(f"No data found for Gladiator {args.gladiator}")
            return

        print(f"\n{'=' * 80}")
        print(f"GLADIATOR {args.gladiator} - DETAILED STATS")
        print(f"{'=' * 80}")
        print(f"Total Points:      {stats['total_points']}")
        print(f"Total Votes:       {stats['total_votes']}")
        print(f"Win Rate:          {stats['win_rate']:.1f}%")
        print(f"Correct Votes:     {stats['correct_votes']}")
        print(f"Wrong Votes:       {stats['wrong_votes']}")
        print(f"HOLD Votes:        {stats['hold_votes']}")
        print(f"Best Asset:        {stats.get('best_asset', 'N/A')}")
        print(f"Worst Asset:       {stats.get('worst_asset', 'N/A')}")
        print(f"Last Updated:      {stats.get('last_updated', 'N/A')}")

        # Per-asset breakdown
        votes_by_asset = stats.get('votes_by_asset', {})
        if votes_by_asset:
            print(f"\n{'Asset':<12} {'Total Votes':<15} {'Correct':<10} {'Win Rate':<10}")
            print("-" * 50)
            for asset, asset_stats in sorted(votes_by_asset.items()):
                total = asset_stats['total']
                correct = asset_stats['correct']
                win_rate = (correct / total * 100) if total > 0 else 0
                print(f"{asset:<12} {total:<15} {correct:<10} {win_rate:.1f}%")

        print(f"{'=' * 80}\n")

    elif args.recent:
        # Show recent performance for all gladiators
        print(f"\n{'=' * 80}")
        print(f"RECENT PERFORMANCE (Last {args.recent} hours)")
        print(f"{'=' * 80}")
        print(f"{'Gladiator':<12} {'Votes':<10} {'Points':<10} {'Win Rate':<10}")
        print("-" * 80)

        for glad in ['A', 'B', 'C', 'D']:
            recent = tracker.get_recent_performance(glad, hours=args.recent)
            if recent['total_votes'] > 0:
                print(f"Gladiator {glad:<4} {recent['total_votes']:<10} "
                      f"{recent['points_earned']:<10} {recent['win_rate']:.1f}%")

        print(f"{'=' * 80}\n")

    else:
        # Show full leaderboard
        sort_map = {
            'points': 'total_points',
            'win_rate': 'win_rate',
            'votes': 'total_votes'
        }
        tracker.print_leaderboard()


if __name__ == "__main__":
    main()
