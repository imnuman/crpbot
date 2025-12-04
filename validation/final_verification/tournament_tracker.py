"""
HYDRA 3.0 - Tournament Tracker

Tracks gladiator performance, scores votes, and maintains leaderboard.

Design:
- Each gladiator vote is recorded with prediction (BUY/SELL/HOLD)
- When paper trade completes, votes are scored:
  - Correct prediction = +1 point
  - Wrong prediction = 0 points
  - HOLD vote = 0 points (neutral)
- Leaderboard shows win rate, total points, best/worst assets
- Data stored in JSONL for easy analysis
"""

from typing import Dict, List, Optional, Literal
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import json
from collections import defaultdict


VoteDirection = Literal["BUY", "SELL", "HOLD"]
Outcome = Literal["win", "loss", "neutral"]


class TournamentTracker:
    """
    Tracks gladiator voting performance and maintains competitive leaderboard.

    Scoring Rules:
    - Correct prediction (vote matches trade outcome): +1 point
    - Wrong prediction: 0 points
    - HOLD vote: 0 points (neutral, not counted)

    Examples:
    - Gladiator votes BUY → Trade wins → +1 point
    - Gladiator votes SELL → Trade wins → +1 point
    - Gladiator votes BUY → Trade loses → 0 points
    - Gladiator votes HOLD → 0 points (regardless of outcome)
    """

    def __init__(self, data_dir: str = "data/hydra"):
        self.data_dir = Path(data_dir)
        self.votes_file = self.data_dir / "tournament_votes.jsonl"
        self.scores_file = self.data_dir / "tournament_scores.jsonl"

        # Ensure directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache for fast lookups
        self.votes: Dict[str, List[Dict]] = defaultdict(list)  # trade_id -> votes
        self.scores: Dict[str, Dict] = {}  # gladiator_id -> score_stats

        self._load_existing_data()

    def _load_existing_data(self):
        """Load existing votes and scores from disk."""
        # Load votes
        if self.votes_file.exists():
            with open(self.votes_file, 'r') as f:
                for line in f:
                    if line.strip():
                        vote = json.loads(line)
                        trade_id = vote.get('trade_id')
                        if trade_id:
                            self.votes[trade_id].append(vote)
            logger.info(f"Loaded {sum(len(v) for v in self.votes.values())} votes from {self.votes_file}")

        # Load scores
        if self.scores_file.exists():
            with open(self.scores_file, 'r') as f:
                for line in f:
                    if line.strip():
                        score = json.loads(line)
                        gladiator = score.get('gladiator')
                        if gladiator:
                            self.scores[gladiator] = score
            logger.info(f"Loaded scores for {len(self.scores)} gladiators from {self.scores_file}")

    def record_vote(
        self,
        trade_id: str,
        gladiator: str,
        asset: str,
        vote: VoteDirection,
        confidence: float,
        reasoning: str = "",
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record a gladiator's vote on a potential trade.

        Args:
            trade_id: Unique trade identifier
            gladiator: Gladiator ID (A, B, C, D)
            asset: Asset being traded (e.g., BTC-USD)
            vote: BUY, SELL, or HOLD
            confidence: Vote confidence (0.0 to 1.0)
            reasoning: Why the gladiator voted this way
            timestamp: Vote timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        vote_record = {
            "trade_id": trade_id,
            "gladiator": gladiator,
            "asset": asset,
            "vote": vote,
            "confidence": confidence,
            "reasoning": reasoning,
            "timestamp": timestamp.isoformat(),
            "scored": False  # Not yet scored
        }

        # Add to in-memory cache
        self.votes[trade_id].append(vote_record)

        # Append to file
        with open(self.votes_file, 'a') as f:
            f.write(json.dumps(vote_record) + '\n')

        logger.debug(f"Recorded vote: Gladiator {gladiator} votes {vote} on {asset} (trade: {trade_id})")

    def score_trade_outcome(
        self,
        trade_id: str,
        actual_direction: Literal["BUY", "SELL"],
        outcome: Outcome,
        exit_reason: str = ""
    ) -> Dict[str, int]:
        """
        Score all gladiator votes for a completed trade.

        Scoring Logic:
        - If trade WON: gladiators who voted same direction as trade get +1
        - If trade LOST: gladiators who voted opposite direction get +1
        - HOLD votes always get 0 (neutral)

        Args:
            trade_id: Trade identifier
            actual_direction: Direction trade was executed (BUY or SELL)
            outcome: Trade outcome (win, loss, or neutral)
            exit_reason: Why trade closed (take_profit, stop_loss, timeout)

        Returns:
            Dict mapping gladiator -> points_earned
        """
        if trade_id not in self.votes:
            logger.warning(f"No votes found for trade {trade_id}")
            return {}

        points_awarded = {}

        for vote_record in self.votes[trade_id]:
            if vote_record.get('scored'):
                continue  # Already scored

            gladiator = vote_record['gladiator']
            vote = vote_record['vote']

            # Scoring logic
            points = 0

            if vote == "HOLD":
                # HOLD votes don't earn points
                points = 0
                vote_record['score_reason'] = "HOLD vote - neutral"

            elif outcome == "win":
                # Trade won - did gladiator vote same direction?
                if vote == actual_direction:
                    points = 1
                    vote_record['score_reason'] = f"Correct: voted {vote}, trade {actual_direction} won"
                else:
                    points = 0
                    vote_record['score_reason'] = f"Wrong: voted {vote}, but trade {actual_direction} won"

            elif outcome == "loss":
                # Trade lost - did gladiator vote opposite direction?
                opposite = "SELL" if actual_direction == "BUY" else "BUY"
                if vote == opposite:
                    points = 1
                    vote_record['score_reason'] = f"Correct: voted {vote} (opposite of losing {actual_direction})"
                else:
                    points = 0
                    vote_record['score_reason'] = f"Wrong: voted {vote}, trade {actual_direction} lost"

            else:
                # Neutral outcome
                points = 0
                vote_record['score_reason'] = "Neutral outcome"

            # Update vote record
            vote_record['scored'] = True
            vote_record['points_earned'] = points
            vote_record['trade_outcome'] = outcome
            vote_record['trade_direction'] = actual_direction
            vote_record['exit_reason'] = exit_reason

            points_awarded[gladiator] = points

            # Update gladiator's running score
            self._update_gladiator_score(
                gladiator=gladiator,
                points=points,
                vote=vote,
                asset=vote_record['asset'],
                outcome=outcome
            )

        logger.info(f"Scored trade {trade_id}: {points_awarded}")
        return points_awarded

    def _update_gladiator_score(
        self,
        gladiator: str,
        points: int,
        vote: str,
        asset: str,
        outcome: Outcome
    ) -> None:
        """Update a gladiator's running score statistics."""
        if gladiator not in self.scores:
            self.scores[gladiator] = {
                "gladiator": gladiator,
                "total_points": 0,
                "total_votes": 0,
                "correct_votes": 0,
                "wrong_votes": 0,
                "hold_votes": 0,
                "win_rate": 0.0,
                "votes_by_asset": defaultdict(lambda: {"total": 0, "correct": 0}),
                "best_asset": None,
                "worst_asset": None,
                "last_updated": datetime.utcnow().isoformat()
            }

        stats = self.scores[gladiator]
        stats['total_votes'] += 1

        if vote == "HOLD":
            stats['hold_votes'] += 1
        elif points > 0:
            stats['total_points'] += points
            stats['correct_votes'] += 1
        else:
            stats['wrong_votes'] += 1

        # Track per-asset performance
        asset_stats = stats['votes_by_asset']
        if asset not in asset_stats:
            asset_stats[asset] = {"total": 0, "correct": 0}

        asset_stats[asset]['total'] += 1
        if points > 0:
            asset_stats[asset]['correct'] += 1

        # Calculate win rate (excluding HOLD votes)
        non_hold_votes = stats['correct_votes'] + stats['wrong_votes']
        if non_hold_votes > 0:
            stats['win_rate'] = (stats['correct_votes'] / non_hold_votes) * 100

        # Find best/worst assets
        if len(asset_stats) > 0:
            best = max(asset_stats.items(), key=lambda x: x[1]['correct'] / max(x[1]['total'], 1))
            worst = min(asset_stats.items(), key=lambda x: x[1]['correct'] / max(x[1]['total'], 1))
            stats['best_asset'] = best[0]
            stats['worst_asset'] = worst[0]

        stats['last_updated'] = datetime.utcnow().isoformat()

        # Save to disk
        self._save_scores()

    def _save_scores(self):
        """Save current scores to disk."""
        with open(self.scores_file, 'w') as f:
            for gladiator, stats in self.scores.items():
                # Convert defaultdict to regular dict for JSON serialization
                stats_copy = stats.copy()
                stats_copy['votes_by_asset'] = dict(stats['votes_by_asset'])
                f.write(json.dumps(stats_copy) + '\n')

    def get_leaderboard(self, sort_by: str = "total_points") -> List[Dict]:
        """
        Get current tournament leaderboard.

        Args:
            sort_by: Sort criterion ("total_points", "win_rate", "total_votes")

        Returns:
            List of gladiator stats, sorted by specified criterion
        """
        leaderboard = list(self.scores.values())

        if sort_by == "total_points":
            leaderboard.sort(key=lambda x: x['total_points'], reverse=True)
        elif sort_by == "win_rate":
            leaderboard.sort(key=lambda x: x['win_rate'], reverse=True)
        elif sort_by == "total_votes":
            leaderboard.sort(key=lambda x: x['total_votes'], reverse=True)

        return leaderboard

    def get_gladiator_stats(self, gladiator: str) -> Optional[Dict]:
        """Get detailed stats for a specific gladiator."""
        return self.scores.get(gladiator)

    def get_recent_performance(self, gladiator: str, hours: int = 24) -> Dict:
        """
        Get gladiator performance in recent time window.

        Args:
            gladiator: Gladiator ID
            hours: Time window in hours

        Returns:
            Stats for recent time period
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        recent_votes = []
        for trade_votes in self.votes.values():
            for vote in trade_votes:
                if vote['gladiator'] == gladiator:
                    vote_time = datetime.fromisoformat(vote['timestamp'])
                    if vote_time >= cutoff_time and vote.get('scored'):
                        recent_votes.append(vote)

        if not recent_votes:
            return {
                "gladiator": gladiator,
                "time_window_hours": hours,
                "total_votes": 0,
                "points_earned": 0,
                "win_rate": 0.0
            }

        points = sum(v.get('points_earned', 0) for v in recent_votes)
        correct = sum(1 for v in recent_votes if v.get('points_earned', 0) > 0 and v.get('vote') != "HOLD")
        total_non_hold = sum(1 for v in recent_votes if v.get('vote') != "HOLD")

        return {
            "gladiator": gladiator,
            "time_window_hours": hours,
            "total_votes": len(recent_votes),
            "points_earned": points,
            "win_rate": (correct / total_non_hold * 100) if total_non_hold > 0 else 0.0
        }

    def print_leaderboard(self, top_n: int = 10):
        """Print current leaderboard to console."""
        leaderboard = self.get_leaderboard()

        print("\n" + "=" * 80)
        print("🏆 HYDRA TOURNAMENT LEADERBOARD 🏆")
        print("=" * 80)
        print(f"{'Rank':<6} {'Gladiator':<12} {'Points':<10} {'Win Rate':<12} {'Votes':<10} {'Best Asset':<15}")
        print("-" * 80)

        for i, stats in enumerate(leaderboard[:top_n], 1):
            gladiator = stats['gladiator']
            points = stats['total_points']
            win_rate = f"{stats['win_rate']:.1f}%"
            votes = stats['total_votes']
            best_asset = stats.get('best_asset', 'N/A')

            print(f"{i:<6} Gladiator {gladiator:<4} {points:<10} {win_rate:<12} {votes:<10} {best_asset:<15}")

        print("=" * 80 + "\n")
