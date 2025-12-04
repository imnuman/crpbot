"""
Rejection Logger

Logs all rejected signals for analysis and learning.

Problem:
- Current system only tracks executed trades
- No data on WHY signals were rejected
- Can't learn from "what not to trade"
- No analysis of false positives (over-filtering)

Solution:
Rejection Database:
├── Every rejected signal logged with full context
├── Rejection reason (regime, R:R, correlation, drawdown, etc.)
├── Market conditions at rejection time
├── Counterfactual tracking (what would have happened?)
└── Pattern analysis (most common rejections)

Learning Loop:
1. Log rejection with full context
2. Track hypothetical outcome (would it have won/lost?)
3. Analyze if rejection was correct
4. Generate recommendations for filter tuning

Expected Impact:
- Filter Tuning: Data-driven threshold optimization
- Learning: Understand what not to trade
- Missed Opportunities: Identify over-filtering (rejections that would have won)
- Agent Intelligence: Feed rejection data back to agent for self-improvement
"""
import logging
import sqlite3
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class RejectionLogger:
    """
    Log all rejected signals for analysis and learning

    Purpose:
    - Track why signals are rejected
    - Analyze if rejections were correct (counterfactual tracking)
    - Identify over-filtering (missed profitable opportunities)
    - Provide data-driven feedback for filter tuning

    Usage:
        logger = RejectionLogger(db_path="tradingai.db")

        # When rejecting a signal
        rejection_id = logger.log_rejection(
            symbol='BTC-USD',
            direction='long',
            confidence=0.75,
            rejection_reason='Chop detected (ADX=15)',
            rejection_category='regime',
            rejection_details={'adx': 15, 'threshold': 20},
            market_context={'regime': 'Ranging/Chop', 'volatility': 0.02},
            theory_scores={'shannon_entropy': 0.8, 'hurst': 0.45, ...},
            hypothetical_prices={'entry': 99000, 'sl': 98000, 'tp': 101000}
        )

        # Later, track what actually happened
        logger.track_counterfactual(
            rejection_id=rejection_id,
            outcome='would_lose',  # or 'would_win', 'would_hold'
            hypothetical_pnl=-1.0  # Would have lost 1%
        )

        # Analyze rejection patterns
        analysis = logger.analyze_rejections(time_period='7d')
        print(f"Total rejections: {analysis['total_rejections']}")
        print(f"Rejection accuracy: {analysis['rejection_accuracy']}")
        print(f"Recommendations: {analysis['recommendations']}")
    """

    def __init__(self, db_path: str = "tradingai.db"):
        """
        Initialize Rejection Logger

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._create_table()
        logger.info(f"Rejection Logger initialized | DB: {db_path}")

    def _create_table(self):
        """Create signal_rejections table if it doesn't exist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_rejections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,
            confidence REAL NOT NULL,

            -- Rejection details
            rejection_reason TEXT NOT NULL,
            rejection_category TEXT NOT NULL,
            rejection_details TEXT,  -- JSON

            -- Market context
            regime TEXT,
            volatility REAL,
            trend_strength REAL,

            -- Counterfactual (what would have happened)
            entry_price REAL,
            hypothetical_sl REAL,
            hypothetical_tp REAL,

            -- Outcome tracking
            tracked INTEGER DEFAULT 1,
            outcome TEXT,  -- 'would_win', 'would_lose', 'would_hold', NULL
            hypothetical_pnl REAL,
            rejection_correct INTEGER,  -- 1=correct, 0=wrong (missed opportunity)

            -- Theory scores (for debugging)
            theory_scores TEXT  -- JSON
        )
        """)

            # Create indexes
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rejection_symbol_timestamp
            ON signal_rejections(symbol, timestamp)
            """)

            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rejection_reason
            ON signal_rejections(rejection_reason)
            """)

            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rejection_outcome
            ON signal_rejections(outcome)
            """)

            conn.commit()
            conn.close()

            logger.debug("Rejection table verified/created")

        except Exception as e:
            logger.error(f"Failed to create rejection table: {e}")
            raise

    def log_rejection(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        rejection_reason: str,
        rejection_category: str,
        rejection_details: Dict[str, Any],
        market_context: Dict[str, Any],
        theory_scores: Dict[str, float],
        hypothetical_prices: Dict[str, float]
    ) -> int:
        """
        Log a rejected signal

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            direction: 'long' or 'short'
            confidence: Signal confidence (0.0-1.0)
            rejection_reason: Human-readable reason (e.g., 'Chop detected (ADX=15)')
            rejection_category: Category of rejection
                - 'regime' = Market regime unsuitable
                - 'correlation' = Too correlated with existing positions
                - 'rr_ratio' = Risk:reward ratio too low
                - 'drawdown' = Circuit breaker active
                - 'rate_limit' = Max signals per hour exceeded
                - 'confidence' = Below confidence threshold
            rejection_details: Full details dict (will be JSON-encoded)
            market_context: Market conditions at rejection
                {'regime': 'Ranging/Chop', 'volatility': 0.02, 'trend_strength': 15}
            theory_scores: All 11 theory scores (for debugging)
            hypothetical_prices: Entry, SL, TP if it had traded
                {'entry': 99000, 'sl': 98000, 'tp': 101000}

        Returns:
            rejection_id: ID of inserted rejection record
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
            INSERT INTO signal_rejections (
                symbol, direction, confidence,
                rejection_reason, rejection_category, rejection_details,
                regime, volatility, trend_strength,
                entry_price, hypothetical_sl, hypothetical_tp,
                theory_scores
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                direction,
                confidence,
                rejection_reason,
                rejection_category,
                json.dumps(rejection_details),
                market_context.get('regime'),
                market_context.get('volatility'),
                market_context.get('trend_strength'),
                hypothetical_prices.get('entry'),
                hypothetical_prices.get('sl'),
                hypothetical_prices.get('tp'),
                json.dumps(theory_scores)
            ))

            rejection_id = cursor.lastrowid
            conn.commit()
            conn.close()

            logger.info(
                f"Logged rejection #{rejection_id}: {symbol} {direction} "
                f"({rejection_category}) - {rejection_reason}"
            )

            return rejection_id

        except Exception as e:
            logger.error(f"Failed to log rejection: {e}")
            return -1

    def track_counterfactual(
        self,
        rejection_id: int,
        outcome: str,
        hypothetical_pnl: float
    ):
        """
        Update rejection with what actually happened (counterfactual)

        This tracks whether the rejection was correct or not by simulating
        what would have happened if we had taken the trade.

        Args:
            rejection_id: ID from log_rejection()
            outcome: Outcome of hypothetical trade
                - 'would_win': Hypothetical TP would have been hit
                - 'would_lose': Hypothetical SL would have been hit
                - 'would_hold': Neither TP nor SL hit within tracking period
            hypothetical_pnl: What P&L % would have been
                - Positive = would have won
                - Negative = would have lost
                - Zero = would still be holding

        Logic for determining if rejection was correct:
        - Rejection CORRECT if: outcome = 'would_lose' (saved from loss)
        - Rejection CORRECT if: outcome = 'would_hold' (no clear outcome)
        - Rejection WRONG if: outcome = 'would_win' (missed opportunity)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Determine if rejection was correct
            # Rejection is correct if: would_lose OR would_hold (no clear win)
            # Rejection is wrong if: would_win (missed opportunity)
            rejection_correct = 1 if outcome in ['would_lose', 'would_hold'] else 0

            cursor.execute("""
            UPDATE signal_rejections
            SET outcome = ?,
                hypothetical_pnl = ?,
                rejection_correct = ?
            WHERE id = ?
            """, (outcome, hypothetical_pnl, rejection_correct, rejection_id))

            conn.commit()
            conn.close()

            logger.debug(
                f"Updated rejection #{rejection_id}: {outcome} "
                f"({hypothetical_pnl:+.2f}%) - "
                f"{'Correct' if rejection_correct else 'Wrong (missed opportunity)'}"
            )

        except Exception as e:
            logger.error(f"Failed to track counterfactual for rejection #{rejection_id}: {e}")

    def analyze_rejections(
        self,
        time_period: str = '7d'
    ) -> Dict[str, Any]:
        """
        Analyze rejection patterns over time period

        Args:
            time_period: Time period for analysis (e.g., '7d', '30d', '1d')

        Returns:
            {
                'total_rejections': 45,
                'by_category': {
                    'regime': 20,
                    'correlation': 10,
                    'rr_ratio': 8,
                    'drawdown': 7
                },
                'rejection_accuracy': {
                    'regime': 0.85,  # 85% of regime rejections were correct
                    'correlation': 0.70,
                    'rr_ratio': 0.60,  # 60% correct (may be over-filtering)
                    'drawdown': 0.95
                },
                'missed_opportunities': [
                    {
                        'rejection_id': 123,
                        'symbol': 'BTC-USD',
                        'reason': 'R:R too low (3.2:1)',
                        'hypothetical_pnl': +1.5,  # Would have won +1.5%
                        'lesson': 'R:R threshold too strict for BTC'
                    },
                    ...
                ],
                'correct_rejections': [
                    {
                        'rejection_id': 125,
                        'symbol': 'ETH-USD',
                        'reason': 'Chop detected (ADX=15)',
                        'hypothetical_pnl': -0.8,  # Would have lost
                        'lesson': 'Regime filter saved from -0.8% loss'
                    },
                    ...
                ],
                'recommendations': [
                    'Lower R:R threshold for BTC (high accuracy asset)',
                    'Regime filter is working excellently (85% accuracy)',
                    'Consider relaxing correlation for opposite directions'
                ]
            }
        """
        try:
            conn = sqlite3.connect(self.db_path)

            # Get rejections with tracked outcomes
            df = pd.read_sql(f"""
            SELECT *
            FROM signal_rejections
            WHERE timestamp > datetime('now', '-{time_period}')
            AND outcome IS NOT NULL
            """, conn)

            conn.close()

            if df.empty:
                logger.info(f"No tracked rejections in last {time_period}")
                return {
                    'total_rejections': 0,
                    'by_category': {},
                    'rejection_accuracy': {},
                    'missed_opportunities': [],
                    'correct_rejections': [],
                    'recommendations': ['Insufficient data - need tracked rejections']
                }

            # Analyze by category
            by_category = df['rejection_category'].value_counts().to_dict()

            # Calculate accuracy by category
            rejection_accuracy = {}
            for category in by_category.keys():
                cat_df = df[df['rejection_category'] == category]
                accuracy = cat_df['rejection_correct'].mean()
                rejection_accuracy[category] = float(accuracy)

            # Find missed opportunities (rejection wrong, would have won)
            missed = df[
                (df['rejection_correct'] == 0) &
                (df['hypothetical_pnl'] > 0)
            ].sort_values('hypothetical_pnl', ascending=False)

            missed_opportunities = []
            for _, row in missed.head(5).iterrows():
                missed_opportunities.append({
                    'rejection_id': int(row['id']),
                    'symbol': row['symbol'],
                    'reason': row['rejection_reason'],
                    'hypothetical_pnl': float(row['hypothetical_pnl']),
                    'lesson': f'{row["rejection_category"]} may be over-filtering {row["symbol"]}'
                })

            # Find correct rejections (saved from losses)
            correct = df[
                (df['rejection_correct'] == 1) &
                (df['hypothetical_pnl'] < 0)
            ].sort_values('hypothetical_pnl')

            correct_rejections = []
            for _, row in correct.head(5).iterrows():
                correct_rejections.append({
                    'rejection_id': int(row['id']),
                    'symbol': row['symbol'],
                    'reason': row['rejection_reason'],
                    'hypothetical_pnl': float(row['hypothetical_pnl']),
                    'lesson': f'{row["rejection_category"]} saved from {row["hypothetical_pnl"]:.1%} loss'
                })

            # Generate recommendations
            recommendations = self._generate_recommendations(
                rejection_accuracy,
                missed_opportunities,
                correct_rejections
            )

            logger.info(
                f"Analyzed {len(df)} rejections | "
                f"Avg accuracy: {df['rejection_correct'].mean():.1%} | "
                f"Categories: {list(by_category.keys())}"
            )

            return {
                'total_rejections': len(df),
                'by_category': by_category,
                'rejection_accuracy': rejection_accuracy,
                'missed_opportunities': missed_opportunities,
                'correct_rejections': correct_rejections,
                'recommendations': recommendations
            }

        except Exception as e:
            logger.error(f"Failed to analyze rejections: {e}")
            return {
                'total_rejections': 0,
                'error': str(e)
            }

    def _generate_recommendations(
        self,
        accuracy: Dict[str, float],
        missed: List[Dict],
        correct: List[Dict]
    ) -> List[str]:
        """
        Generate tuning recommendations based on rejection analysis

        Args:
            accuracy: Rejection accuracy by category
            missed: List of missed opportunities
            correct: List of correct rejections

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Check each category accuracy
        for category, acc in accuracy.items():
            if acc < 0.6:
                # Over-filtering (less than 60% correct)
                recommendations.append(
                    f"⚠️  {category} filter may be too strict ({acc:.0%} accuracy). "
                    f"Consider relaxing threshold."
                )
            elif acc > 0.85:
                # Working well
                recommendations.append(
                    f"✅ {category} filter working excellently ({acc:.0%} accuracy)"
                )
            else:
                # Acceptable
                recommendations.append(
                    f"👍 {category} filter acceptable ({acc:.0%} accuracy)"
                )

        # Check for asset-specific patterns in missed opportunities
        if missed:
            common_symbols = {}
            for m in missed:
                symbol = m['symbol']
                common_symbols[symbol] = common_symbols.get(symbol, 0) + 1

            for symbol, count in common_symbols.items():
                if count >= 2:
                    recommendations.append(
                        f"💡 Consider relaxing filters for {symbol} "
                        f"({count} missed opportunities)"
                    )

        # Check for high-value missed opportunities
        if missed:
            high_value_missed = [m for m in missed if m['hypothetical_pnl'] > 2.0]
            if high_value_missed:
                recommendations.append(
                    f"🚨 {len(high_value_missed)} high-value opportunities missed (>2% potential profit)"
                )

        # Praise for high-value saves
        if correct:
            high_value_saves = [c for c in correct if c['hypothetical_pnl'] < -2.0]
            if high_value_saves:
                recommendations.append(
                    f"🛡️  Filters saved from {len(high_value_saves)} large losses (>2% potential loss)"
                )

        return recommendations

    def get_rejection_stats(self) -> Dict[str, Any]:
        """
        Get overall rejection statistics

        Returns:
            {
                'total_rejections_all_time': 150,
                'rejections_7d': 20,
                'rejections_30d': 75,
                'top_categories': {'regime': 50, 'correlation': 30, ...},
                'avg_accuracy': 0.78
            }
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Total rejections
            cursor.execute("SELECT COUNT(*) FROM signal_rejections")
            total_all = cursor.fetchone()[0]

            # Last 7 days
            cursor.execute("""
            SELECT COUNT(*)
            FROM signal_rejections
            WHERE timestamp > datetime('now', '-7 days')
            """)
            total_7d = cursor.fetchone()[0]

            # Last 30 days
            cursor.execute("""
            SELECT COUNT(*)
            FROM signal_rejections
            WHERE timestamp > datetime('now', '-30 days')
            """)
            total_30d = cursor.fetchone()[0]

            # Top categories (all time)
            df = pd.read_sql("""
            SELECT rejection_category, COUNT(*) as count
            FROM signal_rejections
            GROUP BY rejection_category
            ORDER BY count DESC
            LIMIT 5
            """, conn)

            top_categories = dict(zip(df['rejection_category'], df['count']))

            # Average accuracy (tracked rejections only)
            cursor.execute("""
            SELECT AVG(rejection_correct)
            FROM signal_rejections
            WHERE outcome IS NOT NULL
            """)
            avg_accuracy = cursor.fetchone()[0] or 0.0

            conn.close()

            return {
                'total_rejections_all_time': total_all,
                'rejections_7d': total_7d,
                'rejections_30d': total_30d,
                'top_categories': top_categories,
                'avg_accuracy': float(avg_accuracy)
            }

        except Exception as e:
            logger.error(f"Failed to get rejection stats: {e}")
            return {
                'total_rejections_all_time': 0,
                'error': str(e)
            }


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("REJECTION LOGGER TEST")
    print("=" * 70)

    # Initialize (use test database file)
    import tempfile
    import os
    test_db = tempfile.mktemp(suffix='.db')
    logger_instance = RejectionLogger(db_path=test_db)

    # Scenario 1: Log regime rejection
    print("\n[Scenario 1] Log regime rejection:")
    rejection_id_1 = logger_instance.log_rejection(
        symbol='BTC-USD',
        direction='long',
        confidence=0.75,
        rejection_reason='Chop detected (ADX=15)',
        rejection_category='regime',
        rejection_details={'adx': 15.2, 'threshold': 20, 'regime': 'Ranging/Chop'},
        market_context={'regime': 'Ranging/Chop', 'volatility': 0.02, 'trend_strength': 15.2},
        theory_scores={'shannon_entropy': 0.8, 'hurst': 0.45, 'markov': 0.6},
        hypothetical_prices={'entry': 99000, 'sl': 98000, 'tp': 101000}
    )
    print(f"  ✅ Logged rejection #{rejection_id_1}")

    # Track counterfactual (would have lost)
    logger_instance.track_counterfactual(
        rejection_id=rejection_id_1,
        outcome='would_lose',
        hypothetical_pnl=-1.0
    )
    print(f"  ✅ Tracked counterfactual: would_lose (-1.0%)")

    # Scenario 2: Log correlation rejection (missed opportunity)
    print("\n[Scenario 2] Log correlation rejection (missed opportunity):")
    rejection_id_2 = logger_instance.log_rejection(
        symbol='ETH-USD',
        direction='long',
        confidence=0.82,
        rejection_reason='High correlation with BTC-USD (0.95)',
        rejection_category='correlation',
        rejection_details={'correlation': 0.95, 'threshold': 0.70},
        market_context={'regime': 'Strong Trend', 'volatility': 0.025, 'trend_strength': 28.5},
        theory_scores={'shannon_entropy': 0.85, 'hurst': 0.7, 'markov': 0.8},
        hypothetical_prices={'entry': 3800, 'sl': 3750, 'tp': 3900}
    )
    print(f"  ✅ Logged rejection #{rejection_id_2}")

    # Track counterfactual (would have won - missed opportunity!)
    logger_instance.track_counterfactual(
        rejection_id=rejection_id_2,
        outcome='would_win',
        hypothetical_pnl=+2.6
    )
    print(f"  ✅ Tracked counterfactual: would_win (+2.6%) - MISSED OPPORTUNITY")

    # Scenario 3: Log R:R rejection
    print("\n[Scenario 3] Log R:R rejection:")
    rejection_id_3 = logger_instance.log_rejection(
        symbol='SOL-USD',
        direction='short',
        confidence=0.68,
        rejection_reason='R:R ratio too low (2.8:1)',
        rejection_category='rr_ratio',
        rejection_details={'rr_ratio': 2.8, 'threshold': 3.0},
        market_context={'regime': 'Weak Trend', 'volatility': 0.03, 'trend_strength': 22.1},
        theory_scores={'shannon_entropy': 0.7, 'hurst': 0.5, 'markov': 0.65},
        hypothetical_prices={'entry': 180, 'sl': 185, 'tp': 168}
    )
    print(f"  ✅ Logged rejection #{rejection_id_3}")

    # Track counterfactual (would have held - no clear outcome)
    logger_instance.track_counterfactual(
        rejection_id=rejection_id_3,
        outcome='would_hold',
        hypothetical_pnl=0.0
    )
    print(f"  ✅ Tracked counterfactual: would_hold (0.0%)")

    # Analyze rejections
    print("\n[Scenario 4] Analyze rejection patterns:")
    analysis = logger_instance.analyze_rejections(time_period='7d')

    print(f"  Total rejections: {analysis['total_rejections']}")

    if analysis['total_rejections'] > 0:
        print(f"  By category: {analysis['by_category']}")
        print(f"  Rejection accuracy:")
        for category, acc in analysis['rejection_accuracy'].items():
            print(f"    {category}: {acc:.0%}")

    print(f"\n  Missed opportunities ({len(analysis['missed_opportunities'])}):")
    for missed in analysis['missed_opportunities']:
        print(f"    #{missed['rejection_id']}: {missed['symbol']} - {missed['reason']}")
        print(f"      Would have gained: {missed['hypothetical_pnl']:+.1f}%")

    print(f"\n  Correct rejections ({len(analysis['correct_rejections'])}):")
    for correct in analysis['correct_rejections']:
        print(f"    #{correct['rejection_id']}: {correct['symbol']} - {correct['reason']}")
        print(f"      Saved from: {correct['hypothetical_pnl']:.1f}% loss")

    print(f"\n  Recommendations:")
    for rec in analysis['recommendations']:
        print(f"    {rec}")

    # Get overall stats
    print("\n[Scenario 5] Get rejection statistics:")
    stats = logger_instance.get_rejection_stats()
    print(f"  Total rejections (all time): {stats['total_rejections_all_time']}")
    print(f"  Rejections (7d): {stats['rejections_7d']}")
    print(f"  Average accuracy: {stats['avg_accuracy']:.0%}")
    print(f"  Top categories: {stats['top_categories']}")

    print("\n" + "=" * 70)
    print("✅ Rejection Logger ready for production!")
    print("=" * 70)

    # Cleanup test database
    if os.path.exists(test_db):
        os.remove(test_db)
        print(f"\n(Cleaned up test database: {test_db})")
