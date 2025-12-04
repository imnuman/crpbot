"""
Trade Orchestrator - 4-Step HMAS Workflow

Coordinates all 4 agents to generate high-probability trading signals
with 80%+ win rate target and 1.0% risk per trade (FTMO compliant).
"""
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from libs.hmas.config.hmas_config import get_config
from libs.hmas.agents.alpha_generator import AlphaGenerator
from libs.hmas.agents.execution_auditor import ExecutionAuditor
from libs.hmas.agents.rationale_agent import RationaleAgent
from libs.hmas.agents.mother_ai import MotherAI
from libs.hmas.orchestrator.ftmo_calculator import FTMOCalculator


class TradeOrchestrator:
    """
    HMAS Trade Signal Orchestrator

    4-Step Workflow:
    1. Alpha Generation (DeepSeek) - Pattern recognition
    2. Rationale Generation (Claude) - Explanation & confidence
    3. Execution Audit (Grok) - Cost validation & ALM setup
    4. Final Decision (Gemini) - Orchestration & risk governance
    """

    def __init__(self):
        """Initialize orchestrator with all 4 agents"""
        # Load configuration
        self.config = get_config()

        # Initialize all 4 agents
        self.alpha_generator = AlphaGenerator(api_key=self.config.deepseek_api_key)
        self.execution_auditor = ExecutionAuditor(api_key=self.config.xai_api_key)
        self.rationale_agent = RationaleAgent(api_key=self.config.anthropic_api_key)
        self.mother_ai = MotherAI(api_key=self.config.google_api_key)

        # Initialize FTMO calculator
        self.ftmo_calculator = FTMOCalculator(
            account_balance=self.config.account_balance,
            daily_loss_limit=self.config.ftmo_daily_loss_limit,
            max_loss_limit=self.config.ftmo_max_loss_limit
        )

    async def generate_signal(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        spread_pips: float = 1.5,
        fees_pips: float = 0.5
    ) -> Dict[str, Any]:
        """
        Generate trading signal using 4-agent HMAS workflow

        Args:
            symbol: Trading symbol (e.g., 'GBPUSD')
            market_data: Market data and indicators
                {
                    'current_price': float,
                    'ma200': float,
                    'rsi': float,
                    'bbands_upper': float,
                    'bbands_lower': float,
                    'atr': float
                }
            spread_pips: Current spread in pips
            fees_pips: Trading fees in pips

        Returns:
            Final trading signal with full agent outputs
        """
        start_time = datetime.now(timezone.utc)

        try:
            # ==========================================
            # STEP 1: Alpha Generation (DeepSeek)
            # ==========================================
            print(f"\n{'='*80}")
            print(f"STEP 1: Alpha Generation (DeepSeek)")
            print(f"{'='*80}")

            alpha_data = {
                'symbol': symbol,
                'timeframe': self.config.timeframe_primary,
                **market_data
            }

            alpha_hypothesis = await self.alpha_generator.analyze(alpha_data)

            print(f"✅ Alpha Hypothesis: {alpha_hypothesis.get('action', 'HOLD')}")
            print(f"   Confidence: {alpha_hypothesis.get('confidence', 0):.0%}")

            # If HOLD, skip remaining steps
            if alpha_hypothesis.get('action') == 'HOLD':
                return self._build_hold_signal(
                    symbol=symbol,
                    reason="No valid setup detected by Alpha Generator",
                    alpha_hypothesis=alpha_hypothesis,
                    duration=(datetime.now(timezone.utc) - start_time).total_seconds()
                )

            # ==========================================
            # STEP 2: Rationale Generation (Claude)
            # ==========================================
            print(f"\n{'='*80}")
            print(f"STEP 2: Rationale Generation (Claude)")
            print(f"{'='*80}")

            rationale_data = {
                'symbol': symbol,
                'alpha_hypothesis': alpha_hypothesis,
                'execution_audit': {},  # Will be filled in next step
                'historical_performance': {}
            }

            rationale_result = await self.rationale_agent.analyze(rationale_data)
            rationale = rationale_result.get('rationale', '')

            print(f"✅ Rationale generated ({len(rationale)} chars)")

            # ==========================================
            # STEP 3: Execution Audit (Grok)
            # ==========================================
            print(f"\n{'='*80}")
            print(f"STEP 3: Execution Audit (Grok)")
            print(f"{'='*80}")

            audit_data = {
                'trade_hypothesis': alpha_hypothesis,
                'spread_pips': spread_pips,
                'fees_pips': fees_pips,
                'atr': market_data.get('atr', 0.001)
            }

            execution_audit = await self.execution_auditor.analyze(audit_data)

            print(f"✅ Audit Result: {execution_audit.get('audit_result', 'UNKNOWN')}")
            print(f"   Recommendation: {execution_audit.get('recommendation', 'N/A')}")

            # If audit fails, reject trade
            if execution_audit.get('audit_result') == 'FAIL':
                return self._build_rejected_signal(
                    symbol=symbol,
                    reason=f"Execution audit failed: {execution_audit.get('recommendation', 'Unknown')}",
                    alpha_hypothesis=alpha_hypothesis,
                    execution_audit=execution_audit,
                    rationale=rationale,
                    duration=(datetime.now(timezone.utc) - start_time).total_seconds()
                )

            # ==========================================
            # STEP 4: Final Decision (Gemini Mother AI)
            # ==========================================
            print(f"\n{'='*80}")
            print(f"STEP 4: Final Decision (Gemini Mother AI)")
            print(f"{'='*80}")

            mother_ai_data = {
                'symbol': symbol,
                'alpha_hypothesis': alpha_hypothesis,
                'execution_audit': execution_audit,
                'rationale': rationale,
                'account_balance': self.config.account_balance,
                'current_price': market_data.get('current_price', 0)
            }

            final_decision = await self.mother_ai.analyze(mother_ai_data)

            print(f"✅ Final Decision: {final_decision.get('decision', 'UNKNOWN')}")
            print(f"   Action: {final_decision.get('action', 'N/A')}")

            # ==========================================
            # Final Signal Assembly
            # ==========================================
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()

            signal = {
                # Final decision
                **final_decision,

                # Agent outputs
                'alpha_hypothesis': alpha_hypothesis,
                'execution_audit': execution_audit,
                'rationale': rationale,

                # Metadata
                'symbol': symbol,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'generation_time_seconds': duration,
                'agents_used': ['DeepSeek', 'Claude', 'Grok', 'Gemini'],

                # Configuration
                'account_balance': self.config.account_balance,
                'risk_per_trade': self.config.risk_per_trade,
                'target_win_rate': self.config.target_win_rate
            }

            print(f"\n{'='*80}")
            print(f"SIGNAL GENERATED: {signal.get('action', 'N/A')} @ {final_decision.get('entry', 0):.5f}")
            print(f"Lot Size: {final_decision.get('lot_size', 0)} | Confidence: {final_decision.get('confidence', 0):.0%}")
            print(f"Generation Time: {duration:.2f}s")
            print(f"{'='*80}\n")

            return signal

        except Exception as e:
            # Return error signal
            return {
                'decision': 'REJECTED',
                'action': 'HOLD',
                'symbol': symbol,
                'error': str(e),
                'rejection_reason': f'System error: {str(e)}',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'generation_time_seconds': (datetime.now(timezone.utc) - start_time).total_seconds()
            }

    def _build_hold_signal(
        self,
        symbol: str,
        reason: str,
        alpha_hypothesis: Dict[str, Any],
        duration: float
    ) -> Dict[str, Any]:
        """Build HOLD signal"""
        return {
            'decision': 'REJECTED',
            'action': 'HOLD',
            'symbol': symbol,
            'rejection_reason': reason,
            'alpha_hypothesis': alpha_hypothesis,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'generation_time_seconds': duration
        }

    def _build_rejected_signal(
        self,
        symbol: str,
        reason: str,
        alpha_hypothesis: Dict[str, Any],
        execution_audit: Dict[str, Any],
        rationale: str,
        duration: float
    ) -> Dict[str, Any]:
        """Build rejected signal"""
        return {
            'decision': 'REJECTED',
            'action': 'HOLD',
            'symbol': symbol,
            'rejection_reason': reason,
            'alpha_hypothesis': alpha_hypothesis,
            'execution_audit': execution_audit,
            'rationale': rationale,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'generation_time_seconds': duration
        }

    def __repr__(self) -> str:
        return (
            f"TradeOrchestrator(\n"
            f"  Agents: [AlphaGenerator, ExecutionAuditor, RationaleAgent, MotherAI],\n"
            f"  Account: ${self.config.account_balance:,.2f},\n"
            f"  Risk/Trade: {self.config.risk_per_trade:.1%},\n"
            f"  Target WR: {self.config.target_win_rate:.0%}\n"
            f")"
        )
