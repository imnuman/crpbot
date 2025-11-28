"""
HMAS V2 Orchestrator - $1.00/Signal Institutional System
Coordinates all 7 agents in hierarchical analysis flow

Signal Flow:
1. Alpha Generator V2 (DeepSeek $0.30) - Multi-timeframe pattern recognition
2. Technical Agent (DeepSeek $0.10) - Elliott Wave, Fibonacci, harmonics
3. Sentiment Agent (DeepSeek $0.08) - News, social, COT
4. Macro Agent (DeepSeek $0.07) - Economic calendar, correlations
5. Execution Auditor V2 (Grok $0.15) - Deep liquidity & slippage analysis
6. Rationale Agent V2 (Claude $0.20) - 5,000-word trade journal
7. Mother AI V2 (Gemini $0.10) - 3-round deliberation & final decision

Total Cost: $1.00/signal
Total Tokens: ~50,000 tokens
"""
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import os

from libs.hmas.agents.alpha_generator_v2 import AlphaGeneratorV2
from libs.hmas.agents.technical_agent import TechnicalAgent
from libs.hmas.agents.sentiment_agent import SentimentAgent
from libs.hmas.agents.macro_agent import MacroAgent
from libs.hmas.agents.execution_auditor_v2 import ExecutionAuditorV2
from libs.hmas.agents.rationale_agent_v2 import RationaleAgentV2
from libs.hmas.agents.mother_ai_v2 import MotherAIV2


class HMASV2Orchestrator:
    """
    HMAS V2 Orchestrator - Institutional-Grade 7-Agent System

    Budget: $1.00 per signal
    Win Rate Target: 80%+
    Strategy: Mean reversion + 200-MA trend filter
    """

    def __init__(
        self,
        deepseek_api_key: str,
        xai_api_key: str,
        anthropic_api_key: str,
        google_api_key: str
    ):
        """
        Initialize all 7 agents with API keys

        Args:
            deepseek_api_key: DeepSeek API key (4 agents: Alpha, Technical, Sentiment, Macro)
            xai_api_key: X.AI API key (Grok - Execution Auditor)
            anthropic_api_key: Anthropic API key (Claude - Rationale)
            google_api_key: Google API key (Gemini - Mother AI)
        """
        # Layer 2 - Specialist Agents (run in parallel)
        self.alpha_generator = AlphaGeneratorV2(api_key=deepseek_api_key)
        self.technical_agent = TechnicalAgent(api_key=deepseek_api_key)
        self.sentiment_agent = SentimentAgent(api_key=deepseek_api_key)
        self.macro_agent = MacroAgent(api_key=deepseek_api_key)

        # Layer 2 - Execution Validation (runs after specialists)
        self.execution_auditor = ExecutionAuditorV2(api_key=xai_api_key)

        # Layer 2 - Comprehensive Documentation (runs after execution audit)
        self.rationale_agent = RationaleAgentV2(api_key=anthropic_api_key)

        # Layer 1 - Final Decision Maker (runs last)
        self.mother_ai = MotherAIV2(api_key=google_api_key)

    async def generate_signal(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        account_balance: float = 10000.0
    ) -> Dict[str, Any]:
        """
        Generate complete trading signal using all 7 agents

        Args:
            symbol: Trading pair (e.g., 'GBPUSD', 'BTCUSD')
            market_data: Market data for analysis
                {
                    'ohlcv': [...],              # 200+ candles
                    'current_price': 1.25500,
                    'order_book': {...},         # Order book depth
                    'broker_spreads': {...},     # Multi-broker spreads
                    'volatility': {...},         # ATR, volatility state
                    'news_headlines': [...],     # Recent news
                    'economic_calendar': [...],  # Upcoming events
                    'correlations': {...},       # DXY, Gold, Oil, Bonds
                    'cot_data': {...},          # COT positioning
                    'social_mentions': {...}     # Twitter/Reddit sentiment
                }
            account_balance: Account size for position sizing (default $10,000 FTMO)

        Returns:
            Complete signal with all agent analyses and final decision
            {
                'decision': 'APPROVED' or 'REJECTED',
                'action': 'BUY_STOP' or 'SELL_STOP' or 'HOLD',
                'trade_parameters': {...},
                'agent_analyses': {...},
                'cost_breakdown': {...},
                'timestamp': '...'
            }
        """

        print(f"\n{'='*80}")
        print(f"HMAS V2 - Generating Signal for {symbol}")
        print(f"{'='*80}\n")

        start_time = datetime.now(timezone.utc)

        # ==========================================
        # LAYER 2A: SPECIALIST AGENTS (Run in Parallel)
        # ==========================================

        print("Layer 2A: Running 4 specialist agents in parallel...")
        print("  - Alpha Generator V2 (DeepSeek $0.30)")
        print("  - Technical Agent (DeepSeek $0.10)")
        print("  - Sentiment Agent (DeepSeek $0.08)")
        print("  - Macro Agent (DeepSeek $0.07)")
        print()

        # Run all 4 specialist agents concurrently
        alpha_task = self.alpha_generator.analyze(market_data)
        technical_task = self.technical_agent.analyze(market_data)
        sentiment_task = self.sentiment_agent.analyze(market_data)
        macro_task = self.macro_agent.analyze(market_data)

        # Wait for all to complete
        alpha_hypothesis, technical_analysis, sentiment_analysis, macro_analysis = \
            await asyncio.gather(alpha_task, technical_task, sentiment_task, macro_task)

        print(f"✓ Alpha Generator: {alpha_hypothesis.get('action', 'N/A')} "
              f"(confidence: {alpha_hypothesis.get('confidence', 0):.0%})")

        # Technical Agent - handle both success and error cases
        if 'error' in technical_analysis:
            tech_display = technical_analysis.get('evidence', 'Error in analysis...')
        else:
            tech_display = technical_analysis.get('elliott_wave', {}).get('primary_count', 'N/A')
        print(f"✓ Technical Agent: {str(tech_display)[:50]}...")

        print(f"✓ Sentiment Agent: {sentiment_analysis.get('overall_assessment', {}).get('sentiment_bias', 'N/A')}")
        print(f"✓ Macro Agent: {macro_analysis.get('overall_macro_assessment', {}).get('macro_bias', 'N/A')}")
        print()

        # ==========================================
        # LAYER 2B: EXECUTION AUDIT
        # ==========================================

        print("Layer 2B: Running execution audit...")
        print("  - Execution Auditor V2 (Grok $0.15)")
        print()

        # Prepare execution audit data
        execution_data = {
            'symbol': symbol,
            'trade_hypothesis': alpha_hypothesis,
            'order_book': market_data.get('order_book', {}),
            'broker_spreads': market_data.get('broker_spreads', {}),
            'market_depth': market_data.get('market_depth', {}),
            'volatility': market_data.get('volatility', {}),
            'session': market_data.get('session', 'unknown'),
            'position_size_usd': account_balance * 0.01  # 1% risk
        }

        execution_audit = await self.execution_auditor.analyze(execution_data)

        print(f"✓ Execution Auditor: {execution_audit.get('overall_audit', {}).get('audit_result', 'N/A')} "
              f"(grade: {execution_audit.get('overall_audit', {}).get('overall_grade', 'N/A')})")
        print()

        # ==========================================
        # LAYER 2C: COMPREHENSIVE RATIONALE
        # ==========================================

        print("Layer 2C: Generating comprehensive rationale...")
        print("  - Rationale Agent V2 (Claude $0.20)")
        print()

        # Prepare rationale data
        rationale_data = {
            'symbol': symbol,
            'alpha_hypothesis': alpha_hypothesis,
            'execution_audit': execution_audit,
            'technical_analysis': technical_analysis,
            'sentiment_analysis': sentiment_analysis,
            'macro_analysis': macro_analysis,
            'historical_performance': {
                # TODO: Pull from database
                'similar_setups': [],
                'win_rate': alpha_hypothesis.get('historical_win_rate', 0)
            },
            'market_context': {
                'regime': market_data.get('market_regime', 'unknown'),
                'volatility': market_data.get('volatility', {}).get('state', 'unknown'),
                'news_count': len(market_data.get('news_headlines', [])),
                'session': market_data.get('session', 'unknown')
            }
        }

        rationale_result = await self.rationale_agent.analyze(rationale_data)

        print(f"✓ Rationale Agent: {rationale_result.get('word_count', 0)} words generated")
        print()

        # ==========================================
        # LAYER 1: MOTHER AI - FINAL DECISION
        # ==========================================

        print("Layer 1: Mother AI - 3-round deliberation...")
        print("  - Mother AI V2 (Gemini $0.10)")
        print("    Round 1: Gather all outputs, detect conflicts")
        print("    Round 2: Resolve conflicts via scenario analysis")
        print("    Round 3: Final decision with lot sizing")
        print()

        # Prepare Mother AI data
        mother_ai_data = {
            'symbol': symbol,
            'alpha_hypothesis': alpha_hypothesis,
            'execution_audit': execution_audit,
            'rationale': rationale_result,
            'technical_analysis': technical_analysis,
            'sentiment_analysis': sentiment_analysis,
            'macro_analysis': macro_analysis,
            'account_balance': account_balance,
            'current_price': market_data.get('current_price', 0)
        }

        final_decision = await self.mother_ai.analyze(mother_ai_data)

        print(f"✓ Mother AI Decision: {final_decision.get('decision', 'N/A')}")
        if final_decision.get('decision') == 'APPROVED':
            print(f"  Action: {final_decision.get('action', 'N/A')}")
            print(f"  Entry: {final_decision.get('trade_parameters', {}).get('entry', 0):.5f}")
            print(f"  Lot Size: {final_decision.get('trade_parameters', {}).get('lot_size', 0):.2f}")
            print(f"  R:R Ratio: {final_decision.get('trade_parameters', {}).get('reward_risk_ratio', 0):.2f}:1")
        else:
            print(f"  Reason: {final_decision.get('rejection_reason', 'N/A')}")
        print()

        # ==========================================
        # PACKAGE COMPLETE SIGNAL
        # ==========================================

        end_time = datetime.now(timezone.utc)
        processing_time = (end_time - start_time).total_seconds()

        complete_signal = {
            # Final decision
            'decision': final_decision.get('decision', 'REJECTED'),
            'action': final_decision.get('action', 'HOLD'),
            'trade_parameters': final_decision.get('trade_parameters', {}),
            'ftmo_compliance': final_decision.get('ftmo_compliance', {}),

            # Agent analyses
            'agent_analyses': {
                'alpha_generator': alpha_hypothesis,
                'technical_analysis': technical_analysis,
                'sentiment_analysis': sentiment_analysis,
                'macro_analysis': macro_analysis,
                'execution_audit': execution_audit,
                'rationale': rationale_result,
                'mother_ai': final_decision
            },

            # Cost tracking
            'cost_breakdown': {
                'alpha_generator_v2': 0.30,
                'technical_agent': 0.10,
                'sentiment_agent': 0.08,
                'macro_agent': 0.07,
                'execution_auditor_v2': 0.15,
                'rationale_agent_v2': 0.20,
                'mother_ai_v2': 0.10,
                'total_cost': 1.00
            },

            # Metadata
            'symbol': symbol,
            'timestamp': end_time.isoformat(),
            'processing_time_seconds': processing_time,
            'hmas_version': 'V2',
            'system_cost': '$1.00/signal',
            'target_win_rate': '80%+'
        }

        print(f"{'='*80}")
        print(f"HMAS V2 Signal Complete")
        print(f"Processing Time: {processing_time:.1f} seconds")
        print(f"Total Cost: $1.00")
        print(f"Decision: {complete_signal['decision']}")
        print(f"{'='*80}\n")

        return complete_signal

    @classmethod
    def from_env(cls) -> 'HMASV2Orchestrator':
        """
        Create orchestrator from environment variables

        Required env vars:
        - DEEPSEEK_API_KEY
        - XAI_API_KEY
        - ANTHROPIC_API_KEY
        - GOOGLE_API_KEY
        """
        return cls(
            deepseek_api_key=os.getenv('DEEPSEEK_API_KEY'),
            xai_api_key=os.getenv('XAI_API_KEY'),
            anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
            google_api_key=os.getenv('GOOGLE_API_KEY')
        )

    def __repr__(self) -> str:
        return (
            "HMASV2Orchestrator(\n"
            "  Layer 2A - Specialists:\n"
            f"    - {self.alpha_generator}\n"
            f"    - {self.technical_agent}\n"
            f"    - {self.sentiment_agent}\n"
            f"    - {self.macro_agent}\n"
            "  Layer 2B - Execution:\n"
            f"    - {self.execution_auditor}\n"
            "  Layer 2C - Documentation:\n"
            f"    - {self.rationale_agent}\n"
            "  Layer 1 - Decision:\n"
            f"    - {self.mother_ai}\n"
            "  Total Cost: $1.00/signal\n"
            ")"
        )
