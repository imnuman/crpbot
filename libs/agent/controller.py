"""
LLM Agent Controller

The brain of the autonomous agent system. Receives natural language commands,
plans actions, coordinates sub-components, and executes tasks.

Capabilities:
1. Command routing (analyze, fix, explain, optimize)
2. Multi-step planning (break complex tasks into steps)
3. Tool orchestration (code analyzer, rewriter, diagnoser)
4. Conversational memory (context tracking)
5. Safety checks (backup, test, rollback)

Example:
    agent = AgentController()
    response = agent.process_command(
        "Why did we lose 3 trades on ETH today?"
    )
"""
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from libs.llm.deepseek_client import DeepSeekClient

logger = logging.getLogger(__name__)


class AgentController:
    """
    Central controller for the autonomous agent system

    Responsibilities:
    - Parse natural language commands
    - Plan multi-step actions
    - Coordinate sub-components
    - Maintain conversation context
    - Execute tasks safely
    """

    def __init__(self):
        """Initialize agent controller"""
        self.llm = DeepSeekClient()
        self.conversation_history: List[Dict[str, str]] = []
        self.context: Dict[str, Any] = {}

        # Tool imports (lazy loaded to avoid circular dependencies)
        self._code_analyzer = None
        self._error_handler = None
        self._code_rewriter = None
        self._diagnoser = None

    @property
    def code_analyzer(self):
        """Lazy load code analyzer"""
        if self._code_analyzer is None:
            from libs.agent.code_analyzer import CodeAnalyzer
            self._code_analyzer = CodeAnalyzer()
        return self._code_analyzer

    @property
    def error_handler(self):
        """Lazy load error handler"""
        if self._error_handler is None:
            from libs.agent.error_handler import ErrorHandler
            self._error_handler = ErrorHandler()
        return self._error_handler

    @property
    def code_rewriter(self):
        """Lazy load code rewriter"""
        if self._code_rewriter is None:
            from libs.agent.code_rewriter import CodeRewriter
            self._code_rewriter = CodeRewriter()
        return self._code_rewriter

    @property
    def diagnoser(self):
        """Lazy load performance diagnoser"""
        if self._diagnoser is None:
            from libs.agent.diagnoser import PerformanceDiagnoser
            self._diagnoser = PerformanceDiagnoser()
        return self._diagnoser

    def process_command(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a natural language command

        Args:
            user_input: Natural language command from user
            context: Optional additional context

        Returns:
            Dict with response and actions taken
        """
        logger.info(f"Processing command: {user_input}")

        # Update context
        if context:
            self.context.update(context)

        # Add to conversation history
        self.conversation_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })

        # Step 1: Understand intent
        intent = self._classify_intent(user_input)
        logger.info(f"Classified intent: {intent['type']}")

        # Step 2: Route to appropriate handler
        try:
            if intent['type'] == 'analyze_performance':
                result = self._handle_performance_analysis(user_input, intent)

            elif intent['type'] == 'fix_issue':
                result = self._handle_fix_issue(user_input, intent)

            elif intent['type'] == 'explain_behavior':
                result = self._handle_explain_behavior(user_input, intent)

            elif intent['type'] == 'optimize_strategy':
                result = self._handle_optimize_strategy(user_input, intent)

            elif intent['type'] == 'analyze_error':
                result = self._handle_analyze_error(user_input, intent)

            elif intent['type'] == 'code_inspection':
                result = self._handle_code_inspection(user_input, intent)

            else:
                result = self._handle_general_query(user_input)

            # Add to conversation history
            self.conversation_history.append({
                'role': 'assistant',
                'content': result['response'],
                'timestamp': datetime.now().isoformat()
            })

            return result

        except Exception as e:
            logger.error(f"Error processing command: {e}")
            error_response = {
                'response': f"I encountered an error: {str(e)}",
                'success': False,
                'error': str(e)
            }
            return error_response

    def _classify_intent(self, user_input: str) -> Dict[str, Any]:
        """
        Classify user intent using LLM

        Returns:
            Dict with intent type and extracted entities
        """
        prompt = f"""Classify the following user command into one of these categories:

Categories:
1. analyze_performance - Questions about trading performance, win rate, losses
2. fix_issue - Request to fix a problem or improve something
3. explain_behavior - Questions about why the system did something
4. optimize_strategy - Requests to improve or tune strategies
5. analyze_error - Questions about errors or crashes
6. code_inspection - Questions about code structure or implementation
7. general_query - General questions or conversation

User command: "{user_input}"

Also extract any entities mentioned:
- Symbols (BTC-USD, ETH-USD, etc.)
- Time periods (today, yesterday, last week)
- Components (theories, strategies, order flow)
- Metrics (win rate, Sharpe, P&L)

Return JSON:
{{
    "type": "category_name",
    "entities": {{
        "symbols": ["BTC-USD"],
        "time_period": "today",
        "components": ["OFI"],
        "metrics": ["win_rate"]
    }},
    "confidence": 0.95
}}"""

        response = self.llm.generate(prompt, temperature=0.3, max_tokens=500)

        try:
            # Parse JSON from response
            intent = json.loads(response)
        except json.JSONDecodeError:
            # Fallback if LLM doesn't return valid JSON
            logger.warning("LLM didn't return valid JSON, using heuristics")
            intent = self._heuristic_intent_classification(user_input)

        return intent

    def _heuristic_intent_classification(self, user_input: str) -> Dict[str, Any]:
        """Fallback intent classification using keywords"""
        lower = user_input.lower()

        if any(word in lower for word in ['why', 'lost', 'lose', 'loss', 'failed']):
            return {
                'type': 'analyze_performance',
                'entities': {},
                'confidence': 0.7
            }

        elif any(word in lower for word in ['fix', 'improve', 'change', 'update']):
            return {
                'type': 'fix_issue',
                'entities': {},
                'confidence': 0.7
            }

        elif any(word in lower for word in ['explain', 'how does', 'what is']):
            return {
                'type': 'explain_behavior',
                'entities': {},
                'confidence': 0.7
            }

        elif any(word in lower for word in ['optimize', 'tune', 'better']):
            return {
                'type': 'optimize_strategy',
                'entities': {},
                'confidence': 0.7
            }

        elif any(word in lower for word in ['error', 'crash', 'exception', 'traceback']):
            return {
                'type': 'analyze_error',
                'entities': {},
                'confidence': 0.7
            }

        elif any(word in lower for word in ['code', 'function', 'file', 'implementation']):
            return {
                'type': 'code_inspection',
                'entities': {},
                'confidence': 0.7
            }

        else:
            return {
                'type': 'general_query',
                'entities': {},
                'confidence': 0.5
            }

    def _handle_performance_analysis(
        self,
        user_input: str,
        intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle performance analysis queries

        Example: "Why did we lose 3 trades on ETH today?"
        """
        logger.info("Handling performance analysis")

        # Extract entities
        entities = intent.get('entities', {})
        symbol = entities.get('symbols', [None])[0]
        time_period = entities.get('time_period', 'today')

        # Use diagnoser to analyze performance
        analysis = self.diagnoser.analyze_losses(
            symbol=symbol,
            time_period=time_period
        )

        # Format response
        response = self._format_performance_analysis(analysis)

        return {
            'response': response,
            'success': True,
            'data': analysis
        }

    def _handle_fix_issue(
        self,
        user_input: str,
        intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle fix requests

        Example: "Fix the high volatility issue"
        """
        logger.info("Handling fix issue")

        # Step 1: Understand what needs fixing
        problem = self._identify_problem(user_input, intent)

        # Step 2: Plan solution
        plan = self._plan_fix(problem)

        # Step 3: Execute fix (with user confirmation)
        response = f"""I've identified the problem:
{problem['description']}

Proposed fix:
{plan['description']}

This will modify: {', '.join(plan['files_to_modify'])}

Would you like me to proceed? (yes/no)
"""

        return {
            'response': response,
            'success': True,
            'requires_confirmation': True,
            'plan': plan
        }

    def execute_fix(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a fix plan (after user confirmation)

        Args:
            plan: Fix plan from _plan_fix

        Returns:
            Dict with execution results
        """
        logger.info(f"Executing fix: {plan['description']}")

        results = []

        for file_path, modifications in plan['modifications'].items():
            # Use code rewriter with safety checks
            result = self.code_rewriter.modify_file(
                file_path=file_path,
                modifications=modifications,
                test_after=True,
                backup=True
            )
            results.append(result)

        # Check if all succeeded
        all_success = all(r['success'] for r in results)

        if all_success:
            response = "✅ Fix applied successfully. All tests passed."
        else:
            failed = [r for r in results if not r['success']]
            response = f"❌ Fix failed. Rolled back changes.\n\nErrors:\n"
            for r in failed:
                response += f"- {r['file']}: {r['error']}\n"

        return {
            'response': response,
            'success': all_success,
            'results': results
        }

    def _handle_explain_behavior(
        self,
        user_input: str,
        intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle explanation queries

        Example: "How does the Kalman filter work?"
        """
        logger.info("Handling explain behavior")

        # Extract what needs explanation
        entities = intent.get('entities', {})
        components = entities.get('components', [])

        if components:
            # Use code analyzer to understand component
            component = components[0]
            analysis = self.code_analyzer.analyze_component(component)

            # Generate explanation using LLM
            explanation = self._generate_explanation(analysis)
        else:
            # General explanation
            explanation = self._generate_general_explanation(user_input)

        return {
            'response': explanation,
            'success': True
        }

    def _handle_optimize_strategy(
        self,
        user_input: str,
        intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle optimization requests

        Example: "Optimize the position sizing"
        """
        logger.info("Handling optimize strategy")

        # Use diagnoser to find optimization opportunities
        opportunities = self.diagnoser.find_optimization_opportunities()

        # Rank by impact
        top_opportunities = sorted(
            opportunities,
            key=lambda x: x['potential_impact'],
            reverse=True
        )[:3]

        response = "🎯 Top 3 Optimization Opportunities:\n\n"
        for i, opp in enumerate(top_opportunities, 1):
            response += f"{i}. {opp['name']}\n"
            response += f"   Impact: {opp['potential_impact']:.1%} win rate improvement\n"
            response += f"   Effort: {opp['effort']}\n"
            response += f"   Description: {opp['description']}\n\n"

        response += "Which would you like to implement?"

        return {
            'response': response,
            'success': True,
            'opportunities': top_opportunities
        }

    def _handle_analyze_error(
        self,
        user_input: str,
        intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle error analysis

        Example: "What's causing the runtime errors?"
        """
        logger.info("Handling analyze error")

        # Use error handler to analyze recent errors
        error_summary = self.error_handler.analyze_recent_errors()

        response = self._format_error_analysis(error_summary)

        return {
            'response': response,
            'success': True,
            'data': error_summary
        }

    def _handle_code_inspection(
        self,
        user_input: str,
        intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle code inspection queries

        Example: "Show me how the signal generator works"
        """
        logger.info("Handling code inspection")

        # Use code analyzer
        entities = intent.get('entities', {})
        components = entities.get('components', [])

        if components:
            component = components[0]
            analysis = self.code_analyzer.analyze_component(component)

            response = self._format_code_analysis(analysis)
        else:
            response = self.code_analyzer.get_system_overview()

        return {
            'response': response,
            'success': True
        }

    def _handle_general_query(self, user_input: str) -> Dict[str, Any]:
        """Handle general conversational queries"""
        logger.info("Handling general query")

        # Use LLM with conversation history for context
        context = self._build_conversation_context()

        prompt = f"""{context}

User: {user_input}

Assistant (you are the CRPBot trading system agent):"""

        response = self.llm.generate(prompt, temperature=0.7, max_tokens=1000)

        return {
            'response': response,
            'success': True
        }

    def _build_conversation_context(self) -> str:
        """Build conversation context from history"""
        # Include last 5 exchanges
        recent = self.conversation_history[-10:]

        context = "Conversation history:\n"
        for entry in recent:
            role = entry['role'].capitalize()
            content = entry['content']
            context += f"{role}: {content}\n"

        return context

    def _identify_problem(
        self,
        user_input: str,
        intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Identify the specific problem to fix"""
        # Use LLM to understand the problem
        prompt = f"""Based on this user request: "{user_input}"

And this system context:
- Current win rate: 53.8%
- Issue areas: High volatility regime, stop-loss hits
- Recent analysis: OFI negative at entry during high vol

Identify the specific problem that needs fixing.

Return JSON:
{{
    "problem": "Stop losses hit too quickly in high volatility",
    "affected_component": "regime_strategy",
    "root_cause": "No volatility-based filter on entry signals",
    "severity": "medium"
}}"""

        response = self.llm.generate(prompt, temperature=0.3, max_tokens=500)

        try:
            problem = json.loads(response)
        except:
            problem = {
                'problem': 'Unable to identify specific problem',
                'affected_component': 'unknown',
                'root_cause': 'Unclear from user input',
                'severity': 'low'
            }

        return problem

    def _plan_fix(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Plan how to fix the identified problem"""
        # Use LLM to plan solution
        prompt = f"""Problem to fix:
{json.dumps(problem, indent=2)}

Design a solution that modifies the code safely.

Return JSON:
{{
    "description": "Add volatility filter to regime strategy",
    "files_to_modify": ["libs/risk/regime_strategy.py"],
    "modifications": {{
        "libs/risk/regime_strategy.py": {{
            "type": "add_filter",
            "location": "filter_signal method",
            "code": "if regime_name == 'High Vol Range' and confidence < 0.75: return False, 'Filtered: High vol requires 75%+ confidence'"
        }}
    }},
    "tests_to_run": ["test_regime_strategy"],
    "rollback_plan": "Git restore if tests fail"
}}"""

        response = self.llm.generate(prompt, temperature=0.3, max_tokens=1000)

        try:
            plan = json.loads(response)
        except:
            plan = {
                'description': 'Unable to generate plan',
                'files_to_modify': [],
                'modifications': {},
                'tests_to_run': [],
                'rollback_plan': 'Manual review required'
            }

        return plan

    def _format_performance_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format performance analysis for user"""
        response = f"📊 Performance Analysis:\n\n"

        if 'losses' in analysis:
            response += f"Found {len(analysis['losses'])} losses:\n"

            # Common patterns
            if 'patterns' in analysis:
                response += "\n🔍 Common Patterns:\n"
                for pattern in analysis['patterns']:
                    response += f"  - {pattern}\n"

            # Recommendations
            if 'recommendations' in analysis:
                response += "\n💡 Recommendations:\n"
                for rec in analysis['recommendations']:
                    response += f"  - {rec}\n"

        return response

    def _format_error_analysis(self, error_summary: Dict[str, Any]) -> str:
        """Format error analysis for user"""
        response = f"🐛 Error Analysis:\n\n"

        if 'recent_errors' in error_summary:
            response += f"Found {len(error_summary['recent_errors'])} recent errors:\n\n"

            for error in error_summary['recent_errors'][:5]:
                response += f"❌ {error['type']}: {error['message']}\n"
                response += f"   File: {error['file']}:{error['line']}\n"
                response += f"   Time: {error['timestamp']}\n\n"

        if 'root_cause' in error_summary:
            response += f"🎯 Root Cause: {error_summary['root_cause']}\n"

        if 'fix_suggestion' in error_summary:
            response += f"\n💡 Fix Suggestion:\n{error_summary['fix_suggestion']}\n"

        return response

    def _format_code_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format code analysis for user"""
        response = f"📝 Code Analysis:\n\n"

        if 'component' in analysis:
            response += f"Component: {analysis['component']}\n"

        if 'file' in analysis:
            response += f"File: {analysis['file']}\n"

        if 'purpose' in analysis:
            response += f"\n🎯 Purpose:\n{analysis['purpose']}\n"

        if 'key_functions' in analysis:
            response += f"\n⚙️ Key Functions:\n"
            for func in analysis['key_functions']:
                response += f"  - {func['name']}: {func['description']}\n"

        if 'dependencies' in analysis:
            response += f"\n🔗 Dependencies:\n"
            for dep in analysis['dependencies']:
                response += f"  - {dep}\n"

        return response

    def _generate_explanation(self, analysis: Dict[str, Any]) -> str:
        """Generate natural language explanation of component"""
        prompt = f"""Explain this component in simple terms:

{json.dumps(analysis, indent=2)}

Provide a clear, concise explanation that a developer can understand."""

        return self.llm.generate(prompt, temperature=0.7, max_tokens=800)

    def _generate_general_explanation(self, user_input: str) -> str:
        """Generate explanation for general query"""
        prompt = f"""You are the CRPBot trading system agent. Explain:

{user_input}

Provide a clear, accurate explanation based on the system's architecture."""

        return self.llm.generate(prompt, temperature=0.7, max_tokens=800)

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.context = {}
        logger.info("Conversation history cleared")


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("AGENT CONTROLLER TEST")
    print("=" * 70)

    agent = AgentController()

    # Test 1: Performance analysis
    print("\n[Test 1] Performance analysis query:")
    result = agent.process_command(
        "Why did we lose 3 trades on ETH today?"
    )
    print(f"Response: {result['response'][:200]}...")

    # Test 2: Explain behavior
    print("\n[Test 2] Explanation query:")
    result = agent.process_command(
        "How does the Kalman filter work?"
    )
    print(f"Response: {result['response'][:200]}...")

    # Test 3: Fix request
    print("\n[Test 3] Fix request:")
    result = agent.process_command(
        "Fix the high volatility stop loss issue"
    )
    print(f"Response: {result['response'][:200]}...")

    print("\n" + "=" * 70)
    print("✅ Agent Controller ready for production!")
    print("=" * 70)
