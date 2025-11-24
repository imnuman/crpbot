"""
Code Analyzer

Reads and understands code structure, dependencies, and functionality.
Provides semantic understanding of the codebase for the agent.

Capabilities:
1. Parse Python files (AST analysis)
2. Extract functions, classes, imports
3. Understand component relationships
4. Generate code summaries
5. Map system architecture

Example:
    analyzer = CodeAnalyzer()
    analysis = analyzer.analyze_component("Kalman Filter")
    # Returns: Purpose, key functions, dependencies, usage
"""
import ast
import os
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CodeAnalyzer:
    """
    Analyzes code structure and relationships

    Uses AST (Abstract Syntax Tree) parsing to understand code
    without executing it.
    """

    def __init__(self, project_root: str = "/root/crpbot"):
        """
        Initialize code analyzer

        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.component_map = self._build_component_map()

    def _build_component_map(self) -> Dict[str, str]:
        """
        Build map of component names to file paths

        Returns:
            Dict mapping component names to file paths
        """
        component_map = {}

        # Core components
        components = {
            # Theories
            'Shannon Entropy': 'libs/analysis/shannon_entropy.py',
            'Hurst Exponent': 'libs/analysis/hurst_exponent.py',
            'Markov Chain': 'libs/analysis/markov_chain.py',
            'Kalman Filter': 'libs/analysis/kalman_filter.py',
            'Bayesian Inference': 'libs/analysis/bayesian_inference.py',
            'Monte Carlo': 'libs/analysis/monte_carlo.py',
            'Random Forest': 'libs/theories/random_forest_validator.py',
            'Autocorrelation': 'libs/theories/autocorrelation_analyzer.py',
            'Stationarity': 'libs/theories/stationarity_test.py',
            'Variance': 'libs/theories/variance_tests.py',
            'Market Context': 'libs/theories/market_context.py',

            # Order Flow
            'Order Flow Imbalance': 'libs/order_flow/order_flow_imbalance.py',
            'OFI': 'libs/order_flow/order_flow_imbalance.py',
            'Volume Profile': 'libs/order_flow/volume_profile.py',
            'Market Microstructure': 'libs/order_flow/market_microstructure.py',
            'Order Flow Integration': 'libs/order_flow/order_flow_integration.py',

            # Risk Management
            'Kelly Criterion': 'libs/risk/kelly_criterion.py',
            'Exit Strategy': 'libs/risk/exit_strategy.py',
            'Correlation': 'libs/risk/correlation_analyzer.py',
            'Regime Strategy': 'libs/risk/regime_strategy.py',

            # LLM
            'Signal Generator': 'libs/llm/signal_generator.py',
            'DeepSeek': 'libs/llm/deepseek_client.py',
            'Signal Synthesizer': 'libs/llm/signal_synthesizer.py',
            'Signal Parser': 'libs/llm/signal_parser.py',

            # Runtime
            'V7 Runtime': 'apps/runtime/v7_runtime.py',
            'FTMO Rules': 'apps/runtime/ftmo_rules.py',

            # Tracking
            'Performance Tracker': 'libs/tracking/performance_tracker.py',
            'Paper Trader': 'libs/tracking/paper_trader.py',
        }

        component_map.update(components)
        return component_map

    def analyze_component(self, component_name: str) -> Dict[str, Any]:
        """
        Analyze a specific component

        Args:
            component_name: Name of component (e.g., "Kalman Filter")

        Returns:
            Dict with component analysis
        """
        # Find file path
        file_path = self.component_map.get(component_name)

        if not file_path:
            # Try fuzzy match
            file_path = self._fuzzy_find_component(component_name)

        if not file_path:
            return {
                'error': f'Component "{component_name}" not found',
                'available': list(self.component_map.keys())
            }

        full_path = self.project_root / file_path

        if not full_path.exists():
            return {
                'error': f'File not found: {file_path}',
                'component': component_name
            }

        # Parse and analyze file
        analysis = self._analyze_file(full_path)
        analysis['component'] = component_name
        analysis['file'] = str(file_path)

        return analysis

    def _fuzzy_find_component(self, query: str) -> Optional[str]:
        """Fuzzy match component name"""
        query_lower = query.lower()

        for name, path in self.component_map.items():
            if query_lower in name.lower():
                return path

        return None

    def _analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Analyze a Python file

        Args:
            file_path: Path to Python file

        Returns:
            Dict with file analysis
        """
        try:
            with open(file_path, 'r') as f:
                source_code = f.read()

            # Parse AST
            tree = ast.parse(source_code)

            analysis = {
                'purpose': self._extract_purpose(source_code),
                'classes': self._extract_classes(tree),
                'functions': self._extract_functions(tree),
                'imports': self._extract_imports(tree),
                'key_functions': self._identify_key_functions(tree, source_code),
                'dependencies': self._extract_dependencies(tree),
                'docstring': ast.get_docstring(tree),
                'line_count': len(source_code.split('\n'))
            }

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return {
                'error': str(e),
                'file': str(file_path)
            }

    def _extract_purpose(self, source_code: str) -> str:
        """Extract purpose from module docstring"""
        # Get first multiline string (module docstring)
        match = re.search(r'"""(.+?)"""', source_code, re.DOTALL)
        if match:
            docstring = match.group(1).strip()
            # Get first paragraph
            first_para = docstring.split('\n\n')[0]
            return first_para
        return "No purpose documented"

    def _extract_classes(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract class definitions"""
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append({
                    'name': node.name,
                    'docstring': ast.get_docstring(node),
                    'methods': [
                        m.name for m in node.body
                        if isinstance(m, ast.FunctionDef)
                    ],
                    'line': node.lineno
                })

        return classes

    def _extract_functions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract function definitions (module-level only)"""
        functions = []

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    'name': node.name,
                    'args': [arg.arg for arg in node.args.args],
                    'docstring': ast.get_docstring(node),
                    'line': node.lineno
                })

        return functions

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements"""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")

        return imports

    def _identify_key_functions(
        self,
        tree: ast.AST,
        source_code: str
    ) -> List[Dict[str, str]]:
        """
        Identify the most important functions

        Key functions are:
        - Public methods (not starting with _)
        - Have docstrings
        - >10 lines of code
        """
        key_functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip private functions
                if node.name.startswith('_') and not node.name.startswith('__'):
                    continue

                # Get docstring
                docstring = ast.get_docstring(node)
                if not docstring:
                    continue

                # Estimate complexity (number of statements)
                statements = len([
                    n for n in ast.walk(node)
                    if isinstance(n, ast.stmt)
                ])

                if statements >= 5:  # Non-trivial function
                    # Extract first line of docstring as description
                    description = docstring.split('\n')[0].strip()

                    key_functions.append({
                        'name': node.name,
                        'description': description,
                        'line': node.lineno
                    })

        return key_functions

    def _extract_dependencies(self, tree: ast.AST) -> List[str]:
        """Extract internal dependencies (libs/* imports)"""
        dependencies = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith('libs'):
                    dependencies.append(node.module)

        return list(set(dependencies))

    def get_system_overview(self) -> str:
        """Get high-level system overview"""
        overview = """
📊 CRPBot V7 Ultimate - System Overview

Architecture (5 Layers):

1. Data Layer
   ├── Coinbase API (OHLCV data)
   ├── Coinbase WebSocket (Level 2 order book)
   ├── CoinGecko API (market context)
   └── Database (SQLite)

2. Analysis Layer
   ├── 11 Theories (Shannon, Hurst, Markov, Kalman, etc.)
   ├── Order Flow Analysis (OFI, Volume Profile, Microstructure)
   └── LLM Synthesis (DeepSeek)

3. Risk Management Layer
   ├── Kelly Criterion (position sizing)
   ├── Exit Strategy (trailing stops)
   ├── Correlation Filter (diversification)
   └── Regime Strategy (market-aware)

4. Execution Layer
   ├── Paper Trading (simulation)
   ├── Performance Tracking
   └── Telegram Notifications

5. Agent Layer (You are here!)
   ├── Command routing
   ├── Code analysis
   ├── Error handling
   ├── Code rewriting
   └── Performance diagnosis

Key Components:
- Signal Generator: Orchestrates all theories → LLM → signal
- V7 Runtime: Main loop (scans every 5 min)
- Performance Tracker: Monitors win rate, Sharpe, P&L

Current Status:
- Win rate: 53.8%
- Sharpe: ~1.0-1.2
- Paper trades: 13+
- Target: 60-65% win rate
"""
        return overview.strip()

    def find_files_by_pattern(self, pattern: str) -> List[str]:
        """
        Find files matching a pattern

        Args:
            pattern: Glob pattern (e.g., "*.py", "libs/analysis/*.py")

        Returns:
            List of matching file paths
        """
        matches = list(self.project_root.glob(pattern))
        return [str(p.relative_to(self.project_root)) for p in matches]

    def get_file_summary(self, file_path: str) -> str:
        """
        Get quick summary of a file

        Args:
            file_path: Relative path to file

        Returns:
            Summary string
        """
        full_path = self.project_root / file_path

        if not full_path.exists():
            return f"File not found: {file_path}"

        analysis = self._analyze_file(full_path)

        summary = f"📄 {file_path}\n"
        summary += f"Lines: {analysis.get('line_count', 'Unknown')}\n"

        if analysis.get('purpose'):
            summary += f"\n🎯 Purpose:\n{analysis['purpose']}\n"

        if analysis.get('classes'):
            summary += f"\n📦 Classes: {len(analysis['classes'])}\n"
            for cls in analysis['classes'][:3]:
                summary += f"  - {cls['name']}: {len(cls['methods'])} methods\n"

        if analysis.get('key_functions'):
            summary += f"\n⚙️ Key Functions: {len(analysis['key_functions'])}\n"
            for func in analysis['key_functions'][:5]:
                summary += f"  - {func['name']}: {func['description']}\n"

        if analysis.get('dependencies'):
            summary += f"\n🔗 Dependencies: {len(analysis['dependencies'])}\n"
            for dep in analysis['dependencies'][:5]:
                summary += f"  - {dep}\n"

        return summary

    def search_code(self, query: str) -> List[Dict[str, Any]]:
        """
        Search codebase for query string

        Args:
            query: Search term

        Returns:
            List of matches with context
        """
        matches = []

        # Search Python files
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    lines = f.readlines()

                for i, line in enumerate(lines, 1):
                    if query.lower() in line.lower():
                        matches.append({
                            'file': str(py_file.relative_to(self.project_root)),
                            'line': i,
                            'content': line.strip(),
                            'context': ''.join(lines[max(0, i-3):i+2])
                        })

            except Exception as e:
                logger.warning(f"Error searching {py_file}: {e}")

        return matches


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("CODE ANALYZER TEST")
    print("=" * 70)

    analyzer = CodeAnalyzer()

    # Test 1: Analyze component
    print("\n[Test 1] Analyze Kalman Filter:")
    analysis = analyzer.analyze_component("Kalman Filter")
    print(f"Purpose: {analysis.get('purpose', 'N/A')}")
    print(f"Classes: {len(analysis.get('classes', []))}")
    print(f"Key functions: {len(analysis.get('key_functions', []))}")

    # Test 2: System overview
    print("\n[Test 2] System Overview:")
    overview = analyzer.get_system_overview()
    print(overview[:300] + "...")

    # Test 3: File summary
    print("\n[Test 3] File Summary:")
    summary = analyzer.get_file_summary("libs/llm/signal_generator.py")
    print(summary[:300] + "...")

    # Test 4: Search code
    print("\n[Test 4] Search for 'Shannon':")
    matches = analyzer.search_code("Shannon")
    print(f"Found {len(matches)} matches")
    if matches:
        print(f"First match: {matches[0]['file']}:{matches[0]['line']}")

    print("\n" + "=" * 70)
    print("✅ Code Analyzer ready for production!")
    print("=" * 70)
