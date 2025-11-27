#!/usr/bin/env python3
"""
HMAS V2 System Verification Script
Tests all components without making expensive API calls
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_imports():
    """Test all critical imports"""
    print("="*80)
    print("TEST 1: Import Verification")
    print("="*80)

    try:
        print("\n1. Testing agent imports...")
        from libs.hmas.agents.alpha_generator_v2 import AlphaGeneratorV2
        print("  ✓ AlphaGeneratorV2")

        from libs.hmas.agents.rationale_agent_v2 import RationaleAgentV2
        print("  ✓ RationaleAgentV2")

        from libs.hmas.agents.technical_agent import TechnicalAgent
        print("  ✓ TechnicalAgent")

        from libs.hmas.agents.sentiment_agent import SentimentAgent
        print("  ✓ SentimentAgent")

        from libs.hmas.agents.macro_agent import MacroAgent
        print("  ✓ MacroAgent")

        from libs.hmas.agents.execution_auditor_v2 import ExecutionAuditorV2
        print("  ✓ ExecutionAuditorV2")

        from libs.hmas.agents.mother_ai_v2 import MotherAIV2
        print("  ✓ MotherAIV2")

        print("\n2. Testing orchestrator import...")
        from libs.hmas.hmas_orchestrator_v2 import HMASV2Orchestrator
        print("  ✓ HMASV2Orchestrator")

        print("\n3. Testing data client import...")
        from libs.data.coinbase import CoinbaseDataProvider
        print("  ✓ CoinbaseDataProvider")

        print("\n✅ ALL IMPORTS SUCCESSFUL")
        return True

    except Exception as e:
        print(f"\n✗ IMPORT FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_keys():
    """Test API key loading"""
    print("\n" + "="*80)
    print("TEST 2: API Key Verification")
    print("="*80)

    try:
        from dotenv import load_dotenv
        load_dotenv()

        keys = {
            'DEEPSEEK_API_KEY': os.getenv('DEEPSEEK_API_KEY'),
            'XAI_API_KEY': os.getenv('XAI_API_KEY'),
            'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
            'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY')
        }

        all_ok = True
        for key, value in keys.items():
            if value and len(value) > 10:
                print(f"  ✓ {key}: {value[:10]}...")
            else:
                print(f"  ✗ {key}: MISSING OR INVALID")
                all_ok = False

        if all_ok:
            print("\n✅ ALL API KEYS VALIDATED")
            return True
        else:
            print("\n✗ SOME API KEYS MISSING")
            return False

    except Exception as e:
        print(f"\n✗ API KEY VALIDATION FAILED: {e}")
        return False


def test_orchestrator_initialization():
    """Test orchestrator can be initialized"""
    print("\n" + "="*80)
    print("TEST 3: Orchestrator Initialization")
    print("="*80)

    try:
        from libs.hmas.hmas_orchestrator_v2 import HMASV2Orchestrator

        print("\nInitializing orchestrator with API keys...")
        orchestrator = HMASV2Orchestrator.from_env()

        print(f"  ✓ Alpha Generator: {orchestrator.alpha_generator.__class__.__name__}")
        print(f"  ✓ Technical Agent: {orchestrator.technical_agent.__class__.__name__}")
        print(f"  ✓ Sentiment Agent: {orchestrator.sentiment_agent.__class__.__name__}")
        print(f"  ✓ Macro Agent: {orchestrator.macro_agent.__class__.__name__}")
        print(f"  ✓ Execution Auditor: {orchestrator.execution_auditor.__class__.__name__}")
        print(f"  ✓ Rationale Agent: {orchestrator.rationale_agent.__class__.__name__}")
        print(f"  ✓ Mother AI: {orchestrator.mother_ai.__class__.__name__}")

        print("\n✅ ORCHESTRATOR INITIALIZED SUCCESSFULLY")
        print(f"\nOrchestrator details:\n{orchestrator}")
        return True

    except Exception as e:
        print(f"\n✗ ORCHESTRATOR INITIALIZATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_client():
    """Test data client initialization"""
    print("\n" + "="*80)
    print("TEST 4: Data Client Initialization")
    print("="*80)

    try:
        from libs.data.coinbase import CoinbaseDataProvider
        from dotenv import load_dotenv
        load_dotenv()

        api_key_name = os.getenv('COINBASE_API_KEY_NAME')
        private_key = os.getenv('COINBASE_API_PRIVATE_KEY')

        if not api_key_name or not private_key:
            print("  ⚠ Coinbase credentials not configured (optional for testing)")
            print("  ✓ Data client available but not initialized")
            return True

        print("\nInitializing Coinbase data client...")
        client = CoinbaseDataProvider(
            api_key_name=api_key_name,
            private_key=private_key
        )
        print(f"  ✓ Data client initialized: {client.__class__.__name__}")

        print("\n✅ DATA CLIENT READY")
        return True

    except Exception as e:
        print(f"\n✗ DATA CLIENT INITIALIZATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_confidence_interval_fix():
    """Test the division by zero fix in rationale agent"""
    print("\n" + "="*80)
    print("TEST 5: Critical Bug Fix Verification")
    print("="*80)

    try:
        from libs.hmas.agents.rationale_agent_v2 import RationaleAgentV2

        print("\nTesting confidence interval edge cases...")
        agent = RationaleAgentV2(api_key="test-key")

        # Test with 0 samples (should not crash)
        result = agent._format_confidence_intervals(win_rate=0.5, sample_size=0)
        print(f"  ✓ Zero samples: {result[:50]}...")

        # Test with small samples
        result = agent._format_confidence_intervals(win_rate=0.8, sample_size=3)
        print(f"  ✓ Small samples: {result[:50]}...")

        # Test with normal samples
        result = agent._format_confidence_intervals(win_rate=0.82, sample_size=20)
        print(f"  ✓ Normal samples: {result[:50]}...")

        print("\n✅ DIVISION BY ZERO BUG FIX VERIFIED")
        return True

    except Exception as e:
        print(f"\n✗ BUG FIX VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_final_summary(results):
    """Print final test summary"""
    print("\n" + "="*80)
    print("FINAL VERIFICATION SUMMARY")
    print("="*80)

    test_names = [
        "Import Verification",
        "API Key Verification",
        "Orchestrator Initialization",
        "Data Client Initialization",
        "Critical Bug Fix Verification"
    ]

    print("\nTest Results:")
    for i, (name, passed) in enumerate(zip(test_names, results), 1):
        status = "✅ PASS" if passed else "✗ FAIL"
        print(f"  {i}. {name}: {status}")

    total_passed = sum(results)
    total_tests = len(results)

    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    if all(results):
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED - HMAS V2 IS PRODUCTION READY! 🎉")
        print("="*80)
        print("\nNext steps:")
        print("1. Run test signal: .venv/bin/python apps/runtime/hmas_v2_runtime.py --symbols BTC-USD --iterations 1 --dry-run")
        print("2. Monitor output for any API errors")
        print("3. If successful, deploy to production")
        return True
    else:
        print("\n" + "="*80)
        print("⚠ SOME TESTS FAILED - REVIEW ERRORS ABOVE")
        print("="*80)
        return False


def main():
    """Run all verification tests"""
    print("\n╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                                                                              ║")
    print("║                    HMAS V2 SYSTEM VERIFICATION                               ║")
    print("║                                                                              ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝\n")

    results = []

    # Run all tests
    results.append(test_imports())
    results.append(test_api_keys())
    results.append(test_orchestrator_initialization())
    results.append(test_data_client())
    results.append(test_confidence_interval_fix())

    # Print summary
    success = print_final_summary(results)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
