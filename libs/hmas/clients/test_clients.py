"""
Test all 4 HMAS API clients
"""
import asyncio
import sys
import os
from libs.hmas.config.hmas_config import get_config


async def test_gemini_client():
    """Test Gemini API client"""
    print("\n" + "=" * 80)
    print("TEST 1: Gemini Client (Mother AI)")
    print("=" * 80)

    from libs.hmas.clients.gemini_client import GeminiClient

    try:
        config = get_config()
        client = GeminiClient(api_key=config.google_api_key)
        print(f"✅ Client initialized: {client}")

        # Test simple generation
        response = await client.generate(
            prompt="In one sentence, what is the primary goal of risk management in trading?",
            temperature=0.7,
            max_tokens=100
        )

        text = client.extract_text(response)
        print(f"\n📝 Response: {text[:200]}...")

        # Test chat
        messages = [
            {"role": "user", "content": "What is 2+2?"}
        ]
        chat_response = await client.chat(messages, temperature=0.0, max_tokens=50)
        print(f"💬 Chat Response: {chat_response}")

        print(f"\n✅ Gemini Client: WORKING")
        return True

    except Exception as e:
        print(f"❌ Gemini Client FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_deepseek_client():
    """Test DeepSeek API client"""
    print("\n" + "=" * 80)
    print("TEST 2: DeepSeek Client (Alpha Generator)")
    print("=" * 80)

    from libs.hmas.clients.deepseek_client import DeepSeekClient

    try:
        config = get_config()
        client = DeepSeekClient(api_key=config.deepseek_api_key)
        print(f"✅ Client initialized: {client}")

        # Test analysis
        response = await client.analyze(
            prompt="In one sentence, what makes a good trading signal?",
            system_prompt="You are a trading analyst.",
            temperature=0.7,
            max_tokens=100
        )

        print(f"\n📝 Response: {response[:200]}...")

        # Test chat
        messages = [
            {"role": "system", "content": "You are brief and concise."},
            {"role": "user", "content": "What is 3+3?"}
        ]
        chat_response = await client.chat(messages, temperature=0.0, max_tokens=50)
        print(f"💬 Chat Response: {chat_response}")

        # Show stats
        stats = client.get_stats()
        print(f"\n📊 Stats: Cost: ${stats['total_cost_usd']:.6f}, Tokens: {stats['total_tokens']}")

        print(f"\n✅ DeepSeek Client: WORKING")
        return True

    except Exception as e:
        print(f"❌ DeepSeek Client FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_xai_client():
    """Test X.AI Grok API client"""
    print("\n" + "=" * 80)
    print("TEST 3: X.AI Grok Client (Execution Auditor)")
    print("=" * 80)

    from libs.hmas.clients.xai_client import XAIClient

    try:
        config = get_config()
        client = XAIClient(api_key=config.xai_api_key)
        print(f"✅ Client initialized: {client}")

        # Test analysis
        response = await client.analyze(
            prompt="In one sentence, what is the purpose of a stop-loss order?",
            system_prompt="You are a risk management expert.",
            temperature=0.0,
            max_tokens=100
        )

        print(f"\n📝 Response: {response[:200]}...")

        # Test chat completion
        messages = [
            {"role": "system", "content": "You are brief and concise."},
            {"role": "user", "content": "What is 5+5?"}
        ]
        chat_resp = await client.chat_completion(messages, temperature=0.0, max_tokens=50)
        chat_text = client.extract_text(chat_resp)
        print(f"💬 Chat Response: {chat_text}")

        print(f"\n✅ X.AI Grok Client: WORKING")
        return True

    except Exception as e:
        print(f"❌ X.AI Grok Client FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_claude_client():
    """Test Claude API client"""
    print("\n" + "=" * 80)
    print("TEST 4: Claude Client (Rationale Agent)")
    print("=" * 80)

    from libs.hmas.clients.claude_client import ClaudeClient

    try:
        config = get_config()
        client = ClaudeClient(api_key=config.anthropic_api_key)
        print(f"✅ Client initialized: {client}")

        # Test analysis
        response = await client.analyze(
            prompt="In one sentence, explain why maintaining a trading journal is important.",
            system_prompt="You are a trading psychology expert.",
            temperature=0.7,
            max_tokens=150
        )

        print(f"\n📝 Response: {response[:200]}...")

        # Test chat
        messages = [
            {"role": "user", "content": "What is 7+7?"}
        ]
        chat_response = await client.chat(
            messages,
            system_prompt="You are brief and concise.",
            temperature=0.0,
            max_tokens=50
        )
        print(f"💬 Chat Response: {chat_response}")

        print(f"\n✅ Claude Client: WORKING")
        return True

    except Exception as e:
        print(f"❌ Claude Client FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all client tests"""
    print("\n" + "=" * 80)
    print("HMAS API CLIENT TESTS")
    print("=" * 80)

    results = {}

    # Test each client
    results['gemini'] = await test_gemini_client()
    results['deepseek'] = await test_deepseek_client()
    results['xai'] = await test_xai_client()
    results['claude'] = await test_claude_client()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name.upper()}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 ALL CLIENTS WORKING! Ready to build HMAS agents.")
    else:
        print("\n⚠️  Some clients failed. Check errors above.")

    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
