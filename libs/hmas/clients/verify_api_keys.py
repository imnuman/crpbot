"""
Verify HMAS API Keys

This script checks which API keys are working and provides
instructions for getting/fixing invalid keys.
"""
import asyncio
import sys
from libs.hmas.config.hmas_config import get_config


async def verify_deepseek():
    """Verify DeepSeek API key"""
    from libs.hmas.clients.deepseek_client import DeepSeekClient

    try:
        config = get_config()
        client = DeepSeekClient(api_key=config.deepseek_api_key)
        response = await client.analyze(
            prompt="Say: OK",
            temperature=0.0,
            max_tokens=5
        )
        return True, response[:20]
    except Exception as e:
        return False, str(e)[:100]


async def verify_gemini():
    """Verify Gemini API key"""
    from libs.hmas.clients.gemini_client import GeminiClient

    try:
        config = get_config()
        client = GeminiClient(api_key=config.google_api_key)
        response = await client.generate(
            prompt="Say: OK",
            temperature=0.0,
            max_tokens=5
        )
        text = client.extract_text(response)
        return True, text[:20]
    except Exception as e:
        error_str = str(e)
        if "429" in error_str:
            return True, "Rate limited (key is valid)"
        return False, str(e)[:100]


async def verify_xai():
    """Verify X.AI Grok API key"""
    from libs.hmas.clients.xai_client import XAIClient

    try:
        config = get_config()
        client = XAIClient(api_key=config.xai_api_key)
        response = await client.analyze(
            prompt="Say: OK",
            temperature=0.0,
            max_tokens=5
        )
        return True, response[:20]
    except Exception as e:
        error_str = str(e)
        if "401" in error_str or "403" in error_str:
            return False, "Invalid API key (authentication failed)"
        return False, str(e)[:100]


async def verify_claude():
    """Verify Claude API key"""
    from libs.hmas.clients.claude_client import ClaudeClient

    try:
        config = get_config()
        client = ClaudeClient(api_key=config.anthropic_api_key)
        response = await client.analyze(
            prompt="Say: OK",
            temperature=0.0,
            max_tokens=5
        )
        return True, response[:20]
    except Exception as e:
        error_str = str(e)
        if "401" in error_str or "404" in error_str or "authentication" in error_str.lower():
            return False, "Invalid API key (authentication failed)"
        return False, str(e)[:100]


async def main():
    print("\n" + "=" * 80)
    print("HMAS API KEY VERIFICATION")
    print("=" * 80)

    results = {}

    print("\nTesting API keys...\n")

    # Test each client
    print("1. DeepSeek (Alpha Generator)...", end=" ")
    results['deepseek'] = await verify_deepseek()
    status = "✅ VALID" if results['deepseek'][0] else "❌ INVALID"
    print(f"{status}")
    if not results['deepseek'][0]:
        print(f"   Error: {results['deepseek'][1]}")

    print("2. Gemini (Mother AI)...", end=" ")
    results['gemini'] = await verify_gemini()
    status = "✅ VALID" if results['gemini'][0] else "❌ INVALID"
    print(f"{status}")
    if not results['gemini'][0]:
        print(f"   Error: {results['gemini'][1]}")

    print("3. X.AI Grok (Execution Auditor)...", end=" ")
    results['xai'] = await verify_xai()
    status = "✅ VALID" if results['xai'][0] else "❌ INVALID"
    print(f"{status}")
    if not results['xai'][0]:
        print(f"   Error: {results['xai'][1]}")

    print("4. Claude (Rationale Agent)...", end=" ")
    results['claude'] = await verify_claude()
    status = "✅ VALID" if results['claude'][0] else "❌ INVALID"
    print(f"{status}")
    if not results['claude'][0]:
        print(f"   Error: {results['claude'][1]}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    valid_count = sum(1 for r in results.values() if r[0])
    print(f"\nValid API keys: {valid_count}/4\n")

    if valid_count < 4:
        print("⚠️  MISSING API KEYS - Follow instructions below:\n")

        if not results['gemini'][0]:
            print("❌ Google Gemini:")
            print("   1. Go to: https://makersuite.google.com/app/apikey")
            print("   2. Create API key")
            print("   3. Update .env: GOOGLE_API_KEY=your_key_here\n")

        if not results['xai'][0]:
            print("❌ X.AI Grok:")
            print("   1. Go to: https://console.x.ai/")
            print("   2. Create API key")
            print("   3. Update .env: XAI_API_KEY=xai-...\n")

        if not results['claude'][0]:
            print("❌ Anthropic Claude:")
            print("   1. Go to: https://console.anthropic.com/")
            print("   2. Create API key")
            print("   3. Update .env: ANTHROPIC_API_KEY=sk-ant-...\n")

        if not results['deepseek'][0]:
            print("❌ DeepSeek:")
            print("   1. Go to: https://platform.deepseek.com/")
            print("   2. Create API key")
            print("   3. Update .env: DEEPSEEK_API_KEY=sk-...\n")

        print("After updating .env, run this script again to verify.\n")

    else:
        print("🎉 ALL API KEYS VALID! Ready to build HMAS agents.\n")

    print("=" * 80)

    return valid_count == 4


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
