"""
DeepSeek LLM Client for V7 Ultimate Signal Synthesis

DeepSeek API is OpenAI-compatible with extremely low pricing:
- DeepSeek-V3: $0.27/M input tokens, $1.10/M output tokens
- Budget: $100/month target

API Documentation: https://api-docs.deepseek.com/
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import requests

logger = logging.getLogger(__name__)


@dataclass
class DeepSeekResponse:
    """Response from DeepSeek API"""
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    finish_reason: str
    timestamp: datetime

    def __str__(self) -> str:
        return (
            f"DeepSeekResponse(\n"
            f"  content_length={len(self.content)},\n"
            f"  tokens={self.total_tokens} ({self.prompt_tokens}+{self.completion_tokens}),\n"
            f"  cost=${self.cost_usd:.6f},\n"
            f"  finish_reason={self.finish_reason}\n"
            f")"
        )


class DeepSeekClient:
    """
    DeepSeek LLM API Client

    Features:
    - OpenAI-compatible API
    - Automatic retries with exponential backoff
    - Cost tracking
    - Rate limiting
    - Error handling

    Usage:
        client = DeepSeekClient(api_key=os.getenv('DEEPSEEK_API_KEY'))
        response = client.chat(
            messages=[
                {"role": "system", "content": "You are a trading analyst."},
                {"role": "user", "content": "Analyze BTC market..."}
            ],
            temperature=0.7
        )
    """

    # API Configuration
    BASE_URL = "https://api.deepseek.com/v1"
    DEFAULT_MODEL = "deepseek-chat"  # DeepSeek-V3 model

    # Pricing (per million tokens)
    # Source: https://api-docs.deepseek.com/quick_start/pricing
    PRICING = {
        "deepseek-chat": {
            "input": 0.27,   # $0.27 per 1M input tokens
            "output": 1.10,  # $1.10 per 1M output tokens
        },
        "deepseek-reasoner": {
            "input": 0.55,   # $0.55 per 1M input tokens
            "output": 2.19,  # $2.19 per 1M output tokens
        }
    }

    # Rate limiting (conservative defaults)
    MAX_REQUESTS_PER_MINUTE = 50
    MAX_TOKENS_PER_REQUEST = 16000  # Increased for comprehensive analysis (was 8000)

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        timeout: int = 180,  # Increased from 90s for HMAS V2 agents
        max_retries: int = 3
    ):
        """
        Initialize DeepSeek API client

        Args:
            api_key: DeepSeek API key (or set DEEPSEEK_API_KEY env var)
            model: Model to use (default: deepseek-chat)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries on failure
        """
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key required. Set DEEPSEEK_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

        # Request tracking
        self.request_times: List[float] = []
        self.total_cost = 0.0
        self.total_tokens = 0

        # API headers
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        logger.info(
            f"DeepSeek client initialized | Model: {model} | "
            f"Timeout: {timeout}s | Max Retries: {max_retries}"
        )

    def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting"""
        current_time = time.time()

        # Remove requests older than 1 minute
        self.request_times = [
            t for t in self.request_times
            if current_time - t < 60
        ]

        # Check if we've hit the rate limit
        if len(self.request_times) >= self.MAX_REQUESTS_PER_MINUTE:
            sleep_time = 60 - (current_time - self.request_times[0])
            if sleep_time > 0:
                logger.warning(
                    f"Rate limit reached. Sleeping for {sleep_time:.1f}s"
                )
                time.sleep(sleep_time)
                self.request_times = []

        self.request_times.append(current_time)

    def _calculate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str
    ) -> float:
        """
        Calculate API call cost in USD

        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            model: Model used

        Returns:
            Cost in USD
        """
        if model not in self.PRICING:
            logger.warning(f"Unknown model pricing: {model}. Using default.")
            model = self.DEFAULT_MODEL

        pricing = self.PRICING[model]

        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> DeepSeekResponse:
        """
        Send chat completion request to DeepSeek API

        Args:
            messages: List of message dicts with 'role' and 'content'
                     Example: [
                         {"role": "system", "content": "You are helpful."},
                         {"role": "user", "content": "Hello!"}
                     ]
            temperature: Sampling temperature (0.0 - 2.0)
            max_tokens: Maximum tokens to generate (default: None for auto)
            model: Model override (default: use client's model)
            **kwargs: Additional API parameters

        Returns:
            DeepSeekResponse object

        Raises:
            requests.exceptions.RequestException: On API error
        """
        # Check rate limit
        self._check_rate_limit()

        # Prepare request payload
        model = model or self.model
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        # Retry logic with exponential backoff
        last_exception = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    f"DeepSeek API request (attempt {attempt}/{self.max_retries})"
                )

                response = requests.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )

                # Raise for HTTP errors
                response.raise_for_status()

                # Parse response
                data = response.json()

                # Extract response data
                choice = data["choices"][0]
                usage = data["usage"]

                content = choice["message"]["content"]
                finish_reason = choice["finish_reason"]

                prompt_tokens = usage["prompt_tokens"]
                completion_tokens = usage["completion_tokens"]
                total_tokens = usage["total_tokens"]

                # Calculate cost
                cost = self._calculate_cost(
                    prompt_tokens, completion_tokens, model
                )

                # Update totals
                self.total_cost += cost
                self.total_tokens += total_tokens

                logger.info(
                    f"DeepSeek response | Tokens: {total_tokens} | "
                    f"Cost: ${cost:.6f} | Total Cost: ${self.total_cost:.6f}"
                )

                return DeepSeekResponse(
                    content=content,
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost_usd=cost,
                    finish_reason=finish_reason,
                    timestamp=datetime.now()
                )

            except requests.exceptions.RequestException as e:
                last_exception = e

                # Check if we should retry
                if attempt < self.max_retries:
                    # Exponential backoff: 2^attempt seconds
                    sleep_time = 2 ** attempt
                    logger.warning(
                        f"DeepSeek API error: {e}. "
                        f"Retrying in {sleep_time}s... "
                        f"(attempt {attempt}/{self.max_retries})"
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error(
                        f"DeepSeek API failed after {self.max_retries} attempts: {e}"
                    )

        # If we get here, all retries failed
        raise last_exception

    def get_stats(self) -> Dict[str, Any]:
        """
        Get client usage statistics

        Returns:
            Dictionary with usage stats
        """
        return {
            "total_requests": len(self.request_times),
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost,
            "avg_cost_per_request": (
                self.total_cost / max(len(self.request_times), 1)
            ),
            "model": self.model,
            "requests_last_minute": len(self.request_times)
        }

    def reset_stats(self) -> None:
        """Reset usage statistics"""
        self.total_cost = 0.0
        self.total_tokens = 0
        logger.info("DeepSeek client stats reset")


# Convenience function for quick usage
def chat_with_deepseek(
    user_message: str,
    system_message: str = "You are a helpful assistant.",
    temperature: float = 0.7,
    api_key: Optional[str] = None
) -> str:
    """
    Quick chat with DeepSeek (creates temporary client)

    Args:
        user_message: User's message
        system_message: System prompt
        temperature: Sampling temperature
        api_key: DeepSeek API key (or use env var)

    Returns:
        Response content string
    """
    client = DeepSeekClient(api_key=api_key)

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    response = client.chat(messages, temperature=temperature)
    return response.content


if __name__ == "__main__":
    # Test DeepSeek client
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-5s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    print("=" * 80)
    print("DeepSeek Client - Test Run")
    print("=" * 80)

    # Check for API key
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("\n❌ ERROR: DEEPSEEK_API_KEY environment variable not set")
        print("\nPlease set your DeepSeek API key:")
        print("  export DEEPSEEK_API_KEY='sk-...'")
        print("\nGet your API key from: https://platform.deepseek.com/")
        print("=" * 80)
        exit(1)

    try:
        # Initialize client
        print("\n1. Initializing DeepSeek Client...")
        client = DeepSeekClient(api_key=api_key)

        # Test 1: Simple chat
        print("\n2. Test: Simple Chat")
        messages = [
            {
                "role": "system",
                "content": "You are a cryptocurrency trading analyst."
            },
            {
                "role": "user",
                "content": "In one sentence, what makes Bitcoin valuable?"
            }
        ]

        response = client.chat(messages, temperature=0.7, max_tokens=100)

        print(f"\n   Response: {response.content}")
        print(f"   Tokens: {response.total_tokens}")
        print(f"   Cost: ${response.cost_usd:.6f}")
        print(f"   Finish Reason: {response.finish_reason}")

        # Test 2: Multi-turn conversation
        print("\n3. Test: Multi-turn Conversation")
        conversation = [
            {
                "role": "system",
                "content": "You are a brief, concise assistant."
            },
            {
                "role": "user",
                "content": "What is 2+2?"
            }
        ]

        response2 = client.chat(conversation, temperature=0.0, max_tokens=50)
        print(f"   Q: What is 2+2?")
        print(f"   A: {response2.content}")

        # Add assistant response and continue conversation
        conversation.append({
            "role": "assistant",
            "content": response2.content
        })
        conversation.append({
            "role": "user",
            "content": "What about 3+3?"
        })

        response3 = client.chat(conversation, temperature=0.0, max_tokens=50)
        print(f"   Q: What about 3+3?")
        print(f"   A: {response3.content}")

        # Test 3: Temperature variations
        print("\n4. Test: Temperature Variations")
        creative_prompt = [
            {
                "role": "system",
                "content": "You are creative and imaginative."
            },
            {
                "role": "user",
                "content": "Describe Bitcoin in 5 words."
            }
        ]

        for temp in [0.0, 0.5, 1.0]:
            resp = client.chat(creative_prompt, temperature=temp, max_tokens=30)
            print(f"   Temperature {temp}: {resp.content}")

        # Display stats
        print("\n5. Client Statistics")
        stats = client.get_stats()
        print(f"   Total Requests: {stats['total_requests']}")
        print(f"   Total Tokens: {stats['total_tokens']}")
        print(f"   Total Cost: ${stats['total_cost_usd']:.6f}")
        print(f"   Avg Cost/Request: ${stats['avg_cost_per_request']:.6f}")

        # Cost projection
        print("\n6. Monthly Cost Projection")
        signals_per_day = 100  # Assume 100 signals per day
        avg_tokens_per_signal = stats['total_tokens'] / stats['total_requests']
        tokens_per_month = signals_per_day * 30 * avg_tokens_per_signal

        # Assume 70% input, 30% output
        input_tokens = tokens_per_month * 0.7
        output_tokens = tokens_per_month * 0.3

        monthly_cost = (
            (input_tokens / 1_000_000) * 0.27 +
            (output_tokens / 1_000_000) * 1.10
        )

        print(f"   Assumption: {signals_per_day} signals/day")
        print(f"   Avg Tokens/Signal: {avg_tokens_per_signal:.0f}")
        print(f"   Monthly Tokens: {tokens_per_month:,.0f}")
        print(f"   Projected Monthly Cost: ${monthly_cost:.2f}")
        print(f"   Budget Status: {'✅ Within $100 budget' if monthly_cost <= 100 else '❌ Exceeds budget'}")

        print("\n" + "=" * 80)
        print("DeepSeek Client Test Complete!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 80)
        exit(1)
