"""
DeepSeek API Client (HMAS Version)
Alpha Generator (L2) - Pattern Recognition & Trade Hypothesis

This is a lightweight wrapper around the V7 DeepSeek client
for compatibility with the HMAS architecture.
"""
import sys
import os

# Import the existing V7 DeepSeek client
from libs.llm.deepseek_client import DeepSeekClient as V7DeepSeekClient
from typing import Dict, Any, List, Optional


class DeepSeekClient:
    """
    HMAS-compatible DeepSeek client wrapper

    Wraps the existing V7 DeepSeek client for use in HMAS architecture.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = V7DeepSeekClient(api_key=api_key)

    async def analyze(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """
        Single-turn analysis with DeepSeek

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            Generated text response
        """
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        # Call V7 client (synchronous)
        response = self.client.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.content

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """
        Multi-turn chat with DeepSeek

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            Generated text response
        """
        # Call V7 client (synchronous)
        response = self.client.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.content

    def get_stats(self) -> Dict[str, Any]:
        """Get client usage statistics"""
        return self.client.get_stats()

    def __repr__(self) -> str:
        masked_key = f"{self.api_key[:8]}..." if len(self.api_key) > 8 else "***"
        return f"DeepSeekClient(model='deepseek-chat', api_key='{masked_key}')"
