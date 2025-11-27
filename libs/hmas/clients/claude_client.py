"""
Anthropic Claude API Client
Rationale Agent (L4) - Explanation & Memory
"""
import httpx
from typing import Dict, Any, Optional, List


class ClaudeClient:
    """Client for Anthropic Claude API"""

    BASE_URL = "https://api.anthropic.com/v1"
    MODEL = "claude-3-5-sonnet-20241022"
    API_VERSION = "2023-06-01"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = self.MODEL

    async def messages_create(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Create a message with Claude

        Args:
            messages: List of message dicts with 'role' and 'content'
            system: Optional system prompt
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate

        Returns:
            API response dictionary
        """
        url = f"{self.BASE_URL}/messages"

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        if system:
            payload["system"] = system

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                url,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            return response.json()

    def extract_text(self, response: Dict[str, Any]) -> str:
        """Extract text from Claude API response"""
        try:
            content = response.get("content", [])
            if not content:
                return ""

            # Claude returns list of content blocks
            text_blocks = [
                block.get("text", "")
                for block in content
                if block.get("type") == "text"
            ]

            return "\n".join(text_blocks)
        except (KeyError, IndexError):
            return ""

    async def analyze(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> str:
        """
        Single-turn analysis with Claude

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            Generated text response
        """
        messages = [{
            "role": "user",
            "content": prompt
        }]

        response = await self.messages_create(
            messages=messages,
            system=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return self.extract_text(response)

    async def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> str:
        """
        Multi-turn chat with Claude

        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            Generated text response
        """
        response = await self.messages_create(
            messages=messages,
            system=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return self.extract_text(response)

    def __repr__(self) -> str:
        masked_key = f"{self.api_key[:8]}..." if len(self.api_key) > 8 else "***"
        return f"ClaudeClient(model='{self.model}', api_key='{masked_key}')"
