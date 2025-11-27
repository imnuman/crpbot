"""
X.AI Grok API Client
Execution Auditor (L3) - Speed & Aggressive Loss Management
"""
import httpx
from typing import Dict, Any, Optional, List


class XAIClient:
    """Client for X.AI Grok API"""

    BASE_URL = "https://api.x.ai/v1"
    MODEL = "grok-beta"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = self.MODEL

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        Chat completion with Grok

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate

        Returns:
            API response dictionary
        """
        url = f"{self.BASE_URL}/chat/completions"

        payload = {
            "messages": messages,
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            return response.json()

    def extract_text(self, response: Dict[str, Any]) -> str:
        """Extract text from Grok API response"""
        try:
            choices = response.get("choices", [])
            if not choices:
                return ""

            message = choices[0].get("message", {})
            return message.get("content", "")
        except (KeyError, IndexError):
            return ""

    async def analyze(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2000
    ) -> str:
        """
        Single-turn analysis with Grok

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

        response = await self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return self.extract_text(response)

    def __repr__(self) -> str:
        masked_key = f"{self.api_key[:8]}..." if len(self.api_key) > 8 else "***"
        return f"XAIClient(model='{self.model}', api_key='{masked_key}')"
