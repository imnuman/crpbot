"""
Google Gemini API Client
Mother AI (L1) - Orchestration & Risk Governance
"""
import os
import httpx
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone


class GeminiClient:
    """Client for Google Gemini 2.5 Flash API"""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
    MODEL = "gemini-2.5-flash"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = self.MODEL

    async def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        Generate response from Gemini

        Args:
            prompt: User prompt
            system_instruction: Optional system instruction
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate

        Returns:
            API response dictionary
        """
        url = f"{self.BASE_URL}/{self.model}:generateContent"

        # Build request payload
        contents = [{"parts": [{"text": prompt}]}]

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            }
        }

        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        # Make API request
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                url,
                params={"key": self.api_key},
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()

    def extract_text(self, response: Dict[str, Any]) -> str:
        """Extract text from Gemini API response"""
        try:
            candidates = response.get("candidates", [])
            if not candidates:
                return ""

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])

            if not parts:
                return ""

            return parts[0].get("text", "")
        except (KeyError, IndexError):
            return ""

    async def chat(
        self,
        messages: List[Dict[str, str]],
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """
        Chat completion with Gemini

        Args:
            messages: List of message dicts with 'role' and 'content'
            system_instruction: Optional system instruction
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            Generated text response
        """
        # Convert messages to Gemini format
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })

        url = f"{self.BASE_URL}/{self.model}:generateContent"

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            }
        }

        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                url,
                params={"key": self.api_key},
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()

        return self.extract_text(result)

    def __repr__(self) -> str:
        masked_key = f"{self.api_key[:8]}..." if len(self.api_key) > 8 else "***"
        return f"GeminiClient(model='{self.model}', api_key='{masked_key}')"
