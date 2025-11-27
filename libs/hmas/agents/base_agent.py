"""
Base Agent Class for HMAS
All specialized agents inherit from this base class
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime, timezone


class BaseAgent(ABC):
    """Abstract base class for all HMAS agents"""

    def __init__(self, name: str, api_key: str):
        self.name = name
        self.api_key = api_key
        self.created_at = datetime.now(timezone.utc)

    @abstractmethod
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze input data and return agent-specific output

        Args:
            data: Input data dictionary

        Returns:
            Analysis results dictionary
        """
        pass

    def validate_api_key(self) -> bool:
        """Validate that API key is set and not a placeholder"""
        if not self.api_key:
            return False

        # Check for placeholder values
        placeholders = [
            'your_',
            'placeholder',
            'REPLACE_ME',
            'xxx',
        ]

        for placeholder in placeholders:
            if placeholder.lower() in self.api_key.lower():
                return False

        return True

    def __repr__(self) -> str:
        masked_key = f"{self.api_key[:8]}..." if len(self.api_key) > 8 else "***"
        return f"{self.__class__.__name__}(name='{self.name}', api_key='{masked_key}')"
