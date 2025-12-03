"""
HYDRA 3.0 - Coinbase Client Wrapper

This module provides a singleton wrapper around the existing CoinbaseDataProvider
to match HYDRA's expected interface.
"""

import os
from typing import Optional
from loguru import logger

from libs.data.coinbase import CoinbaseDataProvider


# Global singleton instance
_coinbase_client: Optional[CoinbaseDataProvider] = None


def get_coinbase_client() -> CoinbaseDataProvider:
    """
    Get singleton instance of Coinbase data provider.

    Returns:
        CoinbaseDataProvider: Initialized Coinbase client

    Raises:
        ValueError: If API credentials are not configured
    """
    global _coinbase_client

    if _coinbase_client is None:
        # Load credentials from environment
        api_key_name = os.getenv("COINBASE_API_KEY_NAME")
        private_key = os.getenv("COINBASE_API_PRIVATE_KEY")

        if not api_key_name or not private_key:
            raise ValueError(
                "Coinbase credentials not found. "
                "Please set COINBASE_API_KEY_NAME and COINBASE_API_PRIVATE_KEY in .env"
            )

        # Initialize the provider
        _coinbase_client = CoinbaseDataProvider(
            api_key_name=api_key_name,
            private_key=private_key
        )

        logger.info("Coinbase client initialized successfully")

    return _coinbase_client


def reset_coinbase_client():
    """Reset the singleton instance (useful for testing)."""
    global _coinbase_client
    _coinbase_client = None
