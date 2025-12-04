"""Coinbase Advanced Trade API data provider implementation using JWT authentication."""
import secrets
import time
from datetime import datetime, timezone
from typing import Any

import jwt
import pandas as pd
import requests
from cryptography.hazmat.primitives import serialization
from loguru import logger

from libs.data.provider import DataProviderInterface


class CoinbaseDataProvider(DataProviderInterface):
    """Coinbase Advanced Trade API data provider using JWT authentication."""

    BASE_URL = "https://api.coinbase.com/api/v3/brokerage"

    def __init__(self, api_key_name: str, private_key: str):
        """
        Initialize Coinbase data provider.

        Args:
            api_key_name: Full API key name (e.g., organizations/.../apiKeys/...)
            private_key: PEM-encoded EC private key
        """
        if not api_key_name or not private_key:
            raise ValueError(
                "Coinbase API credentials are required. "
                "Please set COINBASE_API_KEY_NAME and COINBASE_API_PRIVATE_KEY in .env"
            )
        
        self.api_key_name = api_key_name
        self.private_key_pem = private_key
        
        # Load the private key
        # Handle multi-line private keys (common in .env files)
        # Replace escaped newlines with actual newlines
        private_key_clean = private_key.replace("\\n", "\n")
        
        # Ensure proper PEM format
        if not private_key_clean.strip().startswith("-----BEGIN"):
            logger.error("Private key must start with -----BEGIN EC PRIVATE KEY-----")
            raise ValueError("Invalid private key format: missing PEM header")
        
        try:
            self.private_key = serialization.load_pem_private_key(
                private_key_clean.encode(), password=None
            )
        except Exception as e:
            logger.error(f"Failed to load private key: {e}")
            logger.error(f"Private key format check: starts_with_header={private_key.strip().startswith('-----BEGIN')}, length={len(private_key)} chars")
            logger.error(
                "Troubleshooting:\n"
                "1. Private key should be PEM format (starts with -----BEGIN EC PRIVATE KEY-----\n"
                "2. In .env file, use quotes and keep the \\n newlines\n"
                "3. Or put the private key on multiple lines (without quotes)\n"
                "4. Make sure there are no extra escape characters"
            )
            raise ValueError(
                "Invalid private key format. Expected PEM-encoded EC private key. "
                f"Error: {e}"
            ) from e
        
        logger.info("Coinbase data provider initialized (JWT authentication)")

    def _generate_jwt(self, method: str = "GET", path: str = "") -> str:
        """Generate JWT token for Coinbase Advanced Trade API."""
        now = int(time.time())
        
        # Build URI for the request (e.g., "GET api.coinbase.com/api/v3/brokerage/products")
        if not path:
            path = "/api/v3/brokerage/products"  # Default
        uri = f"{method} api.coinbase.com{path}"
        
        # JWT payload according to Coinbase Advanced Trade API spec
        payload = {
            "iss": "cdp",  # Issuer
            "sub": self.api_key_name,  # Subject (API key name)
            "nbf": now,  # Not before
            "exp": now + 120,  # Expiration (2 minutes)
            "uri": uri,  # Request URI
        }
        
        # JWT headers
        headers = {
            "kid": self.api_key_name,  # Key ID
            "nonce": secrets.token_hex(),  # Random nonce
        }
        
        # Generate JWT using ES256 algorithm (ECDSA with P-256 and SHA-256)
        token = jwt.encode(payload, self.private_key, algorithm="ES256", headers=headers)
        return token

    def _make_request(
        self, method: str, endpoint: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Make authenticated request to Coinbase API using JWT."""
        # Build request path for JWT (without query params - they're sent separately)
        request_path = f"/api/v3/brokerage{endpoint}"
        
        # Generate fresh JWT for each request (includes request method and path only, no query params)
        jwt_token = self._generate_jwt(method=method, path=request_path)
        
        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Content-Type": "application/json",
        }
        
        url = f"{self.BASE_URL}{endpoint}"
        
        logger.debug(f"Request: {method} {url}")
        logger.debug(f"JWT token generated (expires in 2 minutes)")
        
        response = requests.request(method, url, headers=headers, params=params, timeout=30)
        
        # Log response for debugging
        if response.status_code != 200:
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response body: {response.text[:500]}")
            logger.debug(f"Request URL: {url}")
            logger.debug(f"Request params: {params}")

        if response.status_code != 200:
            error_detail = response.text
            logger.error(f"Coinbase API error: {response.status_code}")
            logger.error(f"Response: {error_detail[:500]}")
            
            # Provide helpful error messages
            if response.status_code == 401:
                logger.error(
                    "Authentication failed. Please verify:\n"
                    "1. COINBASE_API_KEY_NAME is the full path (organizations/.../apiKeys/...)\n"
                    "2. COINBASE_API_PRIVATE_KEY is the PEM-encoded private key\n"
                    "3. API key has 'View' permissions enabled\n"
                    "4. API key is not expired or revoked\n"
                    "5. IP whitelisting includes your IP address"
                )
            
            response.raise_for_status()

        return response.json()

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candle data from Coinbase.

        Args:
            symbol: Trading pair (e.g., 'BTC-USD', 'ETH-USD')
            interval: Time interval ('1m', '5m', '1h', '1d')
            start_time: Start datetime (optional)
            end_time: End datetime (optional)
            limit: Maximum number of candles (default: 300, max: 300)

        Returns:
            DataFrame with columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        # Map interval to Coinbase format
        interval_map = {
            "1m": "ONE_MINUTE",
            "5m": "FIVE_MINUTE",
            "15m": "FIFTEEN_MINUTE",
            "1h": "ONE_HOUR",
            "4h": "FOUR_HOUR",
            "1d": "ONE_DAY",
        }
        coinbase_interval = interval_map.get(interval.lower(), "ONE_MINUTE")

        params: dict[str, Any] = {
            "product_id": symbol,
            "granularity": coinbase_interval,
        }

        if start_time:
            params["start"] = int(start_time.timestamp())
        if end_time:
            params["end"] = int(end_time.timestamp())
        if limit:
            params["limit"] = min(limit, 300)  # Coinbase max is 300
        else:
            params["limit"] = 300

        try:
            # Coinbase Advanced Trade API uses /product_book/{product_id} for candles
            # But let's try /products/{product_id}/candles first
            endpoint = f"/products/{symbol}/candles"
            response = self._make_request("GET", endpoint, params)
            candles = response.get("candles", [])

            if not candles:
                logger.warning(f"No candles returned for {symbol} {interval}")
                return pd.DataFrame(
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )

            # Convert to DataFrame
            # Coinbase returns list of dicts: [{"start": "...", "low": "...", "high": "...", "open": "...", "close": "...", "volume": "..."}, ...]
            df = pd.DataFrame(candles)
            
            # Convert timestamp from 'start' field (Unix timestamp as string)
            if "start" in df.columns:
                df["timestamp"] = pd.to_datetime(df["start"].astype(int), unit="s", utc=True)
            else:
                logger.error("Missing 'start' field in candle data")
                return pd.DataFrame(
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
            
            # Convert OHLCV values from strings to floats
            numeric_cols = ["open", "high", "low", "close", "volume"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                else:
                    logger.error(f"Missing '{col}' field in candle data")
                    return pd.DataFrame(
                        columns=["timestamp", "open", "high", "low", "close", "volume"]
                    )
            
            # Select and reorder columns
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            df = df.sort_values("timestamp").reset_index(drop=True)

            return df

        except Exception as e:
            logger.error(f"Error fetching data from Coinbase: {e}")
            raise

    def get_available_symbols(self) -> list[str]:
        """Get list of available trading pairs from Coinbase."""
        try:
            response = self._make_request("GET", "/products")
            products = response.get("products", [])
            return [p["product_id"] for p in products if p.get("status") == "online"]
        except Exception as e:
            logger.error(f"Error fetching symbols from Coinbase: {e}")
            return []

    def test_connection(self) -> bool:
        """Test if API connection works."""
        try:
            # Try to get available symbols as a connection test
            symbols = self.get_available_symbols()
            return len(symbols) > 0
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
