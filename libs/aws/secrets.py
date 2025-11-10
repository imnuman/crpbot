"""Secrets management for CRPBot using AWS Secrets Manager."""

import json
import os
from typing import Any

import boto3


class SecretsManager:
    """Manage secrets using AWS Secrets Manager with env fallback."""

    def __init__(self):
        self.secrets_client = boto3.client("secretsmanager")
        self._cache: dict[str, dict[str, Any]] = {}

    def _get_secret(self, secret_arn: str) -> dict[str, Any]:
        """Get secret from AWS Secrets Manager."""
        if secret_arn in self._cache:
            return self._cache[secret_arn]
        try:
            response = self.secrets_client.get_secret_value(SecretId=secret_arn)
            secret = json.loads(response["SecretString"])
            self._cache[secret_arn] = secret
            return secret
        except Exception:
            return {}

    def get_coinbase_credentials(self) -> dict[str, str]:
        """Get Coinbase Advanced Trade API credentials."""
        secret_arn = os.getenv("COINBASE_SECRET_ARN")
        if secret_arn:
            secret = self._get_secret(secret_arn)
            return {
                "api_key": secret.get("api_key", ""),  # API Key Name
                "private_key": secret.get("private_key", ""),  # Private Key PEM
            }
        # Fallback to env vars
        return {
            "api_key": os.getenv("COINBASE_API_KEY", ""),
            "private_key": os.getenv("COINBASE_API_SECRET", ""),  # Legacy mapping
        }

    def get_telegram_credentials(self) -> dict[str, str]:
        """Get Telegram bot credentials."""
        secret_arn = os.getenv("TELEGRAM_SECRET_ARN")
        if secret_arn:
            secret = self._get_secret(secret_arn)
            return {"bot_token": secret.get("bot_token", ""), "chat_id": secret.get("chat_id", "")}
        # Fallback to env vars
        return {
            "bot_token": os.getenv("TELEGRAM_TOKEN", ""),
            "chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
        }

    def get_ftmo_credentials(self) -> dict[str, str]:
        """Get FTMO account credentials."""
        secret_arn = os.getenv("FTMO_SECRET_ARN")
        if secret_arn:
            secret = self._get_secret(secret_arn)
            return {
                "login": secret.get("login", ""),
                "password": secret.get("password", ""),
                "server": secret.get("server", ""),
            }
        # Fallback to env vars
        return {
            "login": os.getenv("FTMO_LOGIN", ""),
            "password": os.getenv("FTMO_PASS", ""),
            "server": os.getenv("FTMO_SERVER", ""),
        }

    def get_database_url(self) -> str:
        """Get database connection URL."""
        # Check for RDS connection
        db_host = os.getenv("DB_HOST")
        if db_host:
            db_user = os.getenv("DB_USERNAME", "crpbot_admin")
            db_pass = os.getenv("DB_PASSWORD", "TempPassword123!")
            db_name = os.getenv("DB_NAME", "postgres")
            db_port = os.getenv("DB_PORT", "5432")
            return f"postgresql+psycopg://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

        # Fallback to existing DB_URL or SQLite
        return os.getenv("DB_URL", "sqlite:///tradingai.db")
