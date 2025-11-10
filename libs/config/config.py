"""Pydantic-based configuration system with schema validation."""

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BinanceConfig(BaseModel):
    """Binance API configuration."""

    api_key: str = Field(..., description="Binance API key")
    api_secret: str = Field(..., description="Binance API secret")


class TelegramConfig(BaseModel):
    """Telegram bot configuration."""

    token: str = Field(..., description="Telegram bot token")
    chat_id: str = Field(..., description="Telegram chat ID")


class FTMOConfig(BaseModel):
    """FTMO account configuration."""

    login: str = Field(..., description="FTMO login")
    password: str = Field(..., description="FTMO password")
    server: str = Field(..., description="FTMO server")


class SafetyConfig(BaseModel):
    """Safety rail configuration."""

    kill_switch: bool = Field(default=False, description="Kill-switch enabled")
    max_signals_per_hour: int = Field(default=10, ge=1, description="Max signals per hour")
    max_signals_per_hour_high: int = Field(
        default=5, ge=1, description="Max HIGH tier signals per hour"
    )
    latency_budget_ms: int = Field(default=500, ge=0, description="Latency budget in milliseconds")


class ModelConfig(BaseModel):
    """Model configuration."""

    version: str = Field(default="v1.0.0", description="Model version")
    model_path: str = Field(default="models/promoted/", description="Path to promoted models")


class EnsembleWeights(BaseModel):
    """Ensemble model weights."""

    lstm: float = Field(default=0.35, ge=0.0, le=1.0)
    transformer: float = Field(default=0.40, ge=0.0, le=1.0)
    rl: float = Field(default=0.25, ge=0.0, le=1.0)

    @field_validator("*")
    @classmethod
    def validate_weights(cls, v: float) -> float:
        """Ensure weights are valid."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Weight must be between 0.0 and 1.0, got {v}")
        return v

    def normalize(self) -> "EnsembleWeights":
        """Normalize weights to sum to 1.0."""
        total = self.lstm + self.transformer + self.rl
        if total == 0:
            # Fallback: 50/50 LSTM/Transformer if RL is 0
            return EnsembleWeights(lstm=0.5, transformer=0.5, rl=0.0)
        return EnsembleWeights(
            lstm=self.lstm / total, transformer=self.transformer / total, rl=self.rl / total
        )


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Data Provider (supports multiple: coinbase, kraken, cryptocompare, binance)
    data_provider: str = Field(
        default="coinbase", description="Data provider: coinbase, kraken, cryptocompare, binance"
    )
    data_provider_api_key: str = ""
    data_provider_api_secret: str = ""
    data_provider_api_passphrase: str = ""  # For Coinbase

    # Provider-specific env vars (also supported for convenience)
    # Coinbase Advanced Trade API uses JWT authentication
    coinbase_api_key_name: str = ""  # Full path: organizations/.../apiKeys/...
    coinbase_api_private_key: str = ""  # PEM-encoded EC private key
    # Legacy fields (for backward compatibility, but not used with JWT)
    coinbase_api_key: str = ""
    coinbase_api_secret: str = ""
    coinbase_api_passphrase: str = ""
    kraken_api_key: str = ""
    kraken_api_secret: str = ""
    cryptocompare_api_key: str = ""

    # Legacy Binance support (for backward compatibility)
    binance_api_key: str = ""
    binance_api_secret: str = ""

    @property
    def effective_api_key(self) -> str:
        """Get API key from provider-specific or generic field."""
        # For Coinbase Advanced Trade, use the full API key name path
        if self.data_provider == "coinbase" and self.coinbase_api_key_name:
            return self.coinbase_api_key_name
        if self.data_provider_api_key:
            return self.data_provider_api_key
        if self.data_provider == "coinbase" and self.coinbase_api_key:
            return self.coinbase_api_key  # Legacy fallback
        if self.data_provider == "kraken" and self.kraken_api_key:
            return self.kraken_api_key
        if self.data_provider == "cryptocompare" and self.cryptocompare_api_key:
            return self.cryptocompare_api_key
        if self.binance_api_key:  # Legacy support
            return self.binance_api_key
        return ""

    @property
    def effective_api_secret(self) -> str:
        """Get API secret from provider-specific or generic field."""
        # For Coinbase Advanced Trade, use the private key (PEM)
        if self.data_provider == "coinbase" and self.coinbase_api_private_key:
            return self.coinbase_api_private_key
        if self.data_provider_api_secret:
            return self.data_provider_api_secret
        if self.data_provider == "coinbase" and self.coinbase_api_secret:
            return self.coinbase_api_secret  # Legacy fallback
        if self.data_provider == "kraken" and self.kraken_api_secret:
            return self.kraken_api_secret
        if self.binance_api_secret:  # Legacy support
            return self.binance_api_secret
        return ""

    @property
    def effective_api_passphrase(self) -> str:
        """Get API passphrase (not used for Coinbase Advanced Trade JWT)."""
        # Not used with JWT authentication, but kept for backward compatibility
        if self.data_provider_api_passphrase:
            return self.data_provider_api_passphrase
        if self.data_provider == "coinbase" and self.coinbase_api_passphrase:
            return self.coinbase_api_passphrase
        return ""

    # Telegram
    telegram_token: str = ""
    telegram_chat_id: str = ""

    # FTMO
    ftmo_login: str = ""
    ftmo_pass: str = ""
    ftmo_server: str = ""

    # Database
    db_url: str = "sqlite:///tradingai.db"

    # Confidence
    confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    ensemble_weights: str = "0.35,0.40,0.25"  # LSTM, Transformer, RL

    # Safety
    kill_switch: bool = False
    max_signals_per_hour: int = 10
    max_signals_per_hour_high: int = 5
    latency_budget_ms: int = 500

    # Model
    model_version: str = "v1.0.0"
    model_path: str = "models/promoted/"

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # json or text
    runtime_mode: str = Field(default="dryrun", description="Runtime mode: dryrun or live")

    @property
    def ensemble_weights_parsed(self) -> EnsembleWeights:
        """Parse ensemble weights from string."""
        try:
            weights = [float(w.strip()) for w in self.ensemble_weights.split(",")]
            if len(weights) != 3:
                raise ValueError("Expected 3 weights")
            return EnsembleWeights(
                lstm=weights[0], transformer=weights[1], rl=weights[2]
            ).normalize()
        except (ValueError, IndexError):
            # Fallback to 50/50 if parsing fails
            return EnsembleWeights(lstm=0.5, transformer=0.5, rl=0.0).normalize()

    def validate(self) -> None:
        """Validate settings and fail fast on critical errors."""
        errors = []

        # Critical validations
        if self.confidence_threshold < 0.5:
            errors.append("Confidence threshold too low (minimum 0.5)")

        if self.max_signals_per_hour_high > self.max_signals_per_hour:
            errors.append("High-tier signal limit cannot exceed general limit")

        if self.runtime_mode.lower() not in {"dryrun", "dry-run", "live"}:
            errors.append("Runtime mode must be 'dryrun' or 'live'")

        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")


def load_settings(config_file: str | None = None) -> Settings:
    """Load settings from environment or config file."""
    settings = Settings(_env_file=config_file)
    settings.validate()
    return settings
