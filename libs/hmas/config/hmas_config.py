"""
HMAS Configuration
Loads all API keys and system parameters from environment variables
"""
import os
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class HMASConfig:
    """HMAS System Configuration"""

    # API Keys
    xai_api_key: str
    anthropic_api_key: str
    google_api_key: str
    deepseek_api_key: str

    # Trading Parameters
    account_balance: float
    risk_per_trade: float
    target_win_rate: float
    symbols: List[str]

    # Strategy Parameters
    timeframe_primary: str = "M15"
    timeframe_secondary: str = "M30"
    ma200_period: int = 200
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    bbands_period: int = 20
    bbands_std: float = 2.0

    # FTMO Risk Management
    ftmo_daily_loss_limit: float = 0.045  # 4.5%
    ftmo_max_loss_limit: float = 0.09     # 9.0%
    ftmo_max_drawdown: float = 0.10       # 10%

    # ALM (Aggressive Loss Management)
    alm_atr_multiplier: float = 1.0       # Exit if 1× ATR against position
    alm_monitoring_interval: float = 1.0  # Check every 1 second

    # Cost Thresholds
    max_spread_pips: float = 3.0
    max_cost_to_tp_ratio: float = 0.50    # Cost must be < 50% of TP

    # Performance Targets
    min_risk_reward: float = 2.0
    target_trades_per_day: int = 3
    max_trades_per_day: int = 5

    @classmethod
    def from_env(cls) -> 'HMASConfig':
        """Load configuration from environment variables"""

        # Validate API keys
        xai_key = os.getenv('XAI_API_KEY', '')
        anthropic_key = os.getenv('ANTHROPIC_API_KEY', '')
        google_key = os.getenv('GOOGLE_API_KEY', '')
        deepseek_key = os.getenv('DEEPSEEK_API_KEY', '')

        if not xai_key or xai_key == 'your_xai_grok_key_here':
            raise ValueError("XAI_API_KEY not set in .env file")
        if not anthropic_key or anthropic_key == 'your_anthropic_key_here':
            raise ValueError("ANTHROPIC_API_KEY not set in .env file")
        if not google_key or google_key == 'your_google_gemini_key_here':
            raise ValueError("GOOGLE_API_KEY not set in .env file")
        if not deepseek_key:
            raise ValueError("DEEPSEEK_API_KEY not set in .env file")

        # Load trading parameters
        account_balance = float(os.getenv('HMAS_ACCOUNT_BALANCE', '10000'))
        risk_per_trade = float(os.getenv('HMAS_RISK_PER_TRADE', '0.01'))
        target_win_rate = float(os.getenv('HMAS_TARGET_WIN_RATE', '0.80'))

        # Load symbols
        symbols_str = os.getenv('HMAS_SYMBOLS', 'EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD')
        symbols = [s.strip() for s in symbols_str.split(',')]

        return cls(
            xai_api_key=xai_key,
            anthropic_api_key=anthropic_key,
            google_api_key=google_key,
            deepseek_api_key=deepseek_key,
            account_balance=account_balance,
            risk_per_trade=risk_per_trade,
            target_win_rate=target_win_rate,
            symbols=symbols
        )

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert 0 < self.risk_per_trade <= 0.02, "Risk per trade must be between 0% and 2%"
        assert 0.5 <= self.target_win_rate <= 1.0, "Target win rate must be between 50% and 100%"
        assert self.account_balance > 0, "Account balance must be positive"
        assert len(self.symbols) > 0, "At least one symbol must be specified"
        assert self.ftmo_daily_loss_limit < self.ftmo_max_loss_limit, "Daily loss < Max loss"

    def __repr__(self) -> str:
        return f"""HMASConfig(
  Account: ${self.account_balance:,.2f}
  Risk/Trade: {self.risk_per_trade:.1%}
  Target WR: {self.target_win_rate:.0%}
  Symbols: {', '.join(self.symbols)}
  FTMO Limits: Daily {self.ftmo_daily_loss_limit:.1%} / Max {self.ftmo_max_loss_limit:.1%}
)"""


# Global configuration instance
_config = None

def get_config() -> HMASConfig:
    """Get or create global HMAS configuration"""
    global _config
    if _config is None:
        _config = HMASConfig.from_env()
        _config.validate()
    return _config
