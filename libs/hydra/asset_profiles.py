"""
HYDRA 3.0 - Asset Profiles (Upgrade B)

Market-specific configurations for all niche markets.
Each asset has unique characteristics that require custom handling.

Exotic forex and meme perps behave VERY differently from BTC/EUR/USD.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class AssetProfile:
    """
    Complete profile for a single asset.
    """
    asset: str
    type: str  # "exotic_forex" or "meme_perp" or "standard"

    # Spread thresholds
    spread_normal: float  # Normal spread in pips (forex) or % (crypto)
    spread_reject_multiplier: float  # Reject if spread > normal * multiplier

    # Position sizing
    size_modifier: float  # Reduce position size (0.5 = 50%, 0.3 = 30%)

    # Time restrictions
    overnight_allowed: bool  # Can hold overnight?
    max_hold_hours: Optional[int]  # Max hours to hold (None = unlimited)
    best_sessions: List[str]  # Best trading sessions

    # Risk level
    manipulation_risk: str  # "LOW", "MEDIUM", "HIGH", "EXTREME"

    # Crypto-specific
    funding_threshold: Optional[float]  # Funding rate threshold for extreme
    whale_threshold: Optional[float]  # Whale movement threshold (USD)

    # Special rules
    special_rules: List[str]  # Human-readable special handling

    # Market-specific notes
    notes: str


class AssetProfileManager:
    """
    Manages all asset profiles.

    Provides market-specific configurations for Guardian and filters.
    """

    def __init__(self):
        self.profiles = self._load_all_profiles()
        logger.info(f"Asset profiles loaded: {len(self.profiles)} markets")

    def get_profile(self, asset: str) -> AssetProfile:
        """Get profile for asset, raise if not found."""
        if asset not in self.profiles:
            raise ValueError(f"No profile found for {asset}. Available: {list(self.profiles.keys())}")
        return self.profiles[asset]

    def get_all_exotic_forex(self) -> List[AssetProfile]:
        """Get all exotic forex profiles."""
        return [p for p in self.profiles.values() if p.type == "exotic_forex"]

    def get_all_meme_perps(self) -> List[AssetProfile]:
        """Get all meme perp profiles."""
        return [p for p in self.profiles.values() if p.type == "meme_perp"]

    def is_tradeable_now(self, asset: str, current_hour_utc: int) -> bool:
        """
        Check if asset should be traded at current hour.

        Args:
            asset: Asset symbol
            current_hour_utc: Current hour in UTC (0-23)

        Returns:
            True if tradeable, False if outside best sessions
        """
        profile = self.get_profile(asset)

        if profile.type == "meme_perp":
            # Crypto trades 24/7, but check best sessions for liquidity
            # Asia session (12AM-8AM UTC) is best for low-cap alts
            if "Asia" in profile.best_sessions:
                return 0 <= current_hour_utc < 8
            return True  # Otherwise 24/7

        if profile.type == "exotic_forex":
            # London: 8AM-4PM UTC
            # NY: 1PM-9PM UTC
            # Overlap: 1PM-4PM UTC (best liquidity)

            if "London" in profile.best_sessions and 8 <= current_hour_utc < 16:
                return True
            if "NY" in profile.best_sessions and 13 <= current_hour_utc < 21:
                return True

            return False

        return True  # Standard assets trade normally

    def _load_all_profiles(self) -> Dict[str, AssetProfile]:
        """Load all 12 niche market profiles."""
        profiles = {}

        # ==================== EXOTIC FOREX ====================

        # USD/TRY - Turkish Lira
        profiles["USD/TRY"] = AssetProfile(
            asset="USD/TRY",
            type="exotic_forex",
            spread_normal=20.0,  # 20 pips normal
            spread_reject_multiplier=3.0,  # Reject if >60 pips
            size_modifier=0.5,  # 50% of normal position
            overnight_allowed=False,  # NO overnight (gap risk)
            max_hold_hours=8,  # Max 8 hours
            best_sessions=["London", "NY"],
            manipulation_risk="HIGH",
            funding_threshold=None,
            whale_threshold=None,
            special_rules=[
                "Avoid 24 hours before Turkish central bank meetings",
                "Avoid during Erdogan speeches (unpredictable)",
                "Extremely high gap risk on weekends",
                "Political events cause massive spikes",
                "CB intervention common"
            ],
            notes="Highly volatile EM currency. Central bank surprises are common. "
                  "Used for session open volatility strategies."
        )

        # USD/ZAR - South African Rand
        profiles["USD/ZAR"] = AssetProfile(
            asset="USD/ZAR",
            type="exotic_forex",
            spread_normal=25.0,
            spread_reject_multiplier=3.0,
            size_modifier=0.5,
            overnight_allowed=False,
            max_hold_hours=8,
            best_sessions=["London"],
            manipulation_risk="HIGH",
            funding_threshold=None,
            whale_threshold=None,
            special_rules=[
                "Follows gold price closely (SA is gold exporter)",
                "Power outages (loadshedding) affect sentiment",
                "Avoid during SARB meetings",
                "Risk-off moves hit ZAR hard"
            ],
            notes="Commodity currency linked to gold and platinum. "
                  "Sensitive to emerging market sentiment."
        )

        # USD/MXN - Mexican Peso
        profiles["USD/MXN"] = AssetProfile(
            asset="USD/MXN",
            type="exotic_forex",
            spread_normal=15.0,
            spread_reject_multiplier=3.0,
            size_modifier=0.5,
            overnight_allowed=False,
            max_hold_hours=8,
            best_sessions=["NY"],
            manipulation_risk="MEDIUM",
            funding_threshold=None,
            whale_threshold=None,
            special_rules=[
                "MXN leads other EM currencies (TRY, ZAR lag)",
                "Oil price correlation (Mexico is oil exporter)",
                "US-Mexico trade policy affects sentiment",
                "Banxico meetings = volatility"
            ],
            notes="EM leader - often moves before TRY/ZAR. "
                  "More liquid than TRY/ZAR but still risky."
        )

        # EUR/TRY - Euro/Turkish Lira
        profiles["EUR/TRY"] = AssetProfile(
            asset="EUR/TRY",
            type="exotic_forex",
            spread_normal=30.0,  # Wider spread (double exotic)
            spread_reject_multiplier=3.0,
            size_modifier=0.4,  # Even smaller (40%)
            overnight_allowed=False,
            max_hold_hours=6,
            best_sessions=["London"],
            manipulation_risk="EXTREME",
            funding_threshold=None,
            whale_threshold=None,
            special_rules=[
                "Double exotic = extreme volatility",
                "Avoid completely during Turkish CB events",
                "ECB + Turkish CB = double risk",
                "Very thin liquidity"
            ],
            notes="Extreme risk pair. Only for very high conviction trades. "
                  "Spreads can spike to 100+ pips instantly."
        )

        # USD/PLN - Polish Zloty
        profiles["USD/PLN"] = AssetProfile(
            asset="USD/PLN",
            type="exotic_forex",
            spread_normal=12.0,
            spread_reject_multiplier=3.0,
            size_modifier=0.5,
            overnight_allowed=False,
            max_hold_hours=8,
            best_sessions=["London"],
            manipulation_risk="MEDIUM",
            funding_threshold=None,
            whale_threshold=None,
            special_rules=[
                "EU member but own currency (pre-Euro)",
                "Follows EUR/USD trends but with lag",
                "NBP (Polish CB) interventions rare but impactful",
                "Low volume outside London hours"
            ],
            notes="More stable than TRY/ZAR but still exotic. "
                  "Patterns persist longer due to low algorithmic trading."
        )

        # USD/NOK - Norwegian Krone
        profiles["USD/NOK"] = AssetProfile(
            asset="USD/NOK",
            type="exotic_forex",
            spread_normal=10.0,
            spread_reject_multiplier=3.0,
            size_modifier=0.5,
            overnight_allowed=False,
            max_hold_hours=8,
            best_sessions=["London"],
            manipulation_risk="LOW",
            funding_threshold=None,
            whale_threshold=None,
            special_rules=[
                "Strong correlation with oil prices (Norway = oil exporter)",
                "Norges Bank very predictable (low surprise risk)",
                "Safe-haven status in Scandinavia",
                "Seasonal patterns (oil demand)"
            ],
            notes="Safest exotic on the list. Oil correlation is strong edge. "
                  "Good for carry trade strategies."
        )

        # ==================== MEME PERPS ====================

        # BONK - Solana meme coin
        profiles["BONK"] = AssetProfile(
            asset="BONK",
            type="meme_perp",
            spread_normal=0.0005,  # 0.05% spread
            spread_reject_multiplier=3.0,
            size_modifier=0.3,  # 30% size (very risky)
            overnight_allowed=True,  # Can hold overnight (no gap risk in crypto)
            max_hold_hours=4,  # But max 4 hours recommended
            best_sessions=["Asia"],  # Best liquidity during Asia hours
            manipulation_risk="EXTREME",
            funding_threshold=0.5,  # 0.5% funding is extreme
            whale_threshold=500000,  # $500k whale movement
            special_rules=[
                "Check Solana network health (outages kill BONK)",
                "Funding resets every 8 hours (0:00, 8:00, 16:00 UTC)",
                "Extremely thin liquidity outside Asia hours",
                "Correlates with SOL price (check cross-asset)",
                "Weekend pumps common, Monday dumps expected"
            ],
            notes="Extreme volatility meme. Funding rate arbitrage opportunities. "
                  "Whale manipulation is constant. Only trade with strict filters."
        )

        # WIF - Dogwifhat (Solana meme)
        profiles["WIF"] = AssetProfile(
            asset="WIF",
            type="meme_perp",
            spread_normal=0.0005,
            spread_reject_multiplier=3.0,
            size_modifier=0.3,
            overnight_allowed=True,
            max_hold_hours=4,
            best_sessions=["Asia"],
            manipulation_risk="EXTREME",
            funding_threshold=0.5,
            whale_threshold=300000,  # $300k (smaller market cap)
            special_rules=[
                "Follows Solana ecosystem sentiment",
                "Correlates with BONK (check for same-direction moves)",
                "Meme coin = social media driven",
                "Elon Musk tweets can move it 20%+",
                "Low float = easy to manipulate"
            ],
            notes="Sister coin to BONK. Social media sentiment is key. "
                  "Even more manipulated than BONK due to smaller size."
        )

        # PEPE - Pepe meme coin (Ethereum)
        profiles["PEPE"] = AssetProfile(
            asset="PEPE",
            type="meme_perp",
            spread_normal=0.0006,
            spread_reject_multiplier=3.0,
            size_modifier=0.3,
            overnight_allowed=True,
            max_hold_hours=4,
            best_sessions=["Asia", "NY"],
            manipulation_risk="EXTREME",
            funding_threshold=0.6,  # 0.6% (less liquid)
            whale_threshold=400000,
            special_rules=[
                "Ethereum-based (check ETH gas fees)",
                "Meme cycle leader (pumps first, others follow)",
                "4chan/Reddit sentiment drives moves",
                "Liquidation cascades common (high leverage)",
                "CEX listings = massive pumps"
            ],
            notes="OG meme coin on ETH. Social sentiment is everything. "
                  "Watch for exchange listing announcements."
        )

        # FLOKI - Floki Inu
        profiles["FLOKI"] = AssetProfile(
            asset="FLOKI",
            type="meme_perp",
            spread_normal=0.0007,
            spread_reject_multiplier=3.0,
            size_modifier=0.3,
            overnight_allowed=True,
            max_hold_hours=4,
            best_sessions=["Asia"],
            manipulation_risk="EXTREME",
            funding_threshold=0.6,
            whale_threshold=250000,
            special_rules=[
                "Elon Musk dog references = pumps",
                "Follows DOGE/SHIB trends with lag",
                "Marketing campaigns = predictable pumps",
                "Utility announcements = brief pumps then dumps"
            ],
            notes="Elon-related meme. Predictable patterns around his tweets. "
                  "Lower liquidity than BONK/WIF."
        )

        # SUI - Sui blockchain (mid-cap)
        profiles["SUI"] = AssetProfile(
            asset="SUI",
            type="meme_perp",  # Treated as meme due to hype
            spread_normal=0.0003,
            spread_reject_multiplier=3.0,
            size_modifier=0.4,  # Slightly less risky (40%)
            overnight_allowed=True,
            max_hold_hours=8,  # Can hold longer (more legitimate)
            best_sessions=["Asia", "London"],
            manipulation_risk="HIGH",
            funding_threshold=0.3,
            whale_threshold=1000000,  # $1M (larger market cap)
            special_rules=[
                "New L1 (launched 2023) - still finding price",
                "Ecosystem growth = pumps (new dApps)",
                "Token unlocks = dumps (check schedule)",
                "Competes with Solana/Aptos narratives"
            ],
            notes="Newer L1 with real tech but still hype-driven. "
                  "More stable than pure memes but still volatile."
        )

        # INJ - Injective Protocol
        profiles["INJ"] = AssetProfile(
            asset="INJ",
            type="meme_perp",
            spread_normal=0.0004,
            spread_reject_multiplier=3.0,
            size_modifier=0.4,
            overnight_allowed=True,
            max_hold_hours=8,
            best_sessions=["Asia", "London"],
            manipulation_risk="HIGH",
            funding_threshold=0.3,
            whale_threshold=800000,
            special_rules=[
                "DeFi protocol token (real utility)",
                "Funding rate swings create arb opportunities",
                "Ecosystem announcements = pumps",
                "Competes with dYdX narrative"
            ],
            notes="DeFi mid-cap with funding rate volatility. "
                  "More predictable than pure memes."
        )

        return profiles


# Global singleton instance
_asset_profile_manager = None

def get_asset_profile_manager() -> AssetProfileManager:
    """Get global AssetProfileManager singleton."""
    global _asset_profile_manager
    if _asset_profile_manager is None:
        _asset_profile_manager = AssetProfileManager()
    return _asset_profile_manager
