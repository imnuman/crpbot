"""
HYDRA Configuration - Central path configuration

Uses environment variable HYDRA_DATA_DIR or falls back to default.
This enables the same code to run on bare metal or in Docker.

Environment variables:
- HYDRA_DATA_DIR: Base data directory (default: auto-detected)
- HYDRA_PROJECT_ROOT: Project root (default: auto-detected)
"""

import os
from pathlib import Path


def _detect_project_root() -> Path:
    """Detect project root dynamically."""
    # Check environment variable first
    if env_root := os.environ.get("HYDRA_PROJECT_ROOT"):
        return Path(env_root)

    # Try to find project root from this file's location
    # This file is at libs/hydra/config.py, so go up 2 levels
    this_file = Path(__file__).resolve()
    potential_root = this_file.parent.parent.parent

    # Verify it's the project root by checking for pyproject.toml
    if (potential_root / "pyproject.toml").exists():
        return potential_root

    # Fallback: check common locations
    for path in ["/app", "/root/crpbot", Path.home() / "crpbot"]:
        p = Path(path)
        if p.exists() and (p / "pyproject.toml").exists():
            return p

    # Last resort: use current working directory
    return Path.cwd()


def _detect_data_dir() -> Path:
    """Detect data directory dynamically."""
    # Check environment variable first
    if env_dir := os.environ.get("HYDRA_DATA_DIR"):
        return Path(env_dir)

    # Use project root + data/hydra
    return PROJECT_ROOT / "data" / "hydra"


# Project root detection
PROJECT_ROOT = _detect_project_root()

# Data directory
HYDRA_DATA_DIR = _detect_data_dir()

# Ensure directory exists
HYDRA_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ==================== Data Files ====================
PAPER_TRADES_FILE = HYDRA_DATA_DIR / "paper_trades.jsonl"
TOURNAMENT_VOTES_FILE = HYDRA_DATA_DIR / "tournament_votes.jsonl"
TOURNAMENT_SCORES_FILE = HYDRA_DATA_DIR / "tournament_scores.jsonl"
MOTHER_AI_STATE_FILE = HYDRA_DATA_DIR / "mother_ai_state.json"
GUARDIAN_STATE_FILE = HYDRA_DATA_DIR / "guardian_state.json"
RUNTIME_CHECKPOINT_FILE = HYDRA_DATA_DIR / "runtime_checkpoint.json"
LESSONS_FILE = HYDRA_DATA_DIR / "lessons.jsonl"
CHAT_HISTORY_FILE = HYDRA_DATA_DIR / "chat_history.jsonl"
USER_FEEDBACK_FILE = HYDRA_DATA_DIR / "user_feedback.jsonl"
RECOMMENDATIONS_FILE = HYDRA_DATA_DIR / "recommendations.jsonl"

# ==================== Database Files ====================
HYDRA_DB_FILE = HYDRA_DATA_DIR / "hydra.db"
HISTORICAL_30D_DB_FILE = HYDRA_DATA_DIR / "historical_30d.db"
TRADINGAI_DB_FILE = PROJECT_ROOT / "tradingai.db"

# ==================== Cache Directories ====================
SEARCH_CACHE_DIR = HYDRA_DATA_DIR / "search_cache"
FUNDING_CACHE_DIR = HYDRA_DATA_DIR / "funding_cache"
LIQUIDATIONS_CACHE_DIR = HYDRA_DATA_DIR / "liquidations_cache"

# ==================== Other Directories ====================
EXPLAINABILITY_DIR = HYDRA_DATA_DIR / "explainability"
STRATEGIES_DIR = HYDRA_DATA_DIR / "strategies"
TOURNAMENT_RESULTS_DIR = HYDRA_DATA_DIR / "tournament_results"
LESSONS_DIR = HYDRA_DATA_DIR / "lessons"

# ==================== Ensure Directories Exist ====================
_all_dirs = [
    SEARCH_CACHE_DIR,
    FUNDING_CACHE_DIR,
    LIQUIDATIONS_CACHE_DIR,
    EXPLAINABILITY_DIR,
    STRATEGIES_DIR,
    TOURNAMENT_RESULTS_DIR,
    LESSONS_DIR,
]

for _dir in _all_dirs:
    _dir.mkdir(parents=True, exist_ok=True)
