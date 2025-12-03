"""
HYDRA Configuration - Central path configuration

Uses environment variable HYDRA_DATA_DIR or falls back to default.
This enables the same code to run on bare metal or in Docker.
"""

import os
from pathlib import Path

# Get data directory from environment or use default
# Docker sets HYDRA_DATA_DIR=/app/data/hydra
# Bare metal uses /root/crpbot/data/hydra
_default_data_dir = "/root/crpbot/data/hydra"
HYDRA_DATA_DIR = Path(os.environ.get("HYDRA_DATA_DIR", _default_data_dir))

# Ensure directory exists
HYDRA_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Common paths used throughout the codebase
PAPER_TRADES_FILE = HYDRA_DATA_DIR / "paper_trades.jsonl"
TOURNAMENT_VOTES_FILE = HYDRA_DATA_DIR / "tournament_votes.jsonl"
MOTHER_AI_STATE_FILE = HYDRA_DATA_DIR / "mother_ai_state.json"
GUARDIAN_STATE_FILE = HYDRA_DATA_DIR / "guardian_state.json"
LESSONS_FILE = HYDRA_DATA_DIR / "lessons.jsonl"
CHAT_HISTORY_FILE = HYDRA_DATA_DIR / "chat_history.jsonl"

# Cache directories
SEARCH_CACHE_DIR = HYDRA_DATA_DIR / "search_cache"
FUNDING_CACHE_DIR = HYDRA_DATA_DIR / "funding_cache"
LIQUIDATIONS_CACHE_DIR = HYDRA_DATA_DIR / "liquidations_cache"

# Ensure cache directories exist
for cache_dir in [SEARCH_CACHE_DIR, FUNDING_CACHE_DIR, LIQUIDATIONS_CACHE_DIR]:
    cache_dir.mkdir(parents=True, exist_ok=True)
