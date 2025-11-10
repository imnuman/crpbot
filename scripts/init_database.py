#!/usr/bin/env python3
"""Initialize database and create tables."""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from libs.config.config import Settings
from libs.db.database import init_database


def main():
    """Initialize database."""
    config = Settings()
    logger.info(f"Initializing database: {config.db_url}")

    # Initialize database and create tables
    init_database(db_url=config.db_url, create_tables=True)

    logger.info("✅ Database initialized successfully!")
    logger.info("   Tables created:")
    logger.info("   - patterns")
    logger.info("   - risk_book_snapshots")
    logger.info("   - model_deployments")


if __name__ == "__main__":
    main()
