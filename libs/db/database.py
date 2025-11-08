"""Database connection and session management."""
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from loguru import logger

from libs.config.config import Settings
from libs.db.models import Base


class Database:
    """Database connection manager."""

    def __init__(self, db_url: str | None = None):
        """
        Initialize database connection.

        Args:
            db_url: Database URL (default: from config)
        """
        config = Settings()
        self.db_url = db_url or config.db_url

        # SQLite-specific configuration
        if self.db_url.startswith("sqlite"):
            # Use StaticPool for SQLite to avoid connection issues
            self.engine = create_engine(
                self.db_url,
                poolclass=StaticPool,
                connect_args={"check_same_thread": False},
                echo=False,
            )
        else:
            self.engine = create_engine(self.db_url, echo=False)

        # Create session factory
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        logger.info(f"Database initialized: {self.db_url}")

    def create_tables(self) -> None:
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created")

    def get_session(self) -> Generator[Session, None, None]:
        """
        Get database session (context manager).

        Yields:
            Database session
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_session_direct(self) -> Session:
        """
        Get database session directly (manual management).

        Returns:
            Database session
        """
        return self.SessionLocal()


# Global database instance
_db: Database | None = None


def get_database(db_url: str | None = None) -> Database:
    """
    Get global database instance.

    Args:
        db_url: Database URL (optional)

    Returns:
        Database instance
    """
    global _db
    if _db is None:
        _db = Database(db_url)
    return _db


def init_database(db_url: str | None = None, create_tables: bool = True) -> Database:
    """
    Initialize database and create tables.

    Args:
        db_url: Database URL (optional)
        create_tables: Whether to create tables (default: True)

    Returns:
        Database instance
    """
    db = get_database(db_url)
    if create_tables:
        db.create_tables()
    return db

