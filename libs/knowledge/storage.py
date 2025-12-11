"""
Storage layer for Knowledge Aggregation System.

Supports:
- SQLite (default, lightweight)
- PostgreSQL (production, optional)
- ChromaDB (vector embeddings)
"""

import os
import sqlite3
import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from loguru import logger

from .base import (
    KnowledgeItem,
    EconomicEvent,
    CodeFile,
    ScrapeLog,
    KnowledgeSource,
    ContentType,
    ImpactLevel,
)

# Default storage paths
DEFAULT_DB_PATH = Path("/root/crpbot/data/hydra/knowledge.db")
DEFAULT_CHROMA_PATH = Path("/root/crpbot/data/hydra/chroma")


class KnowledgeStorage:
    """SQLite-based storage for knowledge items."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _get_conn(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        """Initialize database schema."""
        schema = """
        -- Knowledge items (strategies, indicators, articles)
        CREATE TABLE IF NOT EXISTS knowledge_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            source_url TEXT,
            title TEXT NOT NULL,
            content_type TEXT,
            summary TEXT,
            full_content TEXT,
            tags TEXT,  -- JSON array
            symbols TEXT,  -- JSON array
            timeframes TEXT,  -- JSON array
            win_rate REAL,
            risk_reward REAL,
            author TEXT,
            quality_score REAL,
            embedding_id TEXT,
            upvotes INTEGER,
            comments_count INTEGER,
            source_created_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            content_hash TEXT UNIQUE
        );

        -- Economic calendar events
        CREATE TABLE IF NOT EXISTS economic_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_date TEXT NOT NULL,
            currency TEXT,
            event_name TEXT,
            impact TEXT,
            previous TEXT,
            forecast TEXT,
            actual TEXT,
            source TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        -- EA/Indicator code files
        CREATE TABLE IF NOT EXISTS code_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            knowledge_item_id INTEGER REFERENCES knowledge_items(id),
            filename TEXT,
            language TEXT,
            content TEXT,
            extracted_params TEXT,  -- JSON
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        -- Scraping job logs
        CREATE TABLE IF NOT EXISTS scrape_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            job_type TEXT,
            started_at TEXT,
            completed_at TEXT,
            items_found INTEGER DEFAULT 0,
            items_new INTEGER DEFAULT 0,
            error_message TEXT,
            status TEXT DEFAULT 'running'
        );

        -- Indexes
        CREATE INDEX IF NOT EXISTS idx_knowledge_source ON knowledge_items(source);
        CREATE INDEX IF NOT EXISTS idx_knowledge_type ON knowledge_items(content_type);
        CREATE INDEX IF NOT EXISTS idx_knowledge_hash ON knowledge_items(content_hash);
        CREATE INDEX IF NOT EXISTS idx_events_date ON economic_events(event_date);
        CREATE INDEX IF NOT EXISTS idx_events_currency ON economic_events(currency);
        """

        with self._get_conn() as conn:
            conn.executescript(schema)

    # ==================== Knowledge Items ====================

    def save_item(self, item: KnowledgeItem) -> int:
        """Save or update a knowledge item. Returns item ID."""
        content_hash = item.get_content_hash()

        # Check if exists
        existing = self.get_item_by_hash(content_hash)
        if existing:
            # Update existing
            return self._update_item(existing.id, item)

        # Insert new
        return self._insert_item(item, content_hash)

    def _insert_item(self, item: KnowledgeItem, content_hash: str) -> int:
        """Insert new knowledge item."""
        sql = """
        INSERT INTO knowledge_items (
            source, source_url, title, content_type, summary, full_content,
            tags, symbols, timeframes, win_rate, risk_reward, author,
            quality_score, embedding_id, upvotes, comments_count,
            source_created_at, content_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        with self._get_conn() as conn:
            cursor = conn.execute(sql, (
                item.source.value,
                item.source_url,
                item.title,
                item.content_type.value,
                item.summary,
                item.full_content,
                json.dumps(item.tags),
                json.dumps(item.symbols),
                json.dumps(item.timeframes),
                item.win_rate,
                item.risk_reward,
                item.author,
                item.quality_score,
                item.embedding_id,
                item.upvotes,
                item.comments_count,
                item.source_created_at.isoformat() if item.source_created_at else None,
                content_hash,
            ))
            return cursor.lastrowid

    def _update_item(self, item_id: int, item: KnowledgeItem) -> int:
        """Update existing knowledge item."""
        sql = """
        UPDATE knowledge_items SET
            summary = ?,
            full_content = ?,
            tags = ?,
            symbols = ?,
            timeframes = ?,
            win_rate = ?,
            risk_reward = ?,
            quality_score = ?,
            embedding_id = ?,
            upvotes = ?,
            comments_count = ?,
            updated_at = ?
        WHERE id = ?
        """

        with self._get_conn() as conn:
            conn.execute(sql, (
                item.summary,
                item.full_content,
                json.dumps(item.tags),
                json.dumps(item.symbols),
                json.dumps(item.timeframes),
                item.win_rate,
                item.risk_reward,
                item.quality_score,
                item.embedding_id,
                item.upvotes,
                item.comments_count,
                datetime.utcnow().isoformat(),
                item_id,
            ))
            return item_id

    def get_item_by_hash(self, content_hash: str) -> Optional[KnowledgeItem]:
        """Get item by content hash."""
        sql = "SELECT * FROM knowledge_items WHERE content_hash = ?"

        with self._get_conn() as conn:
            row = conn.execute(sql, (content_hash,)).fetchone()
            if row:
                return self._row_to_item(row)
        return None

    def get_item_by_id(self, item_id: int) -> Optional[KnowledgeItem]:
        """Get item by ID."""
        sql = "SELECT * FROM knowledge_items WHERE id = ?"

        with self._get_conn() as conn:
            row = conn.execute(sql, (item_id,)).fetchone()
            if row:
                return self._row_to_item(row)
        return None

    def search_items(
        self,
        query: Optional[str] = None,
        source: Optional[KnowledgeSource] = None,
        content_type: Optional[ContentType] = None,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        min_quality: Optional[float] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[KnowledgeItem]:
        """Search knowledge items with filters."""
        conditions = []
        params = []

        if source:
            conditions.append("source = ?")
            params.append(source.value)

        if content_type:
            conditions.append("content_type = ?")
            params.append(content_type.value)

        if min_quality is not None:
            conditions.append("quality_score >= ?")
            params.append(min_quality)

        if query:
            conditions.append("(title LIKE ? OR summary LIKE ? OR full_content LIKE ?)")
            like_query = f"%{query}%"
            params.extend([like_query, like_query, like_query])

        if symbols:
            # Check if any symbol matches
            symbol_conditions = []
            for s in symbols:
                symbol_conditions.append("symbols LIKE ?")
                params.append(f'%"{s}"%')
            conditions.append(f"({' OR '.join(symbol_conditions)})")

        if timeframes:
            tf_conditions = []
            for tf in timeframes:
                tf_conditions.append("timeframes LIKE ?")
                params.append(f'%"{tf}"%')
            conditions.append(f"({' OR '.join(tf_conditions)})")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        sql = f"""
        SELECT * FROM knowledge_items
        WHERE {where_clause}
        ORDER BY quality_score DESC NULLS LAST, created_at DESC
        LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        with self._get_conn() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_item(row) for row in rows]

    def count_items(
        self,
        source: Optional[KnowledgeSource] = None,
        content_type: Optional[ContentType] = None,
    ) -> int:
        """Count items with optional filters."""
        conditions = []
        params = []

        if source:
            conditions.append("source = ?")
            params.append(source.value)

        if content_type:
            conditions.append("content_type = ?")
            params.append(content_type.value)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT COUNT(*) FROM knowledge_items WHERE {where_clause}"

        with self._get_conn() as conn:
            return conn.execute(sql, params).fetchone()[0]

    def get_items_without_embeddings(self, limit: int = 100) -> List[KnowledgeItem]:
        """Get items that need embeddings generated."""
        sql = """
        SELECT * FROM knowledge_items
        WHERE embedding_id IS NULL
        ORDER BY created_at DESC
        LIMIT ?
        """

        with self._get_conn() as conn:
            rows = conn.execute(sql, (limit,)).fetchall()
            return [self._row_to_item(row) for row in rows]

    def update_embedding_id(self, item_id: int, embedding_id: str):
        """Update the embedding ID for an item."""
        sql = "UPDATE knowledge_items SET embedding_id = ? WHERE id = ?"

        with self._get_conn() as conn:
            conn.execute(sql, (embedding_id, item_id))

    def _row_to_item(self, row: sqlite3.Row) -> KnowledgeItem:
        """Convert database row to KnowledgeItem."""
        return KnowledgeItem(
            id=row["id"],
            source=KnowledgeSource(row["source"]),
            source_url=row["source_url"],
            title=row["title"],
            content_type=ContentType(row["content_type"]) if row["content_type"] else ContentType.ARTICLE,
            summary=row["summary"],
            full_content=row["full_content"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            symbols=json.loads(row["symbols"]) if row["symbols"] else [],
            timeframes=json.loads(row["timeframes"]) if row["timeframes"] else [],
            win_rate=row["win_rate"],
            risk_reward=row["risk_reward"],
            author=row["author"],
            quality_score=row["quality_score"],
            embedding_id=row["embedding_id"],
            upvotes=row["upvotes"],
            comments_count=row["comments_count"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.utcnow(),
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else datetime.utcnow(),
        )

    # ==================== Economic Events ====================

    def save_event(self, event: EconomicEvent) -> int:
        """Save economic event."""
        sql = """
        INSERT INTO economic_events (
            event_date, currency, event_name, impact,
            previous, forecast, actual, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        with self._get_conn() as conn:
            cursor = conn.execute(sql, (
                event.event_date.isoformat(),
                event.currency,
                event.event_name,
                event.impact.value,
                event.previous,
                event.forecast,
                event.actual,
                event.source.value,
            ))
            return cursor.lastrowid

    def get_events(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        currency: Optional[str] = None,
        impact: Optional[ImpactLevel] = None,
    ) -> List[EconomicEvent]:
        """Get economic events with filters."""
        conditions = ["event_date >= ?"]
        params = [start_date.isoformat()]

        if end_date:
            conditions.append("event_date <= ?")
            params.append(end_date.isoformat())

        if currency:
            conditions.append("currency = ?")
            params.append(currency)

        if impact:
            conditions.append("impact = ?")
            params.append(impact.value)

        where_clause = " AND ".join(conditions)
        sql = f"SELECT * FROM economic_events WHERE {where_clause} ORDER BY event_date"

        with self._get_conn() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_event(row) for row in rows]

    def _row_to_event(self, row: sqlite3.Row) -> EconomicEvent:
        """Convert row to EconomicEvent."""
        return EconomicEvent(
            id=row["id"],
            event_date=datetime.fromisoformat(row["event_date"]),
            currency=row["currency"],
            event_name=row["event_name"],
            impact=ImpactLevel(row["impact"]),
            previous=row["previous"],
            forecast=row["forecast"],
            actual=row["actual"],
            source=KnowledgeSource(row["source"]),
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.utcnow(),
        )

    # ==================== Code Files ====================

    def save_code_file(self, code_file: CodeFile) -> int:
        """Save code file."""
        sql = """
        INSERT INTO code_files (
            knowledge_item_id, filename, language, content, extracted_params
        ) VALUES (?, ?, ?, ?, ?)
        """

        with self._get_conn() as conn:
            cursor = conn.execute(sql, (
                code_file.knowledge_item_id,
                code_file.filename,
                code_file.language,
                code_file.content,
                json.dumps(code_file.extracted_params) if code_file.extracted_params else None,
            ))
            return cursor.lastrowid

    def get_code_files(self, knowledge_item_id: int) -> List[CodeFile]:
        """Get code files for a knowledge item."""
        sql = "SELECT * FROM code_files WHERE knowledge_item_id = ?"

        with self._get_conn() as conn:
            rows = conn.execute(sql, (knowledge_item_id,)).fetchall()
            return [
                CodeFile(
                    id=row["id"],
                    knowledge_item_id=row["knowledge_item_id"],
                    filename=row["filename"],
                    language=row["language"],
                    content=row["content"],
                    extracted_params=json.loads(row["extracted_params"]) if row["extracted_params"] else None,
                    created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.utcnow(),
                )
                for row in rows
            ]

    # ==================== Scrape Logs ====================

    def start_scrape_log(self, source: KnowledgeSource, job_type: str) -> int:
        """Start a scrape log entry."""
        sql = """
        INSERT INTO scrape_logs (source, job_type, started_at, status)
        VALUES (?, ?, ?, 'running')
        """

        with self._get_conn() as conn:
            cursor = conn.execute(sql, (
                source.value,
                job_type,
                datetime.utcnow().isoformat(),
            ))
            return cursor.lastrowid

    def complete_scrape_log(
        self,
        log_id: int,
        status: str,
        items_found: int,
        items_new: int,
        error_message: Optional[str] = None,
    ):
        """Complete a scrape log entry."""
        sql = """
        UPDATE scrape_logs SET
            completed_at = ?,
            status = ?,
            items_found = ?,
            items_new = ?,
            error_message = ?
        WHERE id = ?
        """

        with self._get_conn() as conn:
            conn.execute(sql, (
                datetime.utcnow().isoformat(),
                status,
                items_found,
                items_new,
                error_message,
                log_id,
            ))

    def get_last_scrape(self, source: KnowledgeSource) -> Optional[ScrapeLog]:
        """Get the last scrape log for a source."""
        sql = """
        SELECT * FROM scrape_logs
        WHERE source = ? AND status = 'success'
        ORDER BY completed_at DESC
        LIMIT 1
        """

        with self._get_conn() as conn:
            row = conn.execute(sql, (source.value,)).fetchone()
            if row:
                return ScrapeLog(
                    id=row["id"],
                    source=KnowledgeSource(row["source"]),
                    job_type=row["job_type"],
                    started_at=datetime.fromisoformat(row["started_at"]),
                    completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
                    items_found=row["items_found"],
                    items_new=row["items_new"],
                    error_message=row["error_message"],
                    status=row["status"],
                )
        return None

    # ==================== Stats ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        with self._get_conn() as conn:
            # Total items
            total = conn.execute("SELECT COUNT(*) FROM knowledge_items").fetchone()[0]

            # By source
            by_source = {}
            rows = conn.execute(
                "SELECT source, COUNT(*) as cnt FROM knowledge_items GROUP BY source"
            ).fetchall()
            for row in rows:
                by_source[row["source"]] = row["cnt"]

            # By type
            by_type = {}
            rows = conn.execute(
                "SELECT content_type, COUNT(*) as cnt FROM knowledge_items GROUP BY content_type"
            ).fetchall()
            for row in rows:
                by_type[row["content_type"]] = row["cnt"]

            # Items with embeddings
            with_embeddings = conn.execute(
                "SELECT COUNT(*) FROM knowledge_items WHERE embedding_id IS NOT NULL"
            ).fetchone()[0]

            # Economic events
            events_count = conn.execute("SELECT COUNT(*) FROM economic_events").fetchone()[0]

            # Code files
            code_files_count = conn.execute("SELECT COUNT(*) FROM code_files").fetchone()[0]

            return {
                "total_items": total,
                "by_source": by_source,
                "by_type": by_type,
                "with_embeddings": with_embeddings,
                "economic_events": events_count,
                "code_files": code_files_count,
            }


# Singleton instance
_storage: Optional[KnowledgeStorage] = None


def get_storage() -> KnowledgeStorage:
    """Get or create the storage singleton."""
    global _storage
    if _storage is None:
        _storage = KnowledgeStorage()
    return _storage
