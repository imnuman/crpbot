"""
HYDRA 3.0 - API Caching System

Reduces API costs by caching responses:
- LLM responses (most expensive)
- Market data feeds
- Search results
- Any repeatable API call

Features:
- In-memory LRU cache with TTL
- Disk persistence for expensive calls
- Cost tracking and savings estimation
- Configurable TTL per API type
"""

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Any, Callable

logger = logging.getLogger(__name__)


class CacheType(Enum):
    """Types of cached data with default TTLs."""
    LLM_RESPONSE = "llm"           # 5 min - LLM responses (expensive!)
    MARKET_DATA = "market"         # 30 sec - Prices, order books
    FUNDING_RATE = "funding"       # 1 min - Funding rates
    SEARCH = "search"              # 15 min - Search results
    NEWS = "news"                  # 30 min - News articles
    ANALYSIS = "analysis"          # 5 min - Computed analysis


# Default TTLs in seconds
DEFAULT_TTLS = {
    CacheType.LLM_RESPONSE: 300,    # 5 min
    CacheType.MARKET_DATA: 30,       # 30 sec
    CacheType.FUNDING_RATE: 60,      # 1 min
    CacheType.SEARCH: 900,           # 15 min
    CacheType.NEWS: 1800,            # 30 min
    CacheType.ANALYSIS: 300,         # 5 min
}

# Estimated costs per API call (USD)
ESTIMATED_COSTS = {
    CacheType.LLM_RESPONSE: 0.002,   # ~$0.002 per LLM call
    CacheType.MARKET_DATA: 0.0,      # Free
    CacheType.FUNDING_RATE: 0.0,     # Free
    CacheType.SEARCH: 0.001,         # ~$0.001 per search
    CacheType.NEWS: 0.001,           # ~$0.001 per news fetch
    CacheType.ANALYSIS: 0.0,         # Free (computed locally)
}

# Memory cache limits
MAX_MEMORY_ENTRIES = 1000
MAX_MEMORY_SIZE_MB = 50


@dataclass
class CacheEntry:
    """A single cache entry."""
    key: str
    value: Any
    cache_type: CacheType
    created_at: datetime
    expires_at: datetime
    hit_count: int = 0
    size_bytes: int = 0

    @property
    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at

    @property
    def ttl_remaining(self) -> float:
        return max(0, (self.expires_at - datetime.now()).total_seconds())


@dataclass
class CacheStats:
    """Cache statistics."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    entries_count: int = 0
    memory_size_mb: float = 0.0
    estimated_savings_usd: float = 0.0

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "hit_rate": f"{self.hit_rate:.1%}",
        }


class LRUCache:
    """Thread-safe LRU cache with TTL support."""

    def __init__(self, max_entries: int = MAX_MEMORY_ENTRIES):
        self.max_entries = max_entries
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry if exists and not expired."""
        with self.lock:
            if key not in self.cache:
                return None

            entry = self.cache[key]
            if entry.is_expired:
                del self.cache[key]
                return None

            # Move to end (most recently used)
            self.cache.move_to_end(key)
            entry.hit_count += 1
            return entry

    def set(self, key: str, entry: CacheEntry):
        """Set entry, evicting oldest if needed."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
            elif len(self.cache) >= self.max_entries:
                # Evict oldest (first) entry
                self.cache.popitem(last=False)

            self.cache[key] = entry

    def delete(self, key: str) -> bool:
        """Delete entry."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    def clear(self):
        """Clear all entries."""
        with self.lock:
            self.cache.clear()

    def cleanup_expired(self) -> int:
        """Remove expired entries."""
        with self.lock:
            expired = [k for k, v in self.cache.items() if v.is_expired]
            for key in expired:
                del self.cache[key]
            return len(expired)

    def size(self) -> int:
        return len(self.cache)

    def memory_size_bytes(self) -> int:
        return sum(e.size_bytes for e in self.cache.values())


class APICache:
    """
    Central API caching system for HYDRA.

    Features:
    - In-memory LRU cache for fast access
    - SQLite persistence for expensive calls
    - Automatic TTL management
    - Cost savings tracking
    """

    _instance: Optional["APICache"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, data_dir: Optional[Path] = None):
        if self._initialized:
            return

        # Auto-detect data directory
        if data_dir is None:
            if os.path.exists("/root/crpbot"):
                data_dir = Path("/root/crpbot/data/hydra")
            else:
                data_dir = Path.home() / "crpbot" / "data" / "hydra"

        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self.memory_cache = LRUCache()

        # Disk cache for persistence
        self.db_path = self.data_dir / "api_cache.db"
        self._init_db()

        # Statistics
        self.stats = CacheStats()
        self._load_stats()

        # Custom TTLs (can be overridden)
        self.ttls = dict(DEFAULT_TTLS)

        # Last cleanup time
        self._last_cleanup = datetime.now()

        self._initialized = True
        logger.info(f"APICache initialized (memory: {self.memory_cache.size()}, disk: {self._disk_count()})")

    def _init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                cache_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                hit_count INTEGER DEFAULT 0
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stats (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                total_requests INTEGER DEFAULT 0,
                cache_hits INTEGER DEFAULT 0,
                cache_misses INTEGER DEFAULT 0,
                estimated_savings REAL DEFAULT 0.0,
                updated_at TEXT
            )
        """)

        # Initialize stats row if not exists
        cursor.execute("""
            INSERT OR IGNORE INTO stats (id, total_requests, cache_hits, cache_misses, estimated_savings, updated_at)
            VALUES (1, 0, 0, 0, 0.0, ?)
        """, (datetime.now().isoformat(),))

        conn.commit()
        conn.close()

    def _load_stats(self):
        """Load stats from database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT total_requests, cache_hits, cache_misses, estimated_savings FROM stats WHERE id = 1")
        row = cursor.fetchone()
        if row:
            self.stats.total_requests = row[0]
            self.stats.cache_hits = row[1]
            self.stats.cache_misses = row[2]
            self.stats.estimated_savings_usd = row[3]
        conn.close()

    def _save_stats(self):
        """Save stats to database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE stats SET
                total_requests = ?,
                cache_hits = ?,
                cache_misses = ?,
                estimated_savings = ?,
                updated_at = ?
            WHERE id = 1
        """, (
            self.stats.total_requests,
            self.stats.cache_hits,
            self.stats.cache_misses,
            self.stats.estimated_savings_usd,
            datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()

    def _disk_count(self) -> int:
        """Count entries in disk cache."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM cache")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def _generate_key(self, cache_type: CacheType, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = {
            "type": cache_type.value,
            "args": args,
            "kwargs": sorted(kwargs.items()),
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    def get(
        self,
        cache_type: CacheType,
        *args,
        **kwargs
    ) -> Optional[Any]:
        """
        Get cached value.

        Args:
            cache_type: Type of cached data
            *args, **kwargs: Arguments used to generate cache key

        Returns:
            Cached value or None
        """
        key = self._generate_key(cache_type, *args, **kwargs)
        self.stats.total_requests += 1

        # Check memory cache first
        entry = self.memory_cache.get(key)
        if entry:
            self.stats.cache_hits += 1
            self.stats.estimated_savings_usd += ESTIMATED_COSTS.get(cache_type, 0)
            return entry.value

        # Check disk cache
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            SELECT value, cache_type, created_at, expires_at, hit_count
            FROM cache WHERE key = ?
        """, (key,))
        row = cursor.fetchone()

        if row:
            expires_at = datetime.fromisoformat(row[3])
            if expires_at > datetime.now():
                # Valid cache hit
                value = json.loads(row[0])

                # Update hit count
                cursor.execute("UPDATE cache SET hit_count = hit_count + 1 WHERE key = ?", (key,))
                conn.commit()
                conn.close()

                # Promote to memory cache
                entry = CacheEntry(
                    key=key,
                    value=value,
                    cache_type=cache_type,
                    created_at=datetime.fromisoformat(row[2]),
                    expires_at=expires_at,
                    hit_count=row[4] + 1,
                    size_bytes=len(row[0]),
                )
                self.memory_cache.set(key, entry)

                self.stats.cache_hits += 1
                self.stats.estimated_savings_usd += ESTIMATED_COSTS.get(cache_type, 0)
                return value
            else:
                # Expired, delete
                cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()

        conn.close()
        self.stats.cache_misses += 1
        return None

    def set(
        self,
        cache_type: CacheType,
        value: Any,
        *args,
        ttl: Optional[int] = None,
        persist: bool = True,
        **kwargs
    ) -> str:
        """
        Set cached value.

        Args:
            cache_type: Type of cached data
            value: Value to cache
            *args, **kwargs: Arguments used to generate cache key
            ttl: Time-to-live in seconds (default: type-specific)
            persist: Whether to persist to disk

        Returns:
            Cache key
        """
        key = self._generate_key(cache_type, *args, **kwargs)
        ttl = ttl or self.ttls.get(cache_type, 300)

        now = datetime.now()
        expires_at = now + timedelta(seconds=ttl)

        value_json = json.dumps(value, default=str)
        size_bytes = len(value_json)

        entry = CacheEntry(
            key=key,
            value=value,
            cache_type=cache_type,
            created_at=now,
            expires_at=expires_at,
            hit_count=0,
            size_bytes=size_bytes,
        )

        # Store in memory
        self.memory_cache.set(key, entry)

        # Persist to disk if requested
        if persist:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO cache (key, value, cache_type, created_at, expires_at, hit_count)
                VALUES (?, ?, ?, ?, ?, 0)
            """, (key, value_json, cache_type.value, now.isoformat(), expires_at.isoformat()))
            conn.commit()
            conn.close()

        # Periodic cleanup and stats save
        self._maybe_cleanup()

        return key

    def delete(self, cache_type: CacheType, *args, **kwargs) -> bool:
        """Delete a cached entry."""
        key = self._generate_key(cache_type, *args, **kwargs)

        # Delete from memory
        self.memory_cache.delete(key)

        # Delete from disk
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()

        return deleted

    def invalidate_type(self, cache_type: CacheType) -> int:
        """Invalidate all entries of a specific type."""
        # Clear from memory
        with self.memory_cache.lock:
            keys_to_delete = [
                k for k, v in self.memory_cache.cache.items()
                if v.cache_type == cache_type
            ]
            for key in keys_to_delete:
                del self.memory_cache.cache[key]

        # Clear from disk
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cache WHERE cache_type = ?", (cache_type.value,))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        return deleted + len(keys_to_delete)

    def clear_all(self):
        """Clear all caches."""
        self.memory_cache.clear()

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cache")
        conn.commit()
        conn.close()

        logger.info("All caches cleared")

    def _maybe_cleanup(self):
        """Run cleanup if enough time has passed."""
        now = datetime.now()
        if (now - self._last_cleanup).total_seconds() > 300:  # Every 5 min
            self.cleanup()
            self._save_stats()
            self._last_cleanup = now

    def cleanup(self) -> dict:
        """Remove expired entries from all caches."""
        # Memory cleanup
        memory_cleaned = self.memory_cache.cleanup_expired()

        # Disk cleanup
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cache WHERE expires_at < ?", (datetime.now().isoformat(),))
        disk_cleaned = cursor.rowcount
        conn.commit()
        conn.close()

        if memory_cleaned > 0 or disk_cleaned > 0:
            logger.info(f"Cache cleanup: {memory_cleaned} memory, {disk_cleaned} disk entries removed")

        return {"memory": memory_cleaned, "disk": disk_cleaned}

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        self.stats.entries_count = self.memory_cache.size() + self._disk_count()
        self.stats.memory_size_mb = self.memory_cache.memory_size_bytes() / 1024 / 1024
        return self.stats

    def format_stats(self) -> str:
        """Get human-readable stats."""
        stats = self.get_stats()

        lines = [
            "=== API Cache Stats ===",
            f"Hit Rate: {stats.hit_rate:.1%}",
            f"Requests: {stats.total_requests:,} (Hits: {stats.cache_hits:,}, Misses: {stats.cache_misses:,})",
            f"Entries: {stats.entries_count:,}",
            f"Memory: {stats.memory_size_mb:.2f} MB",
            f"Estimated Savings: ${stats.estimated_savings_usd:.4f}",
        ]

        return "\n".join(lines)


# Singleton accessor
_cache_instance: Optional[APICache] = None


def get_api_cache() -> APICache:
    """Get or create the API cache singleton."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = APICache()
    return _cache_instance


# Decorator for caching function results
def cached(
    cache_type: CacheType,
    ttl: Optional[int] = None,
    key_args: Optional[list[int]] = None,
    key_kwargs: Optional[list[str]] = None
):
    """
    Decorator to cache function results.

    Args:
        cache_type: Type of cache
        ttl: Time-to-live in seconds
        key_args: Indices of positional args to use in cache key
        key_kwargs: Names of keyword args to use in cache key

    Example:
        @cached(CacheType.LLM_RESPONSE, ttl=300)
        def call_llm(prompt: str, model: str) -> str:
            ...
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            cache = get_api_cache()

            # Build cache key from selected args
            cache_args = args if key_args is None else tuple(args[i] for i in key_args if i < len(args))
            cache_kwargs = kwargs if key_kwargs is None else {k: kwargs[k] for k in key_kwargs if k in kwargs}

            # Check cache
            cached_value = cache.get(cache_type, *cache_args, func_name=func.__name__, **cache_kwargs)
            if cached_value is not None:
                return cached_value

            # Call function
            result = func(*args, **kwargs)

            # Store result
            cache.set(cache_type, result, *cache_args, func_name=func.__name__, ttl=ttl, **cache_kwargs)

            return result

        return wrapper
    return decorator


# Async version of the decorator
def cached_async(
    cache_type: CacheType,
    ttl: Optional[int] = None,
    key_args: Optional[list[int]] = None,
    key_kwargs: Optional[list[str]] = None
):
    """Async version of the cached decorator."""
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            cache = get_api_cache()

            cache_args = args if key_args is None else tuple(args[i] for i in key_args if i < len(args))
            cache_kwargs = kwargs if key_kwargs is None else {k: kwargs[k] for k in key_kwargs if k in kwargs}

            cached_value = cache.get(cache_type, *cache_args, func_name=func.__name__, **cache_kwargs)
            if cached_value is not None:
                return cached_value

            result = await func(*args, **kwargs)

            cache.set(cache_type, result, *cache_args, func_name=func.__name__, ttl=ttl, **cache_kwargs)

            return result

        return wrapper
    return decorator
