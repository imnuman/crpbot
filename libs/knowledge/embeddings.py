"""
Embedding generation for semantic search using OpenAI or local models.
"""

import os
import hashlib
from typing import List, Optional, Dict, Any
from loguru import logger

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("chromadb not installed - vector search disabled")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai not installed - using fallback embeddings")

from .base import KnowledgeItem, ContentType
from .storage import get_storage, DEFAULT_CHROMA_PATH


# Embedding configuration
EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dimensions
EMBEDDING_DIMENSIONS = 1536
COLLECTION_NAME = "trading_knowledge"


class EmbeddingService:
    """Service for generating and managing embeddings."""

    def __init__(
        self,
        chroma_path: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        self.chroma_path = chroma_path or str(DEFAULT_CHROMA_PATH)
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        self._chroma_client = None
        self._collection = None
        self._openai_client = None

    @property
    def chroma_client(self):
        """Lazy initialization of ChromaDB client."""
        if self._chroma_client is None and CHROMADB_AVAILABLE:
            os.makedirs(self.chroma_path, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(
                path=self.chroma_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
        return self._chroma_client

    @property
    def collection(self):
        """Get or create the knowledge collection."""
        if self._collection is None and self.chroma_client:
            self._collection = self.chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    @property
    def openai_client(self):
        """Lazy initialization of OpenAI client."""
        if self._openai_client is None and OPENAI_AVAILABLE and self.openai_api_key:
            self._openai_client = openai.OpenAI(api_key=self.openai_api_key)
        return self._openai_client

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using OpenAI."""
        if not self.openai_client:
            logger.warning("OpenAI client not available - using hash-based fallback")
            return self._fallback_embedding(text)

        try:
            # Truncate to ~8000 tokens (~32000 chars) to stay within limits
            text = text[:32000]

            response = self.openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text,
            )
            return response.data[0].embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return self._fallback_embedding(text)

    def _fallback_embedding(self, text: str) -> List[float]:
        """Generate simple hash-based embedding as fallback."""
        # Create deterministic embedding from text hash
        # Not semantic, but allows basic similarity matching
        h = hashlib.sha512(text.encode()).digest()

        # Expand to EMBEDDING_DIMENSIONS using repeated hashing
        embedding = []
        seed = text
        while len(embedding) < EMBEDDING_DIMENSIONS:
            h = hashlib.sha512(seed.encode()).digest()
            # Convert bytes to floats in [-1, 1]
            for i in range(0, len(h), 4):
                if len(embedding) >= EMBEDDING_DIMENSIONS:
                    break
                val = int.from_bytes(h[i:i+4], 'big', signed=True)
                embedding.append(val / (2**31))
            seed = h.hex()

        return embedding[:EMBEDDING_DIMENSIONS]

    def embed_and_store(self, item: KnowledgeItem) -> Optional[str]:
        """Generate embedding for item and store in ChromaDB."""
        if not self.collection:
            logger.warning("ChromaDB not available - skipping embedding storage")
            return None

        # Generate embedding ID
        embedding_id = f"{item.source.value}_{item.get_content_hash()}"

        # Create text for embedding (prioritize summary over full content)
        text = self._prepare_text_for_embedding(item)

        # Generate embedding
        embedding = self.generate_embedding(text)
        if not embedding:
            return None

        # Prepare metadata
        metadata = {
            "source": item.source.value,
            "content_type": item.content_type.value,
            "title": item.title[:500] if item.title else "",
            "symbols": ",".join(item.symbols) if item.symbols else "",
            "timeframes": ",".join(item.timeframes) if item.timeframes else "",
            "quality_score": item.quality_score or 0.5,
        }

        # Store in ChromaDB
        try:
            self.collection.upsert(
                ids=[embedding_id],
                embeddings=[embedding],
                documents=[text[:10000]],  # Store truncated text
                metadatas=[metadata],
            )
            logger.debug(f"Stored embedding: {embedding_id}")
            return embedding_id

        except Exception as e:
            logger.error(f"Failed to store embedding: {e}")
            return None

    def _prepare_text_for_embedding(self, item: KnowledgeItem) -> str:
        """Prepare text for embedding generation."""
        parts = []

        # Title is important
        if item.title:
            parts.append(f"Title: {item.title}")

        # Add symbols and timeframes for context
        if item.symbols:
            parts.append(f"Symbols: {', '.join(item.symbols)}")
        if item.timeframes:
            parts.append(f"Timeframes: {', '.join(item.timeframes)}")

        # Summary or full content
        if item.summary:
            parts.append(f"Summary: {item.summary}")
        elif item.full_content:
            # Use first 5000 chars of content
            parts.append(f"Content: {item.full_content[:5000]}")

        # Tags
        if item.tags:
            parts.append(f"Tags: {', '.join(item.tags)}")

        return "\n".join(parts)

    def search_similar(
        self,
        query: str,
        n_results: int = 10,
        source: Optional[str] = None,
        content_type: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        min_quality: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar items using vector similarity."""
        if not self.collection:
            logger.warning("ChromaDB not available - returning empty results")
            return []

        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        if not query_embedding:
            return []

        # Build where filter
        where_filter = {}
        if source:
            where_filter["source"] = source
        if content_type:
            where_filter["content_type"] = content_type
        if min_quality is not None:
            where_filter["quality_score"] = {"$gte": min_quality}

        # Query ChromaDB
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"],
            )

            # Format results
            formatted = []
            if results and results["ids"]:
                for i, id_ in enumerate(results["ids"][0]):
                    formatted.append({
                        "id": id_,
                        "document": results["documents"][0][i] if results["documents"] else None,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else None,
                        "similarity": 1 - results["distances"][0][i] if results["distances"] else None,
                    })

            return formatted

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def search_by_symbol(
        self,
        symbol: str,
        n_results: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search for items related to a specific symbol."""
        # Create query that emphasizes the symbol
        query = f"Trading strategy for {symbol}"
        return self.search_similar(query, n_results=n_results)

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding collection."""
        if not self.collection:
            return {"error": "ChromaDB not available"}

        try:
            count = self.collection.count()
            return {
                "total_embeddings": count,
                "collection_name": COLLECTION_NAME,
                "model": EMBEDDING_MODEL,
                "dimensions": EMBEDDING_DIMENSIONS,
            }
        except Exception as e:
            return {"error": str(e)}

    def process_pending_embeddings(self, batch_size: int = 50) -> int:
        """Process items without embeddings."""
        storage = get_storage()
        items = storage.get_items_without_embeddings(limit=batch_size)

        processed = 0
        for item in items:
            embedding_id = self.embed_and_store(item)
            if embedding_id:
                storage.update_embedding_id(item.id, embedding_id)
                processed += 1

        logger.info(f"Processed {processed}/{len(items)} embeddings")
        return processed


# Singleton instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
