"""
Aeon RAG Pipeline
Two-tier caching and retrieval system

Tier 1: Redis - Exact query match cache (fast)
Tier 2: Qdrant - Semantic search (slower but more flexible)
"""
from typing import List, Dict, Any, Optional, Tuple
import logging
import hashlib
import json
from datetime import datetime
import redis.asyncio as redis

from .vector_store import QdrantVectorStore
from .chunking import DocumentChunk

logger = logging.getLogger(__name__)


class TwoTierRetriever:
    """
    Two-tier retrieval system for optimal performance

    Tier 1 (Redis): Exact query match - O(1) lookup for repeated queries
    Tier 2 (Qdrant): Semantic search - Finds similar content even if query is different

    This design provides:
    - Fast responses for common queries (cached)
    - Semantic understanding for novel queries
    - Reduced load on embedding service and vector DB
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        vector_store: QdrantVectorStore,
        cache_ttl: int = 3600,  # 1 hour
        similarity_threshold: float = 0.95  # Consider queries similar above this
    ):
        """
        Initialize two-tier retriever

        Args:
            redis_client: Async Redis client
            vector_store: Qdrant vector store
            cache_ttl: Time to live for cached results (seconds)
            similarity_threshold: Threshold for considering queries similar
        """
        self.redis = redis_client
        self.vector_store = vector_store
        self.cache_ttl = cache_ttl
        self.similarity_threshold = similarity_threshold

        logger.info("TwoTierRetriever initialized")

    def _generate_cache_key(self, query: str, **params) -> str:
        """
        Generate cache key for a query

        Args:
            query: Search query
            **params: Additional parameters (limit, filters, etc.)

        Returns:
            Cache key string
        """
        # Create deterministic key from query and parameters
        key_data = {
            "query": query.lower().strip(),
            **params
        }
        key_str = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()[:16]
        return f"rag:cache:{key_hash}"

    async def _get_from_cache(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve results from Redis cache

        Args:
            cache_key: Cache key

        Returns:
            Cached results or None if not found
        """
        try:
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                results = json.loads(cached_data)
                logger.info(f"Cache HIT for key: {cache_key}")
                return results
            else:
                logger.info(f"Cache MISS for key: {cache_key}")
                return None
        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            return None

    async def _save_to_cache(
        self,
        cache_key: str,
        results: List[Tuple[DocumentChunk, float]]
    ):
        """
        Save results to Redis cache

        Args:
            cache_key: Cache key
            results: Search results to cache
        """
        try:
            # Convert results to serializable format
            cache_data = []
            for chunk, score in results:
                cache_data.append({
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "chunk_index": chunk.chunk_index,
                    "char_count": chunk.char_count,
                    "created_at": chunk.created_at,
                    "score": score
                })

            await self.redis.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(cache_data)
            )
            logger.info(f"Saved {len(results)} results to cache: {cache_key}")
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")

    async def retrieve(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.7,
        filter_conditions: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> Tuple[List[Tuple[DocumentChunk, float]], str]:
        """
        Retrieve relevant chunks using two-tier caching

        Args:
            query: Search query
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filter_conditions: Optional filter conditions
            use_cache: Whether to use cache (default True)

        Returns:
            Tuple of (results, source) where source is "cache" or "vector_db"
        """
        # Generate cache key
        cache_key = self._generate_cache_key(
            query,
            limit=limit,
            score_threshold=score_threshold,
            filters=filter_conditions
        )

        # Try Tier 1: Redis cache
        if use_cache:
            cached_results = await self._get_from_cache(cache_key)
            if cached_results:
                # Convert cached data back to DocumentChunk objects
                results = []
                for item in cached_results:
                    chunk = DocumentChunk(
                        chunk_id=item["chunk_id"],
                        document_id=item["document_id"],
                        content=item["content"],
                        metadata=item["metadata"],
                        chunk_index=item["chunk_index"],
                        char_count=item["char_count"],
                        created_at=item["created_at"]
                    )
                    results.append((chunk, item["score"]))
                return results, "cache"

        # Tier 2: Qdrant semantic search
        results = await self.vector_store.search(
            query=query,
            limit=limit,
            score_threshold=score_threshold,
            filter_conditions=filter_conditions
        )

        # Cache the results for future queries
        if use_cache and results:
            await self._save_to_cache(cache_key, results)

        return results, "vector_db"

    async def invalidate_cache(self, pattern: str = "rag:cache:*"):
        """
        Invalidate cache entries matching pattern

        Args:
            pattern: Redis key pattern to match
        """
        try:
            cursor = 0
            deleted_count = 0

            while True:
                cursor, keys = await self.redis.scan(cursor, match=pattern, count=100)
                if keys:
                    deleted_count += await self.redis.delete(*keys)

                if cursor == 0:
                    break

            logger.info(f"Invalidated {deleted_count} cache entries matching: {pattern}")
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")

    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dictionary with cache stats
        """
        try:
            cursor = 0
            total_keys = 0
            total_size = 0

            while True:
                cursor, keys = await self.redis.scan(
                    cursor, match="rag:cache:*", count=100
                )
                total_keys += len(keys)

                for key in keys:
                    value = await self.redis.get(key)
                    if value:
                        total_size += len(value)

                if cursor == 0:
                    break

            return {
                "total_cached_queries": total_keys,
                "total_cache_size_bytes": total_size,
                "cache_ttl_seconds": self.cache_ttl
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}


class RAGPipeline:
    """
    Complete RAG pipeline integrating document processing, storage, and retrieval
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        vector_store: QdrantVectorStore,
        cache_ttl: int = 3600
    ):
        """
        Initialize RAG pipeline

        Args:
            redis_client: Async Redis client
            vector_store: Qdrant vector store
            cache_ttl: Cache TTL in seconds
        """
        self.vector_store = vector_store
        self.retriever = TwoTierRetriever(
            redis_client=redis_client,
            vector_store=vector_store,
            cache_ttl=cache_ttl
        )

        logger.info("RAGPipeline initialized")

    async def add_documents(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 32
    ) -> int:
        """
        Add document chunks to the RAG system

        Args:
            chunks: List of DocumentChunk objects
            batch_size: Batch size for processing

        Returns:
            Number of chunks added
        """
        count = await self.vector_store.add_chunks(chunks, batch_size)

        # Invalidate cache since new documents were added
        await self.retriever.invalidate_cache()

        return count

    async def query(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.7,
        filter_conditions: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system

        Args:
            query: Search query
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filter_conditions: Optional filters
            use_cache: Whether to use cache

        Returns:
            Dictionary with results and metadata
        """
        results, source = await self.retriever.retrieve(
            query=query,
            limit=limit,
            score_threshold=score_threshold,
            filter_conditions=filter_conditions,
            use_cache=use_cache
        )

        # Format response
        formatted_results = []
        for chunk, score in results:
            formatted_results.append({
                "content": chunk.content,
                "metadata": chunk.metadata,
                "score": score,
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id
            })

        return {
            "query": query,
            "results": formatted_results,
            "count": len(formatted_results),
            "source": source,  # "cache" or "vector_db"
            "timestamp": datetime.utcnow().isoformat()
        }

    async def delete_document(self, document_id: str):
        """
        Delete a document from the RAG system

        Args:
            document_id: Document ID to delete
        """
        self.vector_store.delete_document(document_id)

        # Invalidate cache
        await self.retriever.invalidate_cache()

        logger.info(f"Deleted document {document_id} from RAG system")

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get RAG pipeline statistics

        Returns:
            Dictionary with pipeline stats
        """
        cache_stats = await self.retriever.get_cache_stats()
        collection_info = self.vector_store.get_collection_info()

        return {
            "vector_store": collection_info,
            "cache": cache_stats,
            "timestamp": datetime.utcnow().isoformat()
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of RAG pipeline components

        Returns:
            Health status dictionary
        """
        try:
            # Check Redis
            redis_healthy = await self.retriever.redis.ping()

            # Check Qdrant and embedding service
            vector_store_healthy = await self.vector_store.health_check()

            overall_healthy = redis_healthy and vector_store_healthy

            return {
                "status": "healthy" if overall_healthy else "degraded",
                "redis": "healthy" if redis_healthy else "unhealthy",
                "vector_store": "healthy" if vector_store_healthy else "unhealthy",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
