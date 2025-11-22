"""
Aeon RAG Pipeline
Qdrant vector storage integration for semantic search
"""
from typing import List, Dict, Any, Optional, Tuple
import logging
import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchParams
)
from dataclasses import asdict
import os

from .chunking import DocumentChunk

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """
    Qdrant vector store for semantic search and retrieval

    Manages document embeddings and provides semantic search capabilities
    """

    def __init__(
        self,
        collection_name: str = "aeon_documents",
        qdrant_host: Optional[str] = None,
        embedding_endpoint: Optional[str] = None,
        embedding_dim: int = 384,  # all-MiniLM-L6-v2 dimension
        distance_metric: Distance = Distance.COSINE
    ):
        """
        Initialize Qdrant vector store

        Args:
            collection_name: Name of the Qdrant collection
            qdrant_host: Qdrant host URL (default from env)
            embedding_endpoint: Embedding service URL (default from env)
            embedding_dim: Dimension of embedding vectors
            distance_metric: Distance metric for similarity search
        """
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.distance_metric = distance_metric

        # Get endpoints from environment
        self.qdrant_host = qdrant_host or os.getenv(
            "QDRANT_HOST", "http://qdrant.vector-db:6333"
        )
        self.embedding_endpoint = embedding_endpoint or os.getenv(
            "EMBEDDING_ENDPOINT", "http://localhost:8001"
        )

        # Initialize Qdrant client
        self.client = QdrantClient(url=self.qdrant_host)

        # Initialize collection if it doesn't exist
        self._init_collection()

        logger.info(f"QdrantVectorStore initialized with collection: {collection_name}")

    def _init_collection(self):
        """Initialize Qdrant collection if it doesn't exist"""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            exists = any(col.name == self.collection_name for col in collections)

            if not exists:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=self.distance_metric
                    )
                )
                logger.info(f"Collection {self.collection_name} created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")

        except Exception as e:
            logger.error(f"Error initializing collection: {e}")
            raise

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using the embedding service

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.embedding_endpoint}/embed",
                    json={"texts": [text], "normalize": True}
                )
                response.raise_for_status()
                result = response.json()
                return result["embeddings"][0]
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    async def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch

        Returns:
            List of embedding vectors
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        f"{self.embedding_endpoint}/embed",
                        json={"texts": batch, "normalize": True}
                    )
                    response.raise_for_status()
                    result = response.json()
                    all_embeddings.extend(result["embeddings"])

                logger.info(f"Generated embeddings for batch {i // batch_size + 1}")
            except Exception as e:
                logger.error(f"Error generating embeddings for batch: {e}")
                raise

        return all_embeddings

    async def add_chunks(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 32
    ) -> int:
        """
        Add document chunks to vector store

        Args:
            chunks: List of DocumentChunk objects
            batch_size: Batch size for embedding generation

        Returns:
            Number of chunks added
        """
        if not chunks:
            logger.warning("No chunks to add")
            return 0

        try:
            # Generate embeddings for all chunks
            texts = [chunk.content for chunk in chunks]
            embeddings = await self.generate_embeddings_batch(texts, batch_size)

            # Create points for Qdrant
            points = []
            for chunk, embedding in zip(chunks, embeddings):
                point = PointStruct(
                    id=chunk.chunk_id,
                    vector=embedding,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "content": chunk.content,
                        "metadata": chunk.metadata,
                        "chunk_index": chunk.chunk_index,
                        "char_count": chunk.char_count,
                        "created_at": chunk.created_at
                    }
                )
                points.append(point)

            # Upload to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            logger.info(f"Added {len(chunks)} chunks to vector store")
            return len(chunks)

        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}")
            raise

    async def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.7,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Semantic search for relevant chunks

        Args:
            query: Search query
            limit: Maximum number of results
            score_threshold: Minimum similarity score (0-1)
            filter_conditions: Optional filter conditions

        Returns:
            List of (DocumentChunk, score) tuples
        """
        try:
            # Generate query embedding
            query_embedding = await self.generate_embedding(query)

            # Build filter if provided
            query_filter = None
            if filter_conditions:
                conditions = []
                for key, value in filter_conditions.items():
                    conditions.append(
                        FieldCondition(
                            key=f"metadata.{key}",
                            match=MatchValue(value=value)
                        )
                    )
                query_filter = Filter(must=conditions)

            # Search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
                search_params=SearchParams(
                    exact=False,  # Use approximate search for speed
                    hnsw_ef=128   # Higher = more accurate but slower
                )
            )

            # Convert results to DocumentChunk objects
            results = []
            for hit in search_result:
                chunk = DocumentChunk(
                    chunk_id=hit.payload["chunk_id"],
                    document_id=hit.payload["document_id"],
                    content=hit.payload["content"],
                    metadata=hit.payload["metadata"],
                    chunk_index=hit.payload["chunk_index"],
                    char_count=hit.payload["char_count"],
                    created_at=hit.payload["created_at"]
                )
                results.append((chunk, hit.score))

            logger.info(f"Found {len(results)} results for query: {query[:50]}...")
            return results

        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise

    async def search_by_document_id(
        self,
        document_id: str,
        limit: int = 100
    ) -> List[DocumentChunk]:
        """
        Retrieve all chunks for a specific document

        Args:
            document_id: Document ID to search for
            limit: Maximum number of chunks to return

        Returns:
            List of DocumentChunk objects
        """
        try:
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                ),
                limit=limit
            )

            chunks = []
            for point in scroll_result[0]:
                chunk = DocumentChunk(
                    chunk_id=point.payload["chunk_id"],
                    document_id=point.payload["document_id"],
                    content=point.payload["content"],
                    metadata=point.payload["metadata"],
                    chunk_index=point.payload["chunk_index"],
                    char_count=point.payload["char_count"],
                    created_at=point.payload["created_at"]
                )
                chunks.append(chunk)

            # Sort by chunk_index
            chunks.sort(key=lambda x: x.chunk_index)

            logger.info(f"Retrieved {len(chunks)} chunks for document {document_id}")
            return chunks

        except Exception as e:
            logger.error(f"Error retrieving document chunks: {e}")
            raise

    def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks for a document

        Args:
            document_id: Document ID to delete

        Returns:
            Number of chunks deleted
        """
        try:
            result = self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
            )

            logger.info(f"Deleted document {document_id} from vector store")
            return result.status

        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection

        Returns:
            Collection information including size, vector count, etc.
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status,
                "optimizer_status": info.optimizer_status,
                "indexed_vectors_count": info.indexed_vectors_count
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            raise

    async def health_check(self) -> bool:
        """
        Check if Qdrant and embedding service are healthy

        Returns:
            True if both services are healthy
        """
        try:
            # Check Qdrant
            collections = self.client.get_collections()
            qdrant_healthy = len(collections.collections) >= 0

            # Check embedding service
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.embedding_endpoint}/health")
                embedding_healthy = response.status_code == 200

            return qdrant_healthy and embedding_healthy

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
