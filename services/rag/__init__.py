"""
Aeon RAG Pipeline
====================

Complete Retrieval-Augmented Generation (RAG) pipeline with:
- Document processing and semantic chunking
- Qdrant vector storage for semantic search
- Two-tier caching (Redis + Qdrant) for performance
- PostgreSQL analytics for query tracking and optimization

Usage:
    from rag import RAGPipeline, process_and_chunk_document
    from rag.analytics import RAGAnalytics

    # Process document
    chunks = process_and_chunk_document(Path("document.pdf"))

    # Initialize pipeline
    pipeline = RAGPipeline(redis_client, vector_store)

    # Add documents
    await pipeline.add_documents(chunks)

    # Query
    results = await pipeline.query("What is this document about?")
"""

from .chunking import (
    DocumentChunk,
    DocumentProcessor,
    SemanticChunker,
    process_and_chunk_document
)

from .vector_store import QdrantVectorStore

from .retrieval import (
    TwoTierRetriever,
    RAGPipeline
)

from .analytics import RAGAnalytics

__version__ = "0.1.0"

__all__ = [
    # Chunking
    "DocumentChunk",
    "DocumentProcessor",
    "SemanticChunker",
    "process_and_chunk_document",

    # Vector storage
    "QdrantVectorStore",

    # Retrieval
    "TwoTierRetriever",
    "RAGPipeline",

    # Analytics
    "RAGAnalytics",
]
