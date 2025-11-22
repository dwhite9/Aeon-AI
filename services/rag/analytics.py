"""
Aeon RAG Pipeline
PostgreSQL analytics for query tracking and optimization
"""
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timedelta
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Text,
    Boolean,
    JSON,
    Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
import os

logger = logging.getLogger(__name__)

Base = declarative_base()


class QueryLog(Base):
    """Track all RAG queries for analytics"""
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_text = Column(Text, nullable=False)
    query_hash = Column(String(64), nullable=False, index=True)
    session_id = Column(String(64), index=True)

    # Results
    result_count = Column(Integer, default=0)
    top_score = Column(Float)
    cache_hit = Column(Boolean, default=False)

    # Performance
    latency_ms = Column(Float)
    embedding_time_ms = Column(Float, nullable=True)
    search_time_ms = Column(Float, nullable=True)

    # Metadata
    filters_applied = Column(JSON, nullable=True)
    score_threshold = Column(Float)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Indexes for common queries
    __table_args__ = (
        Index('idx_created_at_cache', 'created_at', 'cache_hit'),
        Index('idx_session_query', 'session_id', 'created_at'),
    )


class DocumentMetrics(Base):
    """Track document usage and performance"""
    __tablename__ = "document_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String(64), unique=True, nullable=False, index=True)

    # Document info
    filename = Column(String(255))
    file_type = Column(String(50))
    file_size = Column(Integer)
    chunk_count = Column(Integer, default=0)

    # Usage stats
    total_retrievals = Column(Integer, default=0)
    avg_relevance_score = Column(Float)
    last_accessed = Column(DateTime)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class CachePerformance(Base):
    """Track cache performance over time"""
    __tablename__ = "cache_performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Aggregated metrics (per hour)
    hour = Column(DateTime, unique=True, index=True)
    total_queries = Column(Integer, default=0)
    cache_hits = Column(Integer, default=0)
    cache_misses = Column(Integer, default=0)
    avg_latency_cached_ms = Column(Float)
    avg_latency_uncached_ms = Column(Float)


class RAGAnalytics:
    """
    PostgreSQL analytics for RAG pipeline

    Tracks queries, performance, and usage patterns for optimization
    """

    def __init__(
        self,
        db_url: Optional[str] = None
    ):
        """
        Initialize RAG analytics

        Args:
            db_url: Database URL (default from environment)
        """
        # Build database URL from environment variables
        if not db_url:
            host = os.getenv("POSTGRES_HOST", "postgres-postgresql")
            port = os.getenv("POSTGRES_PORT", "5432")
            db = os.getenv("POSTGRES_DB", "aiplatform")
            user = os.getenv("POSTGRES_USER", "aiuser")
            password = os.getenv("POSTGRES_PASSWORD", "changeme_to_secure_password")
            db_url = f"postgresql://{user}:{password}@{host}:{port}/{db}"

        # Create engine
        self.engine = create_engine(db_url, pool_pre_ping=True)

        # Create tables
        Base.metadata.create_all(self.engine)

        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)

        logger.info("RAGAnalytics initialized")

    def log_query(
        self,
        query_text: str,
        query_hash: str,
        session_id: str,
        result_count: int,
        top_score: Optional[float],
        cache_hit: bool,
        latency_ms: float,
        embedding_time_ms: Optional[float] = None,
        search_time_ms: Optional[float] = None,
        filters_applied: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.7
    ):
        """
        Log a query for analytics

        Args:
            query_text: The query text
            query_hash: Hash of the query
            session_id: Session identifier
            result_count: Number of results returned
            top_score: Highest relevance score
            cache_hit: Whether this was a cache hit
            latency_ms: Total latency in milliseconds
            embedding_time_ms: Time to generate embedding
            search_time_ms: Time to perform search
            filters_applied: Any filters applied
            score_threshold: Score threshold used
        """
        try:
            with self.SessionLocal() as session:
                log_entry = QueryLog(
                    query_text=query_text[:1000],  # Limit length
                    query_hash=query_hash,
                    session_id=session_id,
                    result_count=result_count,
                    top_score=top_score,
                    cache_hit=cache_hit,
                    latency_ms=latency_ms,
                    embedding_time_ms=embedding_time_ms,
                    search_time_ms=search_time_ms,
                    filters_applied=filters_applied,
                    score_threshold=score_threshold
                )
                session.add(log_entry)
                session.commit()

                logger.debug(f"Logged query: {query_hash}")
        except Exception as e:
            logger.error(f"Error logging query: {e}")

    def track_document(
        self,
        document_id: str,
        filename: str,
        file_type: str,
        file_size: int,
        chunk_count: int
    ):
        """
        Track a new document

        Args:
            document_id: Document identifier
            filename: Original filename
            file_type: File extension/type
            file_size: File size in bytes
            chunk_count: Number of chunks created
        """
        try:
            with self.SessionLocal() as session:
                # Check if document exists
                doc = session.query(DocumentMetrics).filter_by(
                    document_id=document_id
                ).first()

                if doc:
                    # Update existing
                    doc.chunk_count = chunk_count
                    doc.updated_at = datetime.utcnow()
                else:
                    # Create new
                    doc = DocumentMetrics(
                        document_id=document_id,
                        filename=filename,
                        file_type=file_type,
                        file_size=file_size,
                        chunk_count=chunk_count
                    )
                    session.add(doc)

                session.commit()
                logger.debug(f"Tracked document: {document_id}")
        except Exception as e:
            logger.error(f"Error tracking document: {e}")

    def update_document_retrieval(
        self,
        document_id: str,
        relevance_score: float
    ):
        """
        Update document retrieval statistics

        Args:
            document_id: Document identifier
            relevance_score: Relevance score from this retrieval
        """
        try:
            with self.SessionLocal() as session:
                doc = session.query(DocumentMetrics).filter_by(
                    document_id=document_id
                ).first()

                if doc:
                    doc.total_retrievals += 1
                    doc.last_accessed = datetime.utcnow()

                    # Update rolling average of relevance
                    if doc.avg_relevance_score is None:
                        doc.avg_relevance_score = relevance_score
                    else:
                        # Exponential moving average
                        alpha = 0.2
                        doc.avg_relevance_score = (
                            alpha * relevance_score +
                            (1 - alpha) * doc.avg_relevance_score
                        )

                    session.commit()
        except Exception as e:
            logger.error(f"Error updating document retrieval: {e}")

    def aggregate_cache_performance(self, hours_back: int = 24):
        """
        Aggregate cache performance metrics

        Args:
            hours_back: How many hours of data to aggregate
        """
        try:
            with self.SessionLocal() as session:
                # Get cutoff time
                cutoff = datetime.utcnow() - timedelta(hours=hours_back)

                # Get queries since cutoff, grouped by hour
                queries = session.query(
                    func.date_trunc('hour', QueryLog.created_at).label('hour'),
                    func.count(QueryLog.id).label('total'),
                    func.sum(func.cast(QueryLog.cache_hit, Integer)).label('hits'),
                    func.avg(func.case(
                        (QueryLog.cache_hit == True, QueryLog.latency_ms),
                        else_=None
                    )).label('avg_latency_cached'),
                    func.avg(func.case(
                        (QueryLog.cache_hit == False, QueryLog.latency_ms),
                        else_=None
                    )).label('avg_latency_uncached')
                ).filter(
                    QueryLog.created_at >= cutoff
                ).group_by(
                    func.date_trunc('hour', QueryLog.created_at)
                ).all()

                # Upsert cache performance records
                for query in queries:
                    perf = session.query(CachePerformance).filter_by(
                        hour=query.hour
                    ).first()

                    if perf:
                        perf.total_queries = query.total
                        perf.cache_hits = query.hits or 0
                        perf.cache_misses = query.total - (query.hits or 0)
                        perf.avg_latency_cached_ms = query.avg_latency_cached
                        perf.avg_latency_uncached_ms = query.avg_latency_uncached
                    else:
                        perf = CachePerformance(
                            hour=query.hour,
                            total_queries=query.total,
                            cache_hits=query.hits or 0,
                            cache_misses=query.total - (query.hits or 0),
                            avg_latency_cached_ms=query.avg_latency_cached,
                            avg_latency_uncached_ms=query.avg_latency_uncached
                        )
                        session.add(perf)

                session.commit()
                logger.info(f"Aggregated cache performance for {len(queries)} hours")
        except Exception as e:
            logger.error(f"Error aggregating cache performance: {e}")

    def get_query_stats(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get query statistics for the last N hours

        Args:
            hours: Number of hours to look back

        Returns:
            Dictionary with query statistics
        """
        try:
            with self.SessionLocal() as session:
                cutoff = datetime.utcnow() - timedelta(hours=hours)

                # Total queries
                total = session.query(func.count(QueryLog.id)).filter(
                    QueryLog.created_at >= cutoff
                ).scalar()

                # Cache hit rate
                cache_hits = session.query(func.count(QueryLog.id)).filter(
                    QueryLog.created_at >= cutoff,
                    QueryLog.cache_hit == True
                ).scalar()

                # Average latency
                avg_latency = session.query(func.avg(QueryLog.latency_ms)).filter(
                    QueryLog.created_at >= cutoff
                ).scalar()

                # Average results per query
                avg_results = session.query(func.avg(QueryLog.result_count)).filter(
                    QueryLog.created_at >= cutoff,
                    QueryLog.result_count > 0
                ).scalar()

                return {
                    "period_hours": hours,
                    "total_queries": total or 0,
                    "cache_hit_rate": (cache_hits / total) if total > 0 else 0,
                    "avg_latency_ms": float(avg_latency) if avg_latency else 0,
                    "avg_results_per_query": float(avg_results) if avg_results else 0
                }
        except Exception as e:
            logger.error(f"Error getting query stats: {e}")
            return {}

    def get_top_documents(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get most frequently retrieved documents

        Args:
            limit: Number of documents to return

        Returns:
            List of document statistics
        """
        try:
            with self.SessionLocal() as session:
                docs = session.query(DocumentMetrics).order_by(
                    DocumentMetrics.total_retrievals.desc()
                ).limit(limit).all()

                return [{
                    "document_id": doc.document_id,
                    "filename": doc.filename,
                    "file_type": doc.file_type,
                    "chunk_count": doc.chunk_count,
                    "total_retrievals": doc.total_retrievals,
                    "avg_relevance_score": float(doc.avg_relevance_score) if doc.avg_relevance_score else 0,
                    "last_accessed": doc.last_accessed.isoformat() if doc.last_accessed else None
                } for doc in docs]
        except Exception as e:
            logger.error(f"Error getting top documents: {e}")
            return []

    def cleanup_old_logs(self, days: int = 30):
        """
        Remove old query logs to manage database size

        Args:
            days: Delete logs older than this many days
        """
        try:
            with self.SessionLocal() as session:
                cutoff = datetime.utcnow() - timedelta(days=days)

                deleted = session.query(QueryLog).filter(
                    QueryLog.created_at < cutoff
                ).delete()

                session.commit()
                logger.info(f"Deleted {deleted} old query logs (older than {days} days)")
        except Exception as e:
            logger.error(f"Error cleaning up old logs: {e}")
