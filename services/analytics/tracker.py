"""
Analytics Tracker - Real-time query and performance tracking

Tracks all system queries and generates actionable insights.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import json

try:
    from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, Session
    SQLALCHEMY_AVAILABLE = True
    Base = declarative_base()
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    Base = None

logger = logging.getLogger(__name__)


@dataclass
class QueryMetric:
    """Metrics for a single query"""
    query_id: str
    query_text: str
    tool_used: str
    execution_time: float
    success: bool
    timestamp: datetime
    session_id: Optional[str] = None
    error_message: Optional[str] = None
    cache_hit: bool = False
    result_quality: Optional[float] = None  # 0-1 score


@dataclass
class ToolUsageMetric:
    """Usage statistics for a specific tool"""
    tool_name: str
    total_calls: int
    successful_calls: int
    failed_calls: int
    avg_execution_time: float
    total_execution_time: float
    cache_hits: int
    period_start: datetime
    period_end: datetime


# SQLAlchemy models
if SQLALCHEMY_AVAILABLE:
    class QueryLog(Base):
        """Persistent storage for query metrics"""
        __tablename__ = "query_logs"

        id = Column(Integer, primary_key=True, autoincrement=True)
        query_id = Column(String(64), unique=True, index=True)
        query_text = Column(Text)
        tool_used = Column(String(50), index=True)
        execution_time = Column(Float)
        success = Column(Boolean, index=True)
        timestamp = Column(DateTime, index=True)
        session_id = Column(String(64), index=True)
        error_message = Column(Text, nullable=True)
        cache_hit = Column(Boolean, default=False)
        result_quality = Column(Float, nullable=True)


class AnalyticsTracker:
    """
    Tracks queries and generates analytics

    Provides real-time and historical analytics for optimization.
    """

    def __init__(
        self,
        postgres_url: Optional[str] = None,
        retention_days: int = 90
    ):
        """
        Initialize analytics tracker

        Args:
            postgres_url: PostgreSQL connection string
            retention_days: How long to keep query logs
        """
        self.retention_days = retention_days
        self.in_memory_metrics: List[QueryMetric] = []
        self.max_memory_metrics = 10000  # Keep last 10k in memory

        # Initialize database if available
        self.db_session = None
        if postgres_url and SQLALCHEMY_AVAILABLE:
            try:
                self.engine = create_engine(postgres_url, pool_pre_ping=True)
                Base.metadata.create_all(self.engine)
                SessionLocal = sessionmaker(bind=self.engine)
                self.db_session = SessionLocal()
                logger.info("Analytics database initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize analytics database: {e}")
                self.db_session = None

    def track_query(self, metric: QueryMetric) -> None:
        """
        Track a query execution

        Args:
            metric: Query metric to track
        """
        # Add to in-memory storage
        self.in_memory_metrics.append(metric)

        # Trim if too large
        if len(self.in_memory_metrics) > self.max_memory_metrics:
            self.in_memory_metrics = self.in_memory_metrics[-self.max_memory_metrics:]

        # Persist to database if available
        if self.db_session:
            try:
                log_entry = QueryLog(
                    query_id=metric.query_id,
                    query_text=metric.query_text,
                    tool_used=metric.tool_used,
                    execution_time=metric.execution_time,
                    success=metric.success,
                    timestamp=metric.timestamp,
                    session_id=metric.session_id,
                    error_message=metric.error_message,
                    cache_hit=metric.cache_hit,
                    result_quality=metric.result_quality
                )
                self.db_session.add(log_entry)
                self.db_session.commit()
            except Exception as e:
                logger.error(f"Failed to persist query metric: {e}")
                self.db_session.rollback()

    def get_tool_usage_stats(
        self,
        hours: int = 24,
        tool_name: Optional[str] = None
    ) -> List[ToolUsageMetric]:
        """
        Get tool usage statistics

        Args:
            hours: Time period to analyze
            tool_name: Specific tool to analyze (None for all)

        Returns:
            List of tool usage metrics
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        # Filter metrics
        recent_metrics = [
            m for m in self.in_memory_metrics
            if m.timestamp >= cutoff and (tool_name is None or m.tool_used == tool_name)
        ]

        # Aggregate by tool
        tool_stats = defaultdict(lambda: {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "execution_times": [],
            "cache_hits": 0
        })

        for metric in recent_metrics:
            stats = tool_stats[metric.tool_used]
            stats["total"] += 1
            if metric.success:
                stats["successful"] += 1
            else:
                stats["failed"] += 1
            stats["execution_times"].append(metric.execution_time)
            if metric.cache_hit:
                stats["cache_hits"] += 1

        # Convert to ToolUsageMetric objects
        period_start = cutoff
        period_end = datetime.utcnow()

        results = []
        for tool, stats in tool_stats.items():
            avg_time = sum(stats["execution_times"]) / len(stats["execution_times"]) if stats["execution_times"] else 0
            total_time = sum(stats["execution_times"])

            results.append(ToolUsageMetric(
                tool_name=tool,
                total_calls=stats["total"],
                successful_calls=stats["successful"],
                failed_calls=stats["failed"],
                avg_execution_time=avg_time,
                total_execution_time=total_time,
                cache_hits=stats["cache_hits"],
                period_start=period_start,
                period_end=period_end
            ))

        return results

    def get_query_trends(self, hours: int = 24) -> Dict[str, Any]:
        """
        Analyze query trends

        Args:
            hours: Time period to analyze

        Returns:
            Dictionary of trend metrics
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent_metrics = [m for m in self.in_memory_metrics if m.timestamp >= cutoff]

        if not recent_metrics:
            return {
                "total_queries": 0,
                "queries_per_hour": 0,
                "avg_execution_time": 0,
                "cache_hit_rate": 0,
                "success_rate": 0,
                "most_used_tool": None
            }

        total = len(recent_metrics)
        successful = sum(1 for m in recent_metrics if m.success)
        cache_hits = sum(1 for m in recent_metrics if m.cache_hit)
        execution_times = [m.execution_time for m in recent_metrics]

        # Tool usage
        tool_counts = defaultdict(int)
        for m in recent_metrics:
            tool_counts[m.tool_used] += 1
        most_used_tool = max(tool_counts.items(), key=lambda x: x[1])[0] if tool_counts else None

        return {
            "total_queries": total,
            "queries_per_hour": total / hours,
            "avg_execution_time": sum(execution_times) / len(execution_times),
            "cache_hit_rate": cache_hits / total if total > 0 else 0,
            "success_rate": successful / total if total > 0 else 0,
            "most_used_tool": most_used_tool,
            "tool_distribution": dict(tool_counts),
            "period_hours": hours
        }

    def get_performance_percentiles(
        self,
        hours: int = 24,
        tool_name: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calculate performance percentiles

        Args:
            hours: Time period to analyze
            tool_name: Specific tool to analyze

        Returns:
            Dictionary of percentile metrics
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.in_memory_metrics
            if m.timestamp >= cutoff and (tool_name is None or m.tool_used == tool_name)
        ]

        if not recent_metrics:
            return {}

        execution_times = sorted([m.execution_time for m in recent_metrics])
        n = len(execution_times)

        def percentile(p):
            k = (n - 1) * p
            f = int(k)
            c = k - f
            if f + 1 < n:
                return execution_times[f] + c * (execution_times[f + 1] - execution_times[f])
            else:
                return execution_times[f]

        return {
            "p50": percentile(0.50),
            "p75": percentile(0.75),
            "p90": percentile(0.90),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "min": min(execution_times),
            "max": max(execution_times),
            "avg": sum(execution_times) / n
        }

    def get_slow_queries(
        self,
        threshold_seconds: float = 5.0,
        hours: int = 24,
        limit: int = 10
    ) -> List[QueryMetric]:
        """
        Find slow queries for optimization

        Args:
            threshold_seconds: Minimum execution time
            hours: Time period to search
            limit: Maximum results to return

        Returns:
            List of slow query metrics
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        slow_queries = [
            m for m in self.in_memory_metrics
            if m.timestamp >= cutoff and m.execution_time >= threshold_seconds
        ]

        # Sort by execution time (slowest first)
        slow_queries.sort(key=lambda m: m.execution_time, reverse=True)

        return slow_queries[:limit]

    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Analyze errors and failures

        Args:
            hours: Time period to analyze

        Returns:
            Dictionary of error metrics
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent_metrics = [m for m in self.in_memory_metrics if m.timestamp >= cutoff]

        failed_queries = [m for m in recent_metrics if not m.success]

        if not failed_queries:
            return {
                "total_errors": 0,
                "error_rate": 0,
                "errors_by_tool": {},
                "common_errors": []
            }

        # Group by tool
        errors_by_tool = defaultdict(int)
        for m in failed_queries:
            errors_by_tool[m.tool_used] += 1

        # Group by error message
        error_messages = defaultdict(int)
        for m in failed_queries:
            if m.error_message:
                error_messages[m.error_message] += 1

        # Top errors
        common_errors = sorted(
            error_messages.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return {
            "total_errors": len(failed_queries),
            "error_rate": len(failed_queries) / len(recent_metrics) if recent_metrics else 0,
            "errors_by_tool": dict(errors_by_tool),
            "common_errors": [{"message": msg, "count": count} for msg, count in common_errors]
        }

    def cleanup_old_data(self) -> int:
        """
        Remove old query logs

        Returns:
            Number of records deleted
        """
        if not self.db_session:
            return 0

        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)

        try:
            deleted = self.db_session.query(QueryLog).filter(
                QueryLog.timestamp < cutoff
            ).delete()
            self.db_session.commit()
            logger.info(f"Cleaned up {deleted} old query logs")
            return deleted
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            self.db_session.rollback()
            return 0

    def export_metrics(
        self,
        hours: int = 24,
        format: str = "json"
    ) -> str:
        """
        Export metrics for external analysis

        Args:
            hours: Time period to export
            format: Export format ("json" or "csv")

        Returns:
            Serialized metrics data
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent_metrics = [m for m in self.in_memory_metrics if m.timestamp >= cutoff]

        if format == "json":
            return json.dumps([asdict(m) for m in recent_metrics], default=str, indent=2)
        else:
            # CSV export
            if not recent_metrics:
                return ""

            header = ",".join(asdict(recent_metrics[0]).keys())
            rows = [",".join(str(v) for v in asdict(m).values()) for m in recent_metrics]
            return header + "\n" + "\n".join(rows)
