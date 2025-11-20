"""
Optimization Jobs - Background jobs for system maintenance

Provides automated optimization and maintenance tasks.
"""

import logging
import asyncio
import os
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


async def cache_warming_job():
    """
    Cache warming job - Pre-loads frequently accessed data into cache

    Runs nightly to optimize cache hit rates.
    """
    logger.info("Starting cache warming job...")

    try:
        # Import here to avoid circular dependencies
        import redis.asyncio as redis
        from rag import QdrantVectorStore

        # Connect to Redis
        redis_host = os.getenv("REDIS_HOST", "redis-master")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))

        redis_client = await redis.from_url(
            f"redis://{redis_host}:{redis_port}",
            encoding="utf-8",
            decode_responses=True
        )

        # Connect to Qdrant
        qdrant_host = os.getenv("QDRANT_HOST", "http://qdrant.vector-db:6333")
        embedding_endpoint = os.getenv("EMBEDDING_ENDPOINT", "http://192.168.1.100:8001")

        vector_store = QdrantVectorStore(
            collection_name="aeon_documents",
            qdrant_host=qdrant_host,
            embedding_endpoint=embedding_endpoint
        )

        # Get most common queries from analytics
        # For now, warm cache with sample common queries
        common_queries = [
            "what is aeon",
            "how to deploy",
            "configuration options",
            "api documentation",
            "troubleshooting guide"
        ]

        warmed = 0
        for query in common_queries:
            try:
                # Search and cache results
                results = await vector_store.search(query, limit=5)

                if results:
                    # Cache the results
                    cache_key = f"rag:query:{query}"
                    import json
                    await redis_client.setex(
                        cache_key,
                        3600,  # 1 hour TTL
                        json.dumps(results)
                    )
                    warmed += 1
                    logger.info(f"Warmed cache for query: {query}")

            except Exception as e:
                logger.warning(f"Failed to warm cache for '{query}': {e}")

        await redis_client.close()

        logger.info(f"Cache warming complete. Warmed {warmed} queries.")

    except Exception as e:
        logger.error(f"Cache warming job failed: {e}", exc_info=True)
        raise


async def analytics_aggregation_job():
    """
    Analytics aggregation job - Aggregates metrics for reporting

    Runs every 30 minutes to maintain up-to-date analytics.
    """
    logger.info("Starting analytics aggregation job...")

    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "postgres-postgresql"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            database=os.getenv("POSTGRES_DB", "aiplatform"),
            user=os.getenv("POSTGRES_USER", "aiuser"),
            password=os.getenv("POSTGRES_PASSWORD", "changeme")
        )

        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Aggregate query stats for the last hour
        cutoff = datetime.utcnow() - timedelta(hours=1)

        # Query performance aggregation
        cursor.execute("""
            INSERT INTO hourly_metrics (
                hour,
                total_queries,
                successful_queries,
                avg_execution_time,
                cache_hit_rate
            )
            SELECT
                date_trunc('hour', timestamp) as hour,
                COUNT(*) as total_queries,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_queries,
                AVG(execution_time) as avg_execution_time,
                SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END)::float / COUNT(*) as cache_hit_rate
            FROM query_logs
            WHERE timestamp >= %s
            GROUP BY hour
            ON CONFLICT (hour) DO UPDATE SET
                total_queries = EXCLUDED.total_queries,
                successful_queries = EXCLUDED.successful_queries,
                avg_execution_time = EXCLUDED.avg_execution_time,
                cache_hit_rate = EXCLUDED.cache_hit_rate
        """, (cutoff,))

        conn.commit()

        # Tool usage aggregation
        cursor.execute("""
            INSERT INTO tool_usage_stats (
                hour,
                tool_name,
                total_calls,
                successful_calls,
                avg_execution_time
            )
            SELECT
                date_trunc('hour', timestamp) as hour,
                tool_used as tool_name,
                COUNT(*) as total_calls,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_calls,
                AVG(execution_time) as avg_execution_time
            FROM query_logs
            WHERE timestamp >= %s
            GROUP BY hour, tool_used
            ON CONFLICT (hour, tool_name) DO UPDATE SET
                total_calls = EXCLUDED.total_calls,
                successful_calls = EXCLUDED.successful_calls,
                avg_execution_time = EXCLUDED.avg_execution_time
        """, (cutoff,))

        conn.commit()

        cursor.close()
        conn.close()

        logger.info("Analytics aggregation complete.")

    except Exception as e:
        logger.error(f"Analytics aggregation job failed: {e}", exc_info=True)
        # Don't raise - allow job to complete


async def data_cleanup_job():
    """
    Data cleanup job - Removes old data to free storage

    Runs weekly to clean up old logs and metrics.
    """
    logger.info("Starting data cleanup job...")

    try:
        import psycopg2

        retention_days = int(os.getenv("RETENTION_DAYS", "90"))
        cutoff = datetime.utcnow() - timedelta(days=retention_days)

        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "postgres-postgresql"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            database=os.getenv("POSTGRES_DB", "aiplatform"),
            user=os.getenv("POSTGRES_USER", "aiuser"),
            password=os.getenv("POSTGRES_PASSWORD", "changeme")
        )

        cursor = conn.cursor()

        # Delete old query logs
        cursor.execute(
            "DELETE FROM query_logs WHERE timestamp < %s",
            (cutoff,)
        )
        deleted_queries = cursor.rowcount

        # Delete old hourly metrics
        cursor.execute(
            "DELETE FROM hourly_metrics WHERE hour < %s",
            (cutoff,)
        )
        deleted_metrics = cursor.rowcount

        # Delete old tool usage stats
        cursor.execute(
            "DELETE FROM tool_usage_stats WHERE hour < %s",
            (cutoff,)
        )
        deleted_tool_stats = cursor.rowcount

        conn.commit()
        cursor.close()
        conn.close()

        logger.info(
            f"Data cleanup complete. Deleted: {deleted_queries} query logs, "
            f"{deleted_metrics} hourly metrics, {deleted_tool_stats} tool stats"
        )

    except Exception as e:
        logger.error(f"Data cleanup job failed: {e}", exc_info=True)
        raise


async def optimization_report_job():
    """
    Optimization report job - Generates daily optimization recommendations

    Analyzes system performance and generates actionable recommendations.
    """
    logger.info("Starting optimization report generation...")

    try:
        from analytics import AnalyticsTracker, PerformanceOptimizer
        import psycopg2
        import redis.asyncio as redis
        import json

        # Initialize components
        postgres_url = (
            f"postgresql://{os.getenv('POSTGRES_USER', 'aiuser')}:"
            f"{os.getenv('POSTGRES_PASSWORD', 'changeme')}@"
            f"{os.getenv('POSTGRES_HOST', 'postgres-postgresql')}:"
            f"{os.getenv('POSTGRES_PORT', '5432')}/"
            f"{os.getenv('POSTGRES_DB', 'aiplatform')}"
        )

        tracker = AnalyticsTracker(postgres_url=postgres_url)
        optimizer = PerformanceOptimizer()

        # Gather metrics
        query_trends = tracker.get_query_trends(hours=24)
        tool_usage = tracker.get_tool_usage_stats(hours=24)
        performance_percentiles = tracker.get_performance_percentiles(hours=24)
        error_summary = tracker.get_error_summary(hours=24)

        # Generate recommendations
        recommendations = optimizer.analyze_and_recommend(
            query_trends=query_trends,
            tool_usage=tool_usage,
            performance_percentiles=performance_percentiles,
            error_summary=error_summary
        )

        # Generate optimization plan
        plan = optimizer.generate_optimization_plan(recommendations)

        # Store report in Redis for API access
        redis_host = os.getenv("REDIS_HOST", "redis-master")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))

        redis_client = await redis.from_url(
            f"redis://{redis_host}:{redis_port}",
            encoding="utf-8",
            decode_responses=True
        )

        # Store latest report
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "metrics": {
                "query_trends": query_trends,
                "performance_percentiles": performance_percentiles,
                "error_summary": error_summary
            },
            "optimization_plan": plan
        }

        await redis_client.setex(
            "optimization:latest_report",
            86400,  # 24 hour TTL
            json.dumps(report, default=str)
        )

        await redis_client.close()

        logger.info(
            f"Optimization report complete. "
            f"Generated {len(recommendations)} recommendations. "
            f"Status: {plan['status']}"
        )

    except Exception as e:
        logger.error(f"Optimization report job failed: {e}", exc_info=True)
        raise
