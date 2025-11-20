"""
Analytics Module - Query tracking, performance monitoring, and optimization

Provides comprehensive analytics for:
- Query patterns and trends
- Tool usage statistics
- Performance metrics
- Cache effectiveness
- Optimization recommendations
"""

from .tracker import AnalyticsTracker, QueryMetric, ToolUsageMetric
from .optimizer import PerformanceOptimizer, OptimizationRecommendation
from .aggregator import MetricsAggregator, TimeSeriesMetric

__all__ = [
    "AnalyticsTracker",
    "QueryMetric",
    "ToolUsageMetric",
    "PerformanceOptimizer",
    "OptimizationRecommendation",
    "MetricsAggregator",
    "TimeSeriesMetric",
]
