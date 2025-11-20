"""
Performance Optimizer - Generates optimization recommendations

Analyzes system metrics and suggests improvements.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class OptimizationPriority(Enum):
    """Priority levels for optimizations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class OptimizationRecommendation:
    """A single optimization recommendation"""
    title: str
    description: str
    priority: OptimizationPriority
    category: str  # "performance", "reliability", "cost", "security"
    estimated_impact: str  # Description of expected improvement
    action_items: List[str]
    metrics_affected: List[str]
    timestamp: datetime


class PerformanceOptimizer:
    """
    Analyzes system performance and generates recommendations

    Provides actionable insights for system optimization.
    """

    def __init__(self):
        """Initialize the optimizer"""
        self.thresholds = {
            "cache_hit_rate_min": 0.6,  # 60% minimum
            "error_rate_max": 0.05,  # 5% maximum
            "p95_latency_max": 5.0,  # 5 seconds
            "queries_per_hour_min": 10,  # Minimum for meaningful stats
        }

    def analyze_and_recommend(
        self,
        query_trends: Dict[str, Any],
        tool_usage: List[Any],
        performance_percentiles: Dict[str, float],
        error_summary: Dict[str, Any]
    ) -> List[OptimizationRecommendation]:
        """
        Analyze metrics and generate recommendations

        Args:
            query_trends: Query trend metrics
            tool_usage: Tool usage statistics
            performance_percentiles: Performance percentiles
            error_summary: Error analysis

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        # Check cache performance
        cache_recs = self._analyze_cache_performance(query_trends)
        recommendations.extend(cache_recs)

        # Check error rates
        error_recs = self._analyze_error_rates(error_summary, tool_usage)
        recommendations.extend(error_recs)

        # Check latency
        latency_recs = self._analyze_latency(performance_percentiles, query_trends)
        recommendations.extend(latency_recs)

        # Check tool distribution
        tool_recs = self._analyze_tool_usage(tool_usage, query_trends)
        recommendations.extend(tool_recs)

        # Sort by priority
        priority_order = {
            OptimizationPriority.CRITICAL: 0,
            OptimizationPriority.HIGH: 1,
            OptimizationPriority.MEDIUM: 2,
            OptimizationPriority.LOW: 3
        }
        recommendations.sort(key=lambda r: priority_order[r.priority])

        return recommendations

    def _analyze_cache_performance(
        self,
        query_trends: Dict[str, Any]
    ) -> List[OptimizationRecommendation]:
        """Analyze cache hit rates"""
        recommendations = []

        cache_hit_rate = query_trends.get("cache_hit_rate", 0)

        if cache_hit_rate < self.thresholds["cache_hit_rate_min"]:
            priority = OptimizationPriority.HIGH if cache_hit_rate < 0.4 else OptimizationPriority.MEDIUM

            recommendations.append(OptimizationRecommendation(
                title="Low Cache Hit Rate",
                description=f"Cache hit rate is {cache_hit_rate:.1%}, below the {self.thresholds['cache_hit_rate_min']:.0%} threshold. This indicates inefficient caching.",
                priority=priority,
                category="performance",
                estimated_impact="Improving cache hit rate to 70% could reduce average query time by 40-60%",
                action_items=[
                    "Review cache TTL settings and increase if appropriate",
                    "Analyze common query patterns and pre-warm cache",
                    "Consider increasing cache memory allocation",
                    "Implement semantic caching for similar queries",
                    "Review cache key generation strategy"
                ],
                metrics_affected=["cache_hit_rate", "avg_execution_time", "p95_latency"],
                timestamp=datetime.utcnow()
            ))

        return recommendations

    def _analyze_error_rates(
        self,
        error_summary: Dict[str, Any],
        tool_usage: List[Any]
    ) -> List[OptimizationRecommendation]:
        """Analyze error rates and patterns"""
        recommendations = []

        error_rate = error_summary.get("error_rate", 0)

        if error_rate > self.thresholds["error_rate_max"]:
            priority = OptimizationPriority.CRITICAL if error_rate > 0.15 else OptimizationPriority.HIGH

            common_errors = error_summary.get("common_errors", [])
            error_details = "\n".join([
                f"- {err['message']}: {err['count']} occurrences"
                for err in common_errors[:3]
            ])

            recommendations.append(OptimizationRecommendation(
                title="High Error Rate Detected",
                description=f"Error rate is {error_rate:.1%}, exceeding the {self.thresholds['error_rate_max']:.0%} threshold.\n\nMost common errors:\n{error_details}",
                priority=priority,
                category="reliability",
                estimated_impact="Reducing errors to <5% will improve user experience and system reliability",
                action_items=[
                    "Investigate root causes of top 3 error types",
                    "Add retry logic with exponential backoff",
                    "Improve error handling in failing tools",
                    "Add circuit breakers for unreliable dependencies",
                    "Review and fix validation errors"
                ],
                metrics_affected=["error_rate", "success_rate"],
                timestamp=datetime.utcnow()
            ))

        # Check tool-specific errors
        errors_by_tool = error_summary.get("errors_by_tool", {})
        for tool_name, error_count in errors_by_tool.items():
            tool_metrics = next((t for t in tool_usage if t.tool_name == tool_name), None)
            if tool_metrics:
                tool_error_rate = error_count / tool_metrics.total_calls if tool_metrics.total_calls > 0 else 0

                if tool_error_rate > 0.2:  # 20% error rate for specific tool
                    recommendations.append(OptimizationRecommendation(
                        title=f"High Error Rate in {tool_name} Tool",
                        description=f"The {tool_name} tool has a {tool_error_rate:.1%} error rate ({error_count}/{tool_metrics.total_calls} calls failed)",
                        priority=OptimizationPriority.HIGH,
                        category="reliability",
                        estimated_impact=f"Fixing {tool_name} errors could improve overall success rate by {(error_count/tool_metrics.total_calls)*100:.0f}%",
                        action_items=[
                            f"Review {tool_name} implementation for bugs",
                            f"Add better error handling to {tool_name} tool",
                            f"Check {tool_name} dependencies and timeouts",
                            f"Consider fallback strategies for {tool_name}",
                            "Add monitoring alerts for this tool"
                        ],
                        metrics_affected=["tool_success_rate", "overall_success_rate"],
                        timestamp=datetime.utcnow()
                    ))

        return recommendations

    def _analyze_latency(
        self,
        performance_percentiles: Dict[str, float],
        query_trends: Dict[str, Any]
    ) -> List[OptimizationRecommendation]:
        """Analyze query latency"""
        recommendations = []

        p95 = performance_percentiles.get("p95", 0)

        if p95 > self.thresholds["p95_latency_max"]:
            priority = OptimizationPriority.HIGH if p95 > 10 else OptimizationPriority.MEDIUM

            recommendations.append(OptimizationRecommendation(
                title="High P95 Latency",
                description=f"95th percentile latency is {p95:.2f}s, above the {self.thresholds['p95_latency_max']:.1f}s threshold. This means 5% of queries are taking longer than {p95:.1f}s.",
                priority=priority,
                category="performance",
                estimated_impact="Reducing P95 latency to <5s will improve user experience for slowest queries",
                action_items=[
                    "Identify and optimize slow queries (see slow query report)",
                    "Add query timeouts to prevent extremely slow operations",
                    "Consider parallel execution for independent operations",
                    "Optimize database queries and indexing",
                    "Review tool execution times and add concurrency",
                    "Scale up resources if consistently hitting limits"
                ],
                metrics_affected=["p95_latency", "p99_latency", "avg_execution_time"],
                timestamp=datetime.utcnow()
            ))

        # Check if average is also high
        avg = performance_percentiles.get("avg", 0)
        if avg > 2.0:  # 2 seconds average
            recommendations.append(OptimizationRecommendation(
                title="High Average Query Time",
                description=f"Average query time is {avg:.2f}s, which impacts overall system responsiveness",
                priority=OptimizationPriority.MEDIUM,
                category="performance",
                estimated_impact="Reducing average time by 50% will significantly improve user experience",
                action_items=[
                    "Profile query execution to identify bottlenecks",
                    "Implement request coalescing for similar queries",
                    "Add lazy loading where appropriate",
                    "Optimize tool implementations",
                    "Consider adding response streaming"
                ],
                metrics_affected=["avg_execution_time", "user_satisfaction"],
                timestamp=datetime.utcnow()
            ))

        return recommendations

    def _analyze_tool_usage(
        self,
        tool_usage: List[Any],
        query_trends: Dict[str, Any]
    ) -> List[OptimizationRecommendation]:
        """Analyze tool usage patterns"""
        recommendations = []

        if not tool_usage:
            return recommendations

        # Check for underutilized tools
        total_queries = query_trends.get("total_queries", 0)
        for tool in tool_usage:
            usage_rate = tool.total_calls / total_queries if total_queries > 0 else 0

            # If a tool is rarely used but has high latency, it's a candidate for removal
            if usage_rate < 0.05 and tool.avg_execution_time > 3.0:  # <5% usage, slow
                recommendations.append(OptimizationRecommendation(
                    title=f"Rarely Used Slow Tool: {tool.tool_name}",
                    description=f"The {tool.tool_name} tool is used in only {usage_rate:.1%} of queries but has high latency ({tool.avg_execution_time:.2f}s average)",
                    priority=OptimizationPriority.LOW,
                    category="cost",
                    estimated_impact="Optimizing or deprecating this tool could reduce maintenance overhead",
                    action_items=[
                        f"Review if {tool.tool_name} tool is still needed",
                        f"Consider optimizing {tool.tool_name} implementation",
                        f"Evaluate alternatives to {tool.tool_name}",
                        "Document use cases for this tool"
                    ],
                    metrics_affected=["tool_distribution", "maintenance_cost"],
                    timestamp=datetime.utcnow()
                ))

        # Check for heavily used tools that could benefit from optimization
        for tool in tool_usage:
            usage_rate = tool.total_calls / total_queries if total_queries > 0 else 0

            if usage_rate > 0.5 and tool.avg_execution_time > 1.0:  # >50% usage, room for improvement
                recommendations.append(OptimizationRecommendation(
                    title=f"Optimize Frequently Used Tool: {tool.tool_name}",
                    description=f"The {tool.tool_name} tool is used in {usage_rate:.1%} of queries with {tool.avg_execution_time:.2f}s average latency. Optimizing this tool will have significant impact.",
                    priority=OptimizationPriority.MEDIUM,
                    category="performance",
                    estimated_impact=f"Reducing {tool.tool_name} latency by 30% could improve overall system performance by {usage_rate*0.3*100:.0f}%",
                    action_items=[
                        f"Profile {tool.tool_name} to identify bottlenecks",
                        f"Add caching to {tool.tool_name} if applicable",
                        f"Optimize {tool.tool_name} algorithm or implementation",
                        f"Consider parallel execution in {tool.tool_name}",
                        "Add performance monitoring for this tool"
                    ],
                    metrics_affected=["avg_execution_time", "user_experience"],
                    timestamp=datetime.utcnow()
                ))

        return recommendations

    def generate_optimization_plan(
        self,
        recommendations: List[OptimizationRecommendation]
    ) -> Dict[str, Any]:
        """
        Generate a prioritized optimization plan

        Args:
            recommendations: List of recommendations

        Returns:
            Structured optimization plan
        """
        if not recommendations:
            return {
                "status": "healthy",
                "message": "No optimizations needed at this time",
                "recommendations": []
            }

        # Group by priority
        by_priority = {}
        for rec in recommendations:
            priority = rec.priority.value
            if priority not in by_priority:
                by_priority[priority] = []
            by_priority[priority].append({
                "title": rec.title,
                "description": rec.description,
                "category": rec.category,
                "impact": rec.estimated_impact,
                "actions": rec.action_items
            })

        # Calculate urgency score
        critical_count = len([r for r in recommendations if r.priority == OptimizationPriority.CRITICAL])
        high_count = len([r for r in recommendations if r.priority == OptimizationPriority.HIGH])

        if critical_count > 0:
            urgency = "critical"
            message = f"{critical_count} critical issues require immediate attention"
        elif high_count > 2:
            urgency = "high"
            message = f"{high_count} high-priority optimizations recommended"
        else:
            urgency = "moderate"
            message = f"{len(recommendations)} optimization opportunities identified"

        return {
            "status": urgency,
            "message": message,
            "total_recommendations": len(recommendations),
            "by_priority": by_priority,
            "generated_at": datetime.utcnow().isoformat()
        }
