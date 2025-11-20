"""
Metrics Aggregator - Time-series metrics and aggregation

Provides time-series analytics and trend analysis.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesMetric:
    """A time-series data point"""
    timestamp: datetime
    metric_name: str
    value: float
    tags: Dict[str, str]


class MetricsAggregator:
    """
    Aggregates metrics over time periods

    Provides time-series analytics for trend detection.
    """

    def __init__(self):
        """Initialize the aggregator"""
        self.time_series_data: List[TimeSeriesMetric] = []
        self.max_data_points = 100000  # Keep last 100k points

    def record_metric(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a metric value

        Args:
            metric_name: Name of the metric
            value: Metric value
            tags: Optional tags for grouping
        """
        metric = TimeSeriesMetric(
            timestamp=datetime.utcnow(),
            metric_name=metric_name,
            value=value,
            tags=tags or {}
        )

        self.time_series_data.append(metric)

        # Trim if too large
        if len(self.time_series_data) > self.max_data_points:
            self.time_series_data = self.time_series_data[-self.max_data_points:]

    def aggregate_by_hour(
        self,
        metric_name: str,
        hours: int = 24,
        aggregation: str = "avg"
    ) -> List[Dict[str, Any]]:
        """
        Aggregate metrics by hour

        Args:
            metric_name: Metric to aggregate
            hours: Number of hours to include
            aggregation: Aggregation function ("avg", "sum", "min", "max", "count")

        Returns:
            List of hourly aggregated values
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        # Filter relevant metrics
        relevant_metrics = [
            m for m in self.time_series_data
            if m.metric_name == metric_name and m.timestamp >= cutoff
        ]

        if not relevant_metrics:
            return []

        # Group by hour
        hourly_buckets = defaultdict(list)
        for metric in relevant_metrics:
            # Round down to hour
            hour = metric.timestamp.replace(minute=0, second=0, microsecond=0)
            hourly_buckets[hour].append(metric.value)

        # Aggregate
        results = []
        for hour in sorted(hourly_buckets.keys()):
            values = hourly_buckets[hour]

            if aggregation == "avg":
                aggregated_value = sum(values) / len(values)
            elif aggregation == "sum":
                aggregated_value = sum(values)
            elif aggregation == "min":
                aggregated_value = min(values)
            elif aggregation == "max":
                aggregated_value = max(values)
            elif aggregation == "count":
                aggregated_value = len(values)
            else:
                aggregated_value = sum(values) / len(values)  # Default to avg

            results.append({
                "timestamp": hour.isoformat(),
                "value": aggregated_value,
                "data_points": len(values)
            })

        return results

    def detect_trends(
        self,
        metric_name: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Detect trends in a metric

        Args:
            metric_name: Metric to analyze
            hours: Time period to analyze

        Returns:
            Trend analysis
        """
        hourly_data = self.aggregate_by_hour(metric_name, hours)

        if len(hourly_data) < 2:
            return {
                "trend": "insufficient_data",
                "direction": "unknown",
                "change_rate": 0
            }

        values = [d["value"] for d in hourly_data]

        # Simple linear regression
        n = len(values)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(values) / n

        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator

        # Determine trend
        if abs(slope) < 0.01:  # Threshold for "stable"
            trend = "stable"
            direction = "flat"
        elif slope > 0:
            trend = "increasing"
            direction = "up"
        else:
            trend = "decreasing"
            direction = "down"

        # Calculate percent change
        if values[0] != 0:
            percent_change = ((values[-1] - values[0]) / values[0]) * 100
        else:
            percent_change = 0

        return {
            "trend": trend,
            "direction": direction,
            "slope": slope,
            "percent_change": percent_change,
            "current_value": values[-1],
            "previous_value": values[0],
            "data_points": len(values)
        }

    def get_metric_summary(
        self,
        metric_name: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get summary statistics for a metric

        Args:
            metric_name: Metric to summarize
            hours: Time period

        Returns:
            Summary statistics
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        values = [
            m.value for m in self.time_series_data
            if m.metric_name == metric_name and m.timestamp >= cutoff
        ]

        if not values:
            return {
                "count": 0,
                "avg": 0,
                "min": 0,
                "max": 0,
                "sum": 0
            }

        return {
            "count": len(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "sum": sum(values),
            "latest": values[-1] if values else 0
        }

    def compare_periods(
        self,
        metric_name: str,
        current_hours: int = 24,
        previous_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Compare current period to previous period

        Args:
            metric_name: Metric to compare
            current_hours: Current period duration
            previous_hours: Previous period duration

        Returns:
            Comparison results
        """
        now = datetime.utcnow()

        # Current period
        current_cutoff = now - timedelta(hours=current_hours)
        current_values = [
            m.value for m in self.time_series_data
            if m.metric_name == metric_name and m.timestamp >= current_cutoff
        ]

        # Previous period
        previous_start = now - timedelta(hours=current_hours + previous_hours)
        previous_end = now - timedelta(hours=current_hours)
        previous_values = [
            m.value for m in self.time_series_data
            if m.metric_name == metric_name
            and m.timestamp >= previous_start
            and m.timestamp < previous_end
        ]

        # Calculate averages
        current_avg = sum(current_values) / len(current_values) if current_values else 0
        previous_avg = sum(previous_values) / len(previous_values) if previous_values else 0

        # Calculate change
        if previous_avg != 0:
            percent_change = ((current_avg - previous_avg) / previous_avg) * 100
        else:
            percent_change = 0

        # Determine if improvement
        # Assuming lower values are better for most metrics (latency, errors)
        # This should be configurable per metric
        is_improvement = percent_change < 0  # Lower is better

        return {
            "current_period": {
                "avg": current_avg,
                "count": len(current_values),
                "hours": current_hours
            },
            "previous_period": {
                "avg": previous_avg,
                "count": len(previous_values),
                "hours": previous_hours
            },
            "change": {
                "absolute": current_avg - previous_avg,
                "percent": percent_change,
                "direction": "up" if percent_change > 0 else "down" if percent_change < 0 else "flat",
                "is_improvement": is_improvement
            }
        }
