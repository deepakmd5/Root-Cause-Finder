"""Metrics query tool."""
from __future__ import annotations

from statistics import mean
from typing import Any

from app.services.data_store import get_store
from app.tools.base import Tool, ToolResult


class QueryMetricsTool(Tool):
    name = "query_metrics"
    description = (
        "Query time-series metrics for a service. Common metrics include "
        "'latency_p95_ms', 'error_rate', 'cpu_utilization', "
        "'db_connection_pool_usage', 'db_connection_wait_ms'."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "service": {"type": "string"},
            "metric": {"type": "string"},
            "since_minutes": {"type": "integer", "default": 30},
            "limit": {"type": "integer", "default": 100},
        },
        "required": ["service"],
    }

    async def run(
        self,
        service: str,
        metric: str | None = None,
        since_minutes: int = 30,
        limit: int = 100,
        **_: Any,
    ) -> ToolResult:
        store = get_store()
        samples = store.get_metrics(
            service=service,
            metric=metric,
            since_minutes=since_minutes,
            limit=limit,
        )

        if not samples:
            summary = (
                f"No samples for {metric or '*'} on {service} in the last "
                f"{since_minutes}m."
            )
        else:
            values = [s.value for s in samples]
            summary = (
                f"{samples[0].metric} on {service}: "
                f"count={len(values)}, "
                f"min={min(values):.3f}, "
                f"avg={mean(values):.3f}, "
                f"max={max(values):.3f} "
                f"(unit={samples[0].unit or 'n/a'})."
            )

        return ToolResult(
            tool=self.name,
            input={
                "service": service,
                "metric": metric,
                "since_minutes": since_minutes,
            },
            summary=summary,
            data={"samples": [s.model_dump(mode="json") for s in samples]},
        )
