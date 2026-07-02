"""Log search tool."""
from __future__ import annotations

from typing import Any

from app.services.data_store import get_store
from app.tools.base import Tool, ToolResult


class SearchLogsTool(Tool):
    name = "search_logs"
    description = (
        "Search structured application logs for one or more services. "
        "Supports filtering by log level and lookback window. Use this to "
        "find error signatures, stack traces, and warnings."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "services": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Service names to search.",
            },
            "levels": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Log levels (ERROR, WARN, INFO, DEBUG).",
            },
            "since_minutes": {
                "type": "integer",
                "default": 30,
            },
            "limit": {"type": "integer", "default": 25},
        },
        "required": ["services"],
    }

    async def run(
        self,
        services: list[str] | None = None,
        levels: list[str] | None = None,
        since_minutes: int = 30,
        limit: int = 25,
        **_: Any,
    ) -> ToolResult:
        store = get_store()
        logs = store.get_logs(
            services=services,
            levels=levels,
            since_minutes=since_minutes,
            limit=limit,
        )
        counts: dict[str, int] = {}
        for log in logs:
            counts[log.level] = counts.get(log.level, 0) + 1
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        summary = (
            f"Found {len(logs)} log entries "
            f"({breakdown or 'no matches'}) across {len(services or [])} "
            "service(s)."
        )
        return ToolResult(
            tool=self.name,
            input={
                "services": services or [],
                "levels": levels or [],
                "since_minutes": since_minutes,
            },
            summary=summary,
            data={"logs": [log.model_dump(mode="json") for log in logs]},
        )
