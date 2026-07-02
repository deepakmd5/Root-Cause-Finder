"""Distributed trace fetch tool."""
from __future__ import annotations

from typing import Any

from app.services.data_store import get_store
from app.tools.base import Tool, ToolResult


class FetchTracesTool(Tool):
    name = "fetch_traces"
    description = (
        "Fetch distributed traces, optionally filtered by service and "
        "error-only. Traces show the request path across services and "
        "isolate where latency or errors originate."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "service": {"type": "string"},
            "error_only": {"type": "boolean", "default": True},
            "limit": {"type": "integer", "default": 10},
        },
    }

    async def run(
        self,
        service: str | None = None,
        error_only: bool = True,
        limit: int = 10,
        **_: Any,
    ) -> ToolResult:
        store = get_store()
        traces = store.get_traces(
            service=service, error_only=error_only, limit=limit
        )
        if not traces:
            summary = "No matching traces found."
        else:
            services_hit = sorted({t.service for t in traces})
            summary = (
                f"{len(traces)} trace span(s) returned across services: "
                + ", ".join(services_hit)
            )
        return ToolResult(
            tool=self.name,
            input={
                "service": service,
                "error_only": error_only,
                "limit": limit,
            },
            summary=summary,
            data={"traces": [t.model_dump(mode="json") for t in traces]},
        )
