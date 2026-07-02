"""Recent deployments tool."""
from __future__ import annotations

from typing import Any

from app.services.data_store import get_store
from app.tools.base import Tool, ToolResult


class RecentDeploymentsTool(Tool):
    name = "recent_deployments"
    description = (
        "List recent deployments, optionally filtered by service. Useful "
        "for change-correlation - a spike that starts near a deploy time "
        "is often caused by that deploy."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "service": {
                "type": "string",
                "description": "Optional service filter.",
            },
            "since_minutes": {
                "type": "integer",
                "default": 240,
                "description": "How far back to look (minutes).",
            },
        },
    }

    async def run(
        self,
        service: str | None = None,
        since_minutes: int = 240,
        **_: Any,
    ) -> ToolResult:
        store = get_store()
        deploys = store.get_deployments(
            service=service, since_minutes=since_minutes
        )
        if not deploys:
            summary = "No deployments in the requested window."
        else:
            latest = deploys[0]
            summary = (
                f"{len(deploys)} deployment(s) in the last "
                f"{since_minutes}m. Most recent: {latest.service} "
                f"{latest.version} at {latest.timestamp.isoformat()}."
            )
        return ToolResult(
            tool=self.name,
            input={"service": service, "since_minutes": since_minutes},
            summary=summary,
            data={
                "deployments": [d.model_dump(mode="json") for d in deploys]
            },
        )
