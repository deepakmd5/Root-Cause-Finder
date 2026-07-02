"""Historical similar-incident lookup tool."""
from __future__ import annotations

from typing import Any

from app.services.data_store import get_store
from app.tools.base import Tool, ToolResult


class FindSimilarIncidentsTool(Tool):
    name = "find_similar_incidents"
    description = (
        "Search the historical incident knowledge base for past incidents "
        "similar to the current situation. Returns each with its "
        "documented root cause and resolution."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "service": {"type": "string"},
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional keywords to boost matches.",
            },
            "limit": {"type": "integer", "default": 5},
        },
        "required": ["service"],
    }

    async def run(
        self,
        service: str,
        keywords: list[str] | None = None,
        limit: int = 5,
        **_: Any,
    ) -> ToolResult:
        store = get_store()
        incidents = store.search_similar_incidents(
            service=service, keywords=keywords, limit=limit
        )
        if not incidents:
            summary = "No similar historical incidents found."
        else:
            top = incidents[0]
            summary = (
                f"Top match: {top.incident_id} (similarity="
                f"{top.similarity_score:.2f}) - {top.title}."
            )
        return ToolResult(
            tool=self.name,
            input={
                "service": service,
                "keywords": keywords or [],
                "limit": limit,
            },
            summary=summary,
            data={
                "incidents": [
                    i.model_dump(mode="json") for i in incidents
                ]
            },
        )
