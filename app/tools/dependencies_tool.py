"""Service dependency graph tool."""
from __future__ import annotations

from typing import Any

from app.services.data_store import get_store
from app.tools.base import Tool, ToolResult


class ServiceDependenciesTool(Tool):
    name = "get_service_dependencies"
    description = (
        "Return the dependency graph for a service - which services it "
        "depends on (upstream) and which services depend on it "
        "(downstream). Essential for blast-radius analysis."
    )
    input_schema = {
        "type": "object",
        "properties": {"service": {"type": "string"}},
        "required": ["service"],
    }

    async def run(self, service: str, **_: Any) -> ToolResult:
        store = get_store()
        dep = store.get_dependencies(service)
        summary = (
            f"{service} depends on {len(dep.depends_on)} services and is "
            f"consumed by {len(dep.consumed_by)} services. Health: "
            f"{dep.health}."
        )
        return ToolResult(
            tool=self.name,
            input={"service": service},
            summary=summary,
            data={"dependency": dep.model_dump(mode="json")},
        )
