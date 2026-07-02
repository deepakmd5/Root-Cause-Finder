"""Tool registry - discovery + safe dispatch by name."""
from __future__ import annotations

from typing import Any

from app.core.exceptions import ToolExecutionError, UnknownToolError
from app.core.logging import get_logger
from app.tools.base import Tool, ToolResult

log = get_logger(__name__)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise UnknownToolError(f"Unknown tool: {name}")
        return self._tools[name]

    def list_specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    async def execute(self, name: str, tool_input: dict[str, Any]) -> ToolResult:
        tool = self.get(name)
        log.info("tool.execute.start", tool=name, input=tool_input)
        try:
            result = await tool.run(**tool_input)
        except UnknownToolError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("tool.execute.failed", tool=name)
            raise ToolExecutionError(name, str(exc)) from exc
        log.info("tool.execute.done", tool=name, summary=result.summary)
        return result


_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """Return the process-wide registry, populated on first access."""
    global _registry
    if _registry is None:
        _registry = _build_default_registry()
    return _registry


def _build_default_registry() -> ToolRegistry:
    from app.tools.aerospike_tool import QueryAerospikeTool
    from app.tools.database_tool import QueryDatabaseTool
    from app.tools.dependencies_tool import ServiceDependenciesTool
    from app.tools.deployments_tool import RecentDeploymentsTool
    from app.tools.logs_tool import SearchLogsTool
    from app.tools.metrics_tool import QueryMetricsTool
    from app.tools.similar_incidents_tool import FindSimilarIncidentsTool
    from app.tools.traces_tool import FetchTracesTool

    registry = ToolRegistry()
    registry.register(SearchLogsTool())
    registry.register(QueryMetricsTool())
    registry.register(RecentDeploymentsTool())
    registry.register(FetchTracesTool())
    registry.register(ServiceDependenciesTool())
    registry.register(FindSimilarIncidentsTool())
    registry.register(QueryDatabaseTool())
    registry.register(QueryAerospikeTool())
    return registry


def reset_registry() -> None:
    global _registry
    _registry = None
