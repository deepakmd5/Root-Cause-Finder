"""Base Tool abstractions."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Result of executing a tool.

    ``summary`` is a short string suitable for the agent's next-step
    reasoning. ``data`` is the full structured payload that will be
    stored in the normalized context.
    """

    tool: str
    input: dict[str, Any] = Field(default_factory=dict)
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)


class Tool(ABC):
    """Every investigative tool implements this contract."""

    name: str
    description: str
    input_schema: dict[str, Any] = {}

    @abstractmethod
    async def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool and return a :class:`ToolResult`."""

    def spec(self) -> dict[str, Any]:
        """The public spec advertised to the LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
