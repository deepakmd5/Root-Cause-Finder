"""Domain-specific exception hierarchy."""
from __future__ import annotations


class RCAError(Exception):
    """Base error for the RCA application."""


class InvestigationNotFoundError(RCAError):
    """Raised when an investigation id cannot be resolved."""


class ToolExecutionError(RCAError):
    """Raised when a tool fails during execution."""

    def __init__(self, tool_name: str, message: str) -> None:
        super().__init__(f"[{tool_name}] {message}")
        self.tool_name = tool_name


class UnknownToolError(RCAError):
    """Raised when an agent requests an unregistered tool."""


class LLMError(RCAError):
    """Raised when the LLM adapter fails."""


class AgentBudgetExceeded(RCAError):
    """Raised when the agent exhausts its iteration or time budget."""
