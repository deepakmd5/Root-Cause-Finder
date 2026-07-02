"""LLM adapter abstractions.

The agent talks to *any* LLM through :class:`LLMAdapter`. It asks the
adapter to make one of two structured decisions:

* ``USE_TOOL`` - continue investigating with a tool call.
* ``FINALIZE`` - produce the final RCA report.

Adapters are responsible for turning free-form model output into that
structured :class:`AgentDecision`.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, Field

from app.config import get_settings


class DecisionType(str, Enum):
    USE_TOOL = "use_tool"
    FINALIZE = "finalize"


class AgentDecision(BaseModel):
    type: DecisionType
    thought: str
    tool: str | None = None
    tool_input: dict[str, Any] = Field(default_factory=dict)
    # Only populated on FINALIZE:
    final_answer: dict[str, Any] | None = None


class LLMMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMAdapter(Protocol):
    """Protocol every LLM backend must implement."""

    name: str

    async def decide(
        self,
        messages: list[LLMMessage],
        available_tools: list[dict[str, Any]],
    ) -> AgentDecision:
        ...


def build_llm() -> LLMAdapter:
    """Factory that returns the configured LLM adapter."""
    from app.llm.mock import MockLLM
    from app.llm.openai_adapter import OpenAILLM

    settings = get_settings()

    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            # Fall back to mock when misconfigured; log a warning at startup.
            return MockLLM()
        return OpenAILLM(
            api_key=settings.openai_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    return MockLLM()
