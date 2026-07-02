"""Investigation state, agent step tracking, ReAct trace models."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from app.models.alert import Alert
from app.models.context import NormalizedContext
from app.models.rca import RCAReport


class InvestigationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


class StepType(str, Enum):
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL = "final"


class AgentStep(BaseModel):
    """One step in the agent's reasoning trace."""

    index: int
    type: StepType
    content: str
    tool: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output_summary: str | None = None
    duration_ms: int | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Investigation(BaseModel):
    """Full lifecycle record of an RCA investigation."""

    id: str = Field(default_factory=lambda: f"inv_{uuid4().hex[:12]}")
    alert: Alert
    status: InvestigationStatus = InvestigationStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    context: NormalizedContext | None = None
    steps: list[AgentStep] = Field(default_factory=list)
    report: RCAReport | None = None
    error: str | None = None

    def mark_started(self) -> None:
        self.started_at = datetime.now(timezone.utc)
        self.status = InvestigationStatus.RUNNING

    def mark_completed(self, report: RCAReport) -> None:
        self.report = report
        self.status = InvestigationStatus.COMPLETED
        self.finished_at = datetime.now(timezone.utc)

    def mark_failed(self, message: str) -> None:
        self.status = InvestigationStatus.FAILED
        self.error = message
        self.finished_at = datetime.now(timezone.utc)
