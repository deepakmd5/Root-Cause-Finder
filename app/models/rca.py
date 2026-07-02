"""Root Cause Analysis report and hypothesis models."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class RemediationPriority(str, Enum):
    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"


class RemediationAction(BaseModel):
    title: str
    description: str
    priority: RemediationPriority = RemediationPriority.SHORT_TERM
    owner_hint: str | None = None


class Hypothesis(BaseModel):
    """A candidate root cause with the evidence that supports it."""

    statement: str
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)


class RCAReport(BaseModel):
    """Final, structured output of an investigation."""

    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    headline: str
    summary: str
    primary_hypothesis: Hypothesis
    alternate_hypotheses: list[Hypothesis] = Field(default_factory=list)
    impacted_services: list[str] = Field(default_factory=list)
    detection_signals: list[str] = Field(default_factory=list)
    timeline: list[str] = Field(default_factory=list)
    remediation: list[RemediationAction] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    references: list[str] = Field(
        default_factory=list,
        description="Tool call ids / evidence pointers.",
    )
