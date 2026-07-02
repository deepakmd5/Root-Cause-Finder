"""Alert models - the entry point of every RCA workflow."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class AlertSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertSource(str, Enum):
    PROMETHEUS = "prometheus"
    DATADOG = "datadog"
    CLOUDWATCH = "cloudwatch"
    NEWRELIC = "newrelic"
    PAGERDUTY = "pagerduty"
    SYNTHETIC = "synthetic"
    CUSTOM = "custom"


class IncomingAlert(BaseModel):
    """Payload accepted by ``POST /alerts``.

    Deliberately permissive so it can accept alerts from many upstream
    systems. Missing fields will be defaulted by the normalizer.
    """

    title: str = Field(..., min_length=1, max_length=256)
    description: str | None = None
    service: str = Field(..., description="Owning service / component name")
    environment: str = Field(default="production")
    severity: AlertSeverity = AlertSeverity.HIGH
    source: AlertSource = AlertSource.CUSTOM
    labels: dict[str, str] = Field(default_factory=dict)
    metric_name: str | None = None
    metric_value: float | None = None
    threshold: float | None = None
    fired_at: datetime | None = None
    raw: dict[str, Any] = Field(
        default_factory=dict,
        description="Original vendor payload for auditability.",
    )


class Alert(IncomingAlert):
    """Alert with server-assigned identity and timestamps."""

    id: str = Field(default_factory=lambda: f"alrt_{uuid4().hex[:12]}")
    received_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @classmethod
    def from_incoming(cls, incoming: IncomingAlert) -> "Alert":
        now = datetime.now(timezone.utc)
        data = incoming.model_dump()
        data["fired_at"] = incoming.fired_at or now
        return cls(**data)
