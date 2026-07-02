"""Normalized context that the agent reasons over."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.models.alert import Alert


class LogEntry(BaseModel):
    timestamp: datetime
    service: str
    level: str
    message: str
    trace_id: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class MetricSample(BaseModel):
    timestamp: datetime
    service: str
    metric: str
    value: float
    unit: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)


class DeploymentEvent(BaseModel):
    timestamp: datetime
    service: str
    version: str
    previous_version: str | None = None
    author: str
    change_summary: str
    rollback: bool = False


class TraceSpan(BaseModel):
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    service: str
    operation: str
    duration_ms: float
    status: str  # "ok" | "error"
    error_message: str | None = None
    timestamp: datetime


class ServiceDependency(BaseModel):
    service: str
    depends_on: list[str] = Field(default_factory=list)
    consumed_by: list[str] = Field(default_factory=list)
    health: str = "healthy"  # healthy | degraded | down


class SimilarIncident(BaseModel):
    incident_id: str
    occurred_at: datetime
    service: str
    title: str
    root_cause: str
    resolution: str
    similarity_score: float = Field(ge=0.0, le=1.0)


class DbRecord(BaseModel):
    """A single result set returned by the ``query_database`` tool.

    Stored on the context so the agent (and every downstream consumer)
    can trace which rows came from which allowlisted query with which
    parameters.
    """

    query_name: str
    params: dict[str, Any] = Field(default_factory=dict)
    row_count: int
    rows: list[dict[str, Any]] = Field(default_factory=list)
    fetched_at: datetime | None = None
    available: bool = True
    error: str | None = None


class AerospikeRecord(BaseModel):
    """A single key lookup returned by the ``query_aerospike`` tool.

    Kept on the context so the agent (and downstream consumers) can
    trace which allowlisted operation produced which cached record.
    ``found=False`` means the key was queried but no record existed -
    still valuable evidence (e.g. "session was already expired").
    """

    operation: str
    params: dict[str, Any] = Field(default_factory=dict)
    namespace: str | None = None
    set_name: str | None = None
    key: str | None = None
    found: bool = False
    record: dict[str, Any] | None = None
    fetched_at: datetime | None = None
    available: bool = True
    error: str | None = None


class NormalizedContext(BaseModel):
    """Vendor-agnostic view of an incident.

    Every source (Datadog, Prometheus, CloudWatch, PagerDuty, ...) is
    projected onto this shape before the agent ever sees it.
    """

    alert: Alert
    lookback_minutes: int = 30
    tags: dict[str, str] = Field(default_factory=dict)
    involved_services: list[str] = Field(default_factory=list)
    key_metrics: list[str] = Field(default_factory=list)
    summary: str = ""
    # Populated lazily by tools as the investigation progresses:
    logs: list[LogEntry] = Field(default_factory=list)
    metrics: list[MetricSample] = Field(default_factory=list)
    deployments: list[DeploymentEvent] = Field(default_factory=list)
    traces: list[TraceSpan] = Field(default_factory=list)
    dependencies: list[ServiceDependency] = Field(default_factory=list)
    similar_incidents: list[SimilarIncident] = Field(default_factory=list)
    db_records: list[DbRecord] = Field(default_factory=list)
    aerospike_records: list[AerospikeRecord] = Field(default_factory=list)
