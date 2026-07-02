"""In-memory synthetic telemetry store.

In a production build, each of these methods would be backed by a real
observability system (Elasticsearch, Prometheus, Jaeger, ArgoCD, etc).
Here they are seeded with realistic incident scenarios so the agent has
something meaningful to reason over.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Iterable

from app.models.context import (
    DeploymentEvent,
    LogEntry,
    MetricSample,
    ServiceDependency,
    SimilarIncident,
    TraceSpan,
)

# ---------------------------------------------------------------------------
# Service topology
# ---------------------------------------------------------------------------

SERVICES: list[str] = [
    "api-gateway",
    "checkout-service",
    "payments-service",
    "wallet-service",
    "user-service",
    "notification-service",
    "orders-db",
    "payments-db",
    "redis-cache",
]

DEPENDENCY_GRAPH: dict[str, list[str]] = {
    "api-gateway": ["checkout-service", "user-service", "wallet-service"],
    "checkout-service": ["payments-service", "orders-db", "redis-cache"],
    "payments-service": ["payments-db", "wallet-service"],
    "wallet-service": ["payments-db", "redis-cache"],
    "user-service": ["redis-cache"],
    "notification-service": [],
    "orders-db": [],
    "payments-db": [],
    "redis-cache": [],
}


def _reverse_edges(graph: dict[str, list[str]]) -> dict[str, list[str]]:
    reverse: dict[str, list[str]] = {svc: [] for svc in graph}
    for parent, children in graph.items():
        for child in children:
            reverse.setdefault(child, []).append(parent)
    return reverse


REVERSE_DEPENDENCY_GRAPH = _reverse_edges(DEPENDENCY_GRAPH)


# ---------------------------------------------------------------------------
# Data store
# ---------------------------------------------------------------------------


class TelemetryStore:
    """A deterministic, seeded, in-memory telemetry backend."""

    def __init__(self, seed: int = 1337) -> None:
        self._rng = random.Random(seed)
        self._now = datetime.now(timezone.utc)
        self.logs: list[LogEntry] = []
        self.metrics: list[MetricSample] = []
        self.deployments: list[DeploymentEvent] = []
        self.traces: list[TraceSpan] = []
        self.incidents: list[SimilarIncident] = []
        self._seed_all()

    # -- Seeding ------------------------------------------------------------

    def _seed_all(self) -> None:
        self._seed_baseline_metrics()
        self._seed_deployments()
        self._seed_incident_scenario()
        self._seed_historical_incidents()

    def _seed_baseline_metrics(self) -> None:
        """Steady-state metrics for the last hour."""
        for svc in SERVICES:
            for minute in range(60, 0, -1):
                ts = self._now - timedelta(minutes=minute)
                self.metrics.extend(
                    [
                        MetricSample(
                            timestamp=ts,
                            service=svc,
                            metric="latency_p95_ms",
                            value=round(self._rng.uniform(80, 140), 2),
                            unit="ms",
                        ),
                        MetricSample(
                            timestamp=ts,
                            service=svc,
                            metric="error_rate",
                            value=round(self._rng.uniform(0.001, 0.01), 4),
                            unit="ratio",
                        ),
                        MetricSample(
                            timestamp=ts,
                            service=svc,
                            metric="cpu_utilization",
                            value=round(self._rng.uniform(20, 55), 2),
                            unit="percent",
                        ),
                    ]
                )

    def _seed_deployments(self) -> None:
        self.deployments.extend(
            [
                DeploymentEvent(
                    timestamp=self._now - timedelta(hours=6),
                    service="user-service",
                    version="v2.4.1",
                    previous_version="v2.4.0",
                    author="alice@paytm",
                    change_summary="Refactor auth caching layer",
                ),
                DeploymentEvent(
                    timestamp=self._now - timedelta(minutes=18),
                    service="payments-service",
                    version="v3.7.0",
                    previous_version="v3.6.4",
                    author="bob@paytm",
                    change_summary=(
                        "Introduce new retry policy and connection pool "
                        "resizing for payments-db"
                    ),
                ),
                DeploymentEvent(
                    timestamp=self._now - timedelta(hours=26),
                    service="checkout-service",
                    version="v1.12.0",
                    previous_version="v1.11.3",
                    author="carol@paytm",
                    change_summary="Add promo code validation",
                ),
            ]
        )

    def _seed_incident_scenario(self) -> None:
        """Simulate a live payments incident triggered by a recent deploy.

        The signals are intentionally correlated so a competent agent
        should be able to conclude: the payments-service v3.7.0 deploy
        exhausted the payments-db connection pool, cascading errors up
        to checkout-service and the api-gateway.
        """
        incident_started = self._now - timedelta(minutes=15)

        # Elevated latency + error rate on payments-service & downstream
        for minute in range(15, 0, -1):
            ts = self._now - timedelta(minutes=minute)
            for svc in ("payments-service", "checkout-service", "api-gateway"):
                self.metrics.append(
                    MetricSample(
                        timestamp=ts,
                        service=svc,
                        metric="latency_p95_ms",
                        value=round(self._rng.uniform(650, 1400), 2),
                        unit="ms",
                    )
                )
                self.metrics.append(
                    MetricSample(
                        timestamp=ts,
                        service=svc,
                        metric="error_rate",
                        value=round(self._rng.uniform(0.08, 0.22), 4),
                        unit="ratio",
                    )
                )
            self.metrics.append(
                MetricSample(
                    timestamp=ts,
                    service="payments-db",
                    metric="db_connection_pool_usage",
                    value=round(self._rng.uniform(0.95, 1.0), 3),
                    unit="ratio",
                )
            )
            self.metrics.append(
                MetricSample(
                    timestamp=ts,
                    service="payments-db",
                    metric="db_connection_wait_ms",
                    value=round(self._rng.uniform(400, 1200), 2),
                    unit="ms",
                )
            )

        # Structured logs telling the same story
        self.logs.extend(
            [
                LogEntry(
                    timestamp=incident_started + timedelta(seconds=45),
                    service="payments-service",
                    level="ERROR",
                    message=(
                        "HikariCP - Connection is not available, request "
                        "timed out after 30000ms"
                    ),
                    trace_id="trc_9f3a2b",
                ),
                LogEntry(
                    timestamp=incident_started + timedelta(minutes=1),
                    service="payments-service",
                    level="ERROR",
                    message=(
                        "SQLTransientConnectionException: pool 'payments-db' "
                        "exhausted"
                    ),
                    trace_id="trc_9f3a2b",
                ),
                LogEntry(
                    timestamp=incident_started + timedelta(minutes=2),
                    service="checkout-service",
                    level="ERROR",
                    message=(
                        "Upstream call to payments-service failed: 504 "
                        "Gateway Timeout"
                    ),
                    trace_id="trc_9f3a2b",
                ),
                LogEntry(
                    timestamp=incident_started + timedelta(minutes=3),
                    service="api-gateway",
                    level="WARN",
                    message="Circuit breaker OPEN for payments-service",
                    trace_id="trc_9f3a2b",
                ),
                LogEntry(
                    timestamp=incident_started + timedelta(minutes=5),
                    service="payments-service",
                    level="INFO",
                    message=(
                        "Startup: connection pool max size increased to 5 "
                        "(previously 40) via v3.7.0 config change"
                    ),
                ),
            ]
        )

        # Traces showing where time is being spent
        self.traces.extend(
            [
                TraceSpan(
                    trace_id="trc_9f3a2b",
                    span_id="sp_001",
                    parent_span_id=None,
                    service="api-gateway",
                    operation="POST /checkout",
                    duration_ms=1180.4,
                    status="error",
                    error_message="upstream timeout",
                    timestamp=incident_started + timedelta(minutes=2),
                ),
                TraceSpan(
                    trace_id="trc_9f3a2b",
                    span_id="sp_002",
                    parent_span_id="sp_001",
                    service="checkout-service",
                    operation="charge_customer",
                    duration_ms=1120.9,
                    status="error",
                    error_message="payments-service 504",
                    timestamp=incident_started + timedelta(minutes=2),
                ),
                TraceSpan(
                    trace_id="trc_9f3a2b",
                    span_id="sp_003",
                    parent_span_id="sp_002",
                    service="payments-service",
                    operation="db.acquire_connection",
                    duration_ms=980.2,
                    status="error",
                    error_message="pool exhausted",
                    timestamp=incident_started + timedelta(minutes=2),
                ),
            ]
        )

    def _seed_historical_incidents(self) -> None:
        self.incidents.extend(
            [
                SimilarIncident(
                    incident_id="INC-2043",
                    occurred_at=self._now - timedelta(days=42),
                    service="payments-service",
                    title="Payments outage after connection pool tuning",
                    root_cause=(
                        "Deployment shrank HikariCP maxPoolSize from 40 to "
                        "5, causing thread starvation under normal load."
                    ),
                    resolution=(
                        "Rolled back deployment and restored maxPoolSize=40."
                    ),
                    similarity_score=0.92,
                ),
                SimilarIncident(
                    incident_id="INC-1988",
                    occurred_at=self._now - timedelta(days=90),
                    service="checkout-service",
                    title="Checkout latency spike",
                    root_cause="Redis cache eviction storm during promo drop",
                    resolution="Increased Redis memory and added TTL jitter",
                    similarity_score=0.41,
                ),
                SimilarIncident(
                    incident_id="INC-1720",
                    occurred_at=self._now - timedelta(days=180),
                    service="user-service",
                    title="Login failures after auth cache refactor",
                    root_cause="Cache key collision after refactor",
                    resolution="Hotfix reverted key schema",
                    similarity_score=0.28,
                ),
            ]
        )

    # -- Query API ---------------------------------------------------------

    def get_logs(
        self,
        services: Iterable[str] | None = None,
        levels: Iterable[str] | None = None,
        since_minutes: int = 30,
        limit: int = 50,
    ) -> list[LogEntry]:
        cutoff = self._now - timedelta(minutes=since_minutes)
        service_set = {s.lower() for s in services} if services else None
        level_set = {l.upper() for l in levels} if levels else None
        result = [
            log
            for log in self.logs
            if log.timestamp >= cutoff
            and (service_set is None or log.service.lower() in service_set)
            and (level_set is None or log.level.upper() in level_set)
        ]
        result.sort(key=lambda x: x.timestamp, reverse=True)
        return result[:limit]

    def get_metrics(
        self,
        service: str,
        metric: str | None = None,
        since_minutes: int = 30,
        limit: int = 200,
    ) -> list[MetricSample]:
        cutoff = self._now - timedelta(minutes=since_minutes)
        result = [
            m
            for m in self.metrics
            if m.service.lower() == service.lower()
            and m.timestamp >= cutoff
            and (metric is None or m.metric == metric)
        ]
        result.sort(key=lambda x: x.timestamp)
        return result[-limit:]

    def get_deployments(
        self,
        service: str | None = None,
        since_minutes: int = 240,
    ) -> list[DeploymentEvent]:
        cutoff = self._now - timedelta(minutes=since_minutes)
        result = [
            d
            for d in self.deployments
            if d.timestamp >= cutoff
            and (service is None or d.service.lower() == service.lower())
        ]
        result.sort(key=lambda x: x.timestamp, reverse=True)
        return result

    def get_traces(
        self,
        service: str | None = None,
        error_only: bool = True,
        limit: int = 20,
    ) -> list[TraceSpan]:
        result = [
            t
            for t in self.traces
            if (service is None or t.service.lower() == service.lower())
            and (not error_only or t.status == "error")
        ]
        result.sort(key=lambda x: x.timestamp, reverse=True)
        return result[:limit]

    def get_dependencies(self, service: str) -> ServiceDependency:
        return ServiceDependency(
            service=service,
            depends_on=DEPENDENCY_GRAPH.get(service, []),
            consumed_by=REVERSE_DEPENDENCY_GRAPH.get(service, []),
            health="degraded"
            if service in {"payments-service", "checkout-service", "payments-db"}
            else "healthy",
        )

    def search_similar_incidents(
        self,
        service: str,
        keywords: list[str] | None = None,
        limit: int = 5,
    ) -> list[SimilarIncident]:
        keywords = [k.lower() for k in (keywords or [])]

        def score(inc: SimilarIncident) -> float:
            score = inc.similarity_score
            if inc.service.lower() == service.lower():
                score += 0.05
            for kw in keywords:
                if kw in inc.title.lower() or kw in inc.root_cause.lower():
                    score += 0.02
            return min(score, 1.0)

        ranked = sorted(self.incidents, key=score, reverse=True)
        return ranked[:limit]


# Global singleton -----------------------------------------------------------

_store: TelemetryStore | None = None


def get_store() -> TelemetryStore:
    global _store
    if _store is None:
        _store = TelemetryStore()
    return _store


def reset_store(seed: int = 1337) -> TelemetryStore:
    """Useful for tests."""
    global _store
    _store = TelemetryStore(seed=seed)
    return _store
