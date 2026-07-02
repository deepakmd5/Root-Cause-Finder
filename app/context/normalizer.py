"""Vendor-agnostic normalized context builder.

Turns raw incoming alerts (which can arrive in many shapes) into a
canonical :class:`NormalizedContext`. The agent only ever sees the
normalized form; every source-specific quirk stops here.
"""
from __future__ import annotations

from app.core.logging import get_logger
from app.models.alert import Alert, AlertSource
from app.models.context import NormalizedContext
from app.services.data_store import (
    DEPENDENCY_GRAPH,
    REVERSE_DEPENDENCY_GRAPH,
)

log = get_logger(__name__)


class ContextNormalizer:
    """Builds the initial :class:`NormalizedContext` for an alert."""

    def __init__(self, lookback_minutes: int = 30) -> None:
        self.lookback_minutes = lookback_minutes

    def build(self, alert: Alert) -> NormalizedContext:
        tags = self._normalize_tags(alert)
        involved = self._infer_involved_services(alert)
        key_metrics = self._infer_key_metrics(alert)
        summary = self._summarize(alert, involved, key_metrics)

        ctx = NormalizedContext(
            alert=alert,
            lookback_minutes=self.lookback_minutes,
            tags=tags,
            involved_services=involved,
            key_metrics=key_metrics,
            summary=summary,
        )
        log.info(
            "context.normalized",
            alert_id=alert.id,
            service=alert.service,
            involved=involved,
        )
        return ctx

    # -- Helpers ----------------------------------------------------------

    def _normalize_tags(self, alert: Alert) -> dict[str, str]:
        """Merge labels + environment + severity into a uniform tag map."""
        tags: dict[str, str] = {}
        for k, v in alert.labels.items():
            tags[k.lower()] = str(v)
        tags["env"] = alert.environment
        tags["severity"] = alert.severity.value
        tags["source"] = alert.source.value
        tags["service"] = alert.service
        return tags

    def _infer_involved_services(self, alert: Alert) -> list[str]:
        """The alerting service plus its immediate neighbours."""
        primary = alert.service
        upstream = REVERSE_DEPENDENCY_GRAPH.get(primary, [])
        downstream = DEPENDENCY_GRAPH.get(primary, [])
        combined = [primary] + downstream + upstream
        # de-dupe while preserving order:
        seen: set[str] = set()
        ordered: list[str] = []
        for svc in combined:
            if svc not in seen:
                seen.add(svc)
                ordered.append(svc)
        return ordered

    def _infer_key_metrics(self, alert: Alert) -> list[str]:
        """Reasonable default set of metrics to watch based on source."""
        base = ["latency_p95_ms", "error_rate"]
        if alert.metric_name and alert.metric_name not in base:
            base.insert(0, alert.metric_name)
        if alert.source == AlertSource.PROMETHEUS:
            base.append("cpu_utilization")
        return base

    def _summarize(
        self,
        alert: Alert,
        involved: list[str],
        key_metrics: list[str],
    ) -> str:
        threshold = (
            f" (threshold: {alert.threshold})"
            if alert.threshold is not None
            else ""
        )
        val = (
            f" observed={alert.metric_value}"
            if alert.metric_value is not None
            else ""
        )
        metric_part = (
            f" Metric of interest: {alert.metric_name}{val}{threshold}."
            if alert.metric_name
            else ""
        )
        return (
            f"[{alert.severity.value.upper()}] {alert.title} on "
            f"'{alert.service}' ({alert.environment}) fired at "
            f"{alert.fired_at.isoformat() if alert.fired_at else 'n/a'}."
            f"{metric_part} "
            f"Blast-radius candidates: {', '.join(involved)}. "
            f"Key metrics: {', '.join(key_metrics)}."
        )
