"""Context normalizer unit tests.

The normalizer is the layer that translates vendor-specific alert
payloads into the single :class:`NormalizedContext` the agent reasons
over. It's the choke point where alert-shape variations end.
"""
from __future__ import annotations

from app.context.normalizer import ContextNormalizer
from app.models.alert import Alert, AlertSeverity, AlertSource, IncomingAlert


def _alert(**overrides) -> Alert:  # noqa: ANN003
    base = dict(
        title="Payments error rate breached SLO",
        description="5xx errors on payments-service crossed 5%",
        service="payments-service",
        environment="production",
        severity=AlertSeverity.CRITICAL,
        source=AlertSource.PROMETHEUS,
        metric_name="error_rate",
        metric_value=0.19,
        threshold=0.05,
        labels={"team": "payments", "region": "ap-south-1"},
    )
    base.update(overrides)
    return Alert.from_incoming(IncomingAlert(**base))


def test_normalizer_infers_blast_radius_from_topology() -> None:
    """Involved services include upstream + downstream neighbours."""
    ctx = ContextNormalizer().build(_alert())
    assert ctx.involved_services[0] == "payments-service"
    # payments-service depends_on payments-db + wallet-service:
    assert "payments-db" in ctx.involved_services
    assert "wallet-service" in ctx.involved_services
    # payments-service is consumed_by checkout-service:
    assert "checkout-service" in ctx.involved_services


def test_normalizer_flattens_labels_into_tags() -> None:
    ctx = ContextNormalizer().build(_alert())
    assert ctx.tags["team"] == "payments"
    assert ctx.tags["region"] == "ap-south-1"
    assert ctx.tags["env"] == "production"
    assert ctx.tags["severity"] == "critical"
    assert ctx.tags["source"] == "prometheus"
    assert ctx.tags["service"] == "payments-service"


def test_normalizer_key_metrics_include_alert_metric() -> None:
    ctx = ContextNormalizer().build(_alert(metric_name="db_pool_saturation"))
    # The alerting metric jumps to the front:
    assert ctx.key_metrics[0] == "db_pool_saturation"
    assert "latency_p95_ms" in ctx.key_metrics
    assert "error_rate" in ctx.key_metrics


def test_normalizer_summary_carries_severity_and_service() -> None:
    ctx = ContextNormalizer().build(_alert())
    assert "[CRITICAL]" in ctx.summary
    assert "payments-service" in ctx.summary
    assert "error_rate" in ctx.summary


def test_normalizer_handles_unknown_service_gracefully() -> None:
    """Unknown services must still normalize without exceptions."""
    ctx = ContextNormalizer().build(_alert(service="brand-new-service"))
    # Even if topology doesn't know it, the service itself is always
    # in the involved list.
    assert ctx.involved_services[0] == "brand-new-service"
    assert ctx.tags["service"] == "brand-new-service"


def test_normalizer_never_leaks_none_into_summary() -> None:
    """Alerts without metric metadata should still produce a valid summary."""
    ctx = ContextNormalizer().build(
        _alert(metric_name=None, metric_value=None, threshold=None)
    )
    assert "None" not in ctx.summary
