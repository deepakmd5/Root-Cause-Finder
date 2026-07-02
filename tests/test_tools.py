"""Per-tool unit tests.

These exercise each tool in isolation against a freshly seeded
:class:`TelemetryStore`, verifying both a happy-path and an empty-result
path. This is the first layer of the testing strategy - before the
full ReAct agent flow is exercised, each tool must behave correctly on
its own.
"""
from __future__ import annotations

import pytest

from app.services.data_store import reset_store
from app.tools.dependencies_tool import ServiceDependenciesTool
from app.tools.deployments_tool import RecentDeploymentsTool
from app.tools.logs_tool import SearchLogsTool
from app.tools.metrics_tool import QueryMetricsTool
from app.tools.similar_incidents_tool import FindSimilarIncidentsTool
from app.tools.traces_tool import FetchTracesTool


@pytest.fixture(autouse=True)
def fresh_store() -> None:
    """Every test gets a deterministically-seeded telemetry store."""
    reset_store()


# --- search_logs ------------------------------------------------------------


async def test_search_logs_returns_seeded_errors() -> None:
    tool = SearchLogsTool()
    result = await tool.run(
        services=["payments-service"], levels=["ERROR"], since_minutes=60
    )
    assert result.tool == "search_logs"
    logs = result.data["logs"]
    assert len(logs) >= 1
    assert all(log["service"] == "payments-service" for log in logs)
    assert all(log["level"] == "ERROR" for log in logs)
    assert "ERROR=" in result.summary


async def test_search_logs_empty_for_unknown_service() -> None:
    tool = SearchLogsTool()
    result = await tool.run(services=["mystery-service"])
    assert result.data["logs"] == []
    assert "0 log entries" in result.summary


async def test_search_logs_respects_lookback_window() -> None:
    tool = SearchLogsTool()
    result = await tool.run(
        services=["payments-service"],
        levels=["ERROR", "WARN", "INFO"],
        since_minutes=1,  # too short to catch the seeded incident logs
    )
    assert result.data["logs"] == []


# --- query_metrics ---------------------------------------------------------


async def test_query_metrics_returns_error_rate_samples() -> None:
    tool = QueryMetricsTool()
    result = await tool.run(
        service="payments-service", metric="error_rate", since_minutes=60
    )
    samples = result.data["samples"]
    assert len(samples) > 0
    assert all(s["metric"] == "error_rate" for s in samples)
    assert "count=" in result.summary
    assert "avg=" in result.summary


async def test_query_metrics_empty_for_unknown_service() -> None:
    tool = QueryMetricsTool()
    result = await tool.run(service="does-not-exist", metric="latency_p95_ms")
    assert result.data["samples"] == []
    assert "No samples" in result.summary


async def test_query_metrics_all_metrics_when_none_specified() -> None:
    tool = QueryMetricsTool()
    result = await tool.run(service="payments-service", since_minutes=60)
    metrics = {s["metric"] for s in result.data["samples"]}
    # Baseline seeder emits latency, error_rate, cpu:
    assert {"latency_p95_ms", "error_rate", "cpu_utilization"} & metrics


# --- recent_deployments ----------------------------------------------------


async def test_recent_deployments_finds_the_payments_deploy() -> None:
    tool = RecentDeploymentsTool()
    result = await tool.run(service="payments-service", since_minutes=60)
    deploys = result.data["deployments"]
    assert len(deploys) == 1
    assert deploys[0]["version"] == "v3.7.0"
    assert "connection pool" in deploys[0]["change_summary"]


async def test_recent_deployments_empty_outside_window() -> None:
    tool = RecentDeploymentsTool()
    # The payments deploy is 18 min ago; ask for 5m only:
    result = await tool.run(service="payments-service", since_minutes=5)
    assert result.data["deployments"] == []
    assert "No deployments" in result.summary


async def test_recent_deployments_all_services() -> None:
    tool = RecentDeploymentsTool()
    result = await tool.run(since_minutes=48 * 60)  # 2 days
    services = {d["service"] for d in result.data["deployments"]}
    assert {"payments-service", "user-service", "checkout-service"} <= services


# --- fetch_traces ----------------------------------------------------------


async def test_fetch_traces_error_only() -> None:
    tool = FetchTracesTool()
    result = await tool.run(error_only=True)
    traces = result.data["traces"]
    assert len(traces) >= 1
    assert all(t["status"] == "error" for t in traces)


async def test_fetch_traces_filtered_by_service() -> None:
    tool = FetchTracesTool()
    result = await tool.run(service="payments-service", error_only=True)
    services = {t["service"] for t in result.data["traces"]}
    assert services == {"payments-service"}


async def test_fetch_traces_empty_for_unknown_service() -> None:
    tool = FetchTracesTool()
    result = await tool.run(service="mystery-service")
    assert result.data["traces"] == []
    assert result.summary == "No matching traces found."


# --- get_service_dependencies ---------------------------------------------


async def test_service_dependencies_known_service() -> None:
    tool = ServiceDependenciesTool()
    result = await tool.run(service="checkout-service")
    dep = result.data["dependency"]
    assert dep["service"] == "checkout-service"
    assert "payments-service" in dep["depends_on"]
    assert "api-gateway" in dep["consumed_by"]
    assert dep["health"] == "degraded"


async def test_service_dependencies_unknown_service() -> None:
    tool = ServiceDependenciesTool()
    result = await tool.run(service="ghost-service")
    dep = result.data["dependency"]
    assert dep["service"] == "ghost-service"
    assert dep["depends_on"] == []
    assert dep["consumed_by"] == []
    assert dep["health"] == "healthy"


# --- find_similar_incidents ------------------------------------------------


async def test_find_similar_incidents_ranks_matching_service_first() -> None:
    tool = FindSimilarIncidentsTool()
    result = await tool.run(
        service="payments-service", keywords=["pool", "timeout"]
    )
    incidents = result.data["incidents"]
    assert len(incidents) >= 1
    # The top match must be the payments incident, not the checkout one.
    assert incidents[0]["service"] == "payments-service"
    assert incidents[0]["similarity_score"] >= 0.9


async def test_find_similar_incidents_limit_respected() -> None:
    tool = FindSimilarIncidentsTool()
    result = await tool.run(service="anything", limit=2)
    assert len(result.data["incidents"]) <= 2


# --- Tool spec is well-formed for LLM consumption --------------------------


def test_every_tool_advertises_a_json_schema() -> None:
    tools = [
        SearchLogsTool(),
        QueryMetricsTool(),
        RecentDeploymentsTool(),
        FetchTracesTool(),
        ServiceDependenciesTool(),
        FindSimilarIncidentsTool(),
    ]
    for t in tools:
        spec = t.spec()
        assert spec["name"] == t.name
        assert spec["description"]
        assert spec["input_schema"]["type"] == "object"
