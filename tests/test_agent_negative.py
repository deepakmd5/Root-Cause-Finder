"""Negative-case and guardrail tests.

Layered on top of the tool-level and full-flow tests, these exercise
what happens when things go wrong:

* No service-specific telemetry exists (the hallucination guardrail
  must trigger and produce an "insufficient evidence" report rather
  than a confident-but-invented RCA).
* The agent runs out of its iteration budget.
* A tool raises during execution - the agent must record the failure
  and complete gracefully instead of crashing.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.agents.rca_agent import RCAAgent, _has_service_specific_evidence
from app.api.dependencies import (
    _shared_llm,
    _shared_service,
    reset_singletons,
)
from app.core.exceptions import ToolExecutionError
from app.llm.mock import MockLLM
from app.main import create_app
from app.models.alert import Alert, AlertSeverity, IncomingAlert
from app.models.context import NormalizedContext
from app.services.data_store import reset_store
from app.services.investigation_service import InvestigationService
from app.tools.base import Tool, ToolResult
from app.tools.registry import ToolRegistry, get_registry, reset_registry


@pytest.fixture
def client() -> TestClient:
    reset_store()
    reset_registry()
    reset_singletons()
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# Guardrail: no service-specific evidence must produce "insufficient evidence"
# ---------------------------------------------------------------------------


def test_guardrail_triggers_for_unknown_service(client: TestClient) -> None:
    """Unknown service -> tools return empty -> insufficient-evidence RCA."""
    payload = {
        "title": "Something broke",
        "service": "unheard-of-service",
        "environment": "production",
        "severity": "high",
        "source": "custom",
    }
    r = client.post("/alerts", json=payload)
    assert r.status_code == 201, r.text
    inv = r.json()
    assert inv["status"] == "completed"

    report = inv["report"]
    assert "Insufficient evidence" in report["headline"]
    assert report["primary_hypothesis"]["confidence"] <= 0.15
    assert report["confidence"] <= 0.15
    assert report["primary_hypothesis"]["supporting_evidence"] == []

    # Remediation should nudge the operator toward telemetry investigation,
    # not toward "rolling back" or other high-risk actions.
    remediation_titles = " ".join(
        a["title"].lower() for a in report["remediation"]
    )
    assert "telemetry" in remediation_titles or "lookback" in remediation_titles


def test_guardrail_helper_detects_missing_evidence() -> None:
    """Unit-level test of the guardrail predicate."""
    alert = Alert.from_incoming(
        IncomingAlert(
            title="x",
            service="ghost-service",
            severity=AlertSeverity.HIGH,
        )
    )
    ctx = NormalizedContext(alert=alert)
    assert _has_service_specific_evidence(ctx) is False


def test_guardrail_helper_accepts_service_specific_logs() -> None:
    from datetime import datetime, timezone

    from app.models.context import LogEntry

    alert = Alert.from_incoming(
        IncomingAlert(title="x", service="svc-a", severity=AlertSeverity.HIGH)
    )
    ctx = NormalizedContext(alert=alert)
    ctx.logs.append(
        LogEntry(
            timestamp=datetime.now(timezone.utc),
            service="svc-a",
            level="ERROR",
            message="boom",
        )
    )
    assert _has_service_specific_evidence(ctx) is True


# ---------------------------------------------------------------------------
# Agent budget: max_iterations = 1 with a plan that needs > 1 step -> failed
# ---------------------------------------------------------------------------


async def test_agent_budget_exceeded_marks_failed() -> None:
    reset_store()
    reset_registry()
    llm = MockLLM()
    registry = get_registry()

    normalizer_service = InvestigationService(llm=llm, registry=registry)
    incoming = IncomingAlert(
        title="High error rate on payments-service",
        service="payments-service",
        severity=AlertSeverity.CRITICAL,
    )
    alert = Alert.from_incoming(incoming)
    context = normalizer_service.normalizer.build(alert)
    from app.models.investigation import Investigation

    investigation = Investigation(alert=alert, context=context)

    # Force the agent to run out of budget - the mock plan needs ~7 steps.
    agent = RCAAgent(
        llm=llm, registry=registry, max_iterations=2, timeout_seconds=30
    )
    result = await agent.investigate(investigation)

    assert result.status.value == "failed"
    assert result.error and "iterations" in result.error.lower()
    assert result.report is None


# ---------------------------------------------------------------------------
# Tool failure: a tool raising should NOT crash the agent uncontrollably.
# ---------------------------------------------------------------------------


class ExplodingLogsTool(Tool):
    """Replacement for search_logs that always fails."""

    name = "search_logs"
    description = "search_logs stub that always raises"
    input_schema = {"type": "object", "properties": {}}

    async def run(self, **_):  # noqa: ANN003, ANN201
        raise RuntimeError("simulated downstream outage")


async def test_agent_survives_tool_failure() -> None:
    reset_store()
    reset_registry()

    # Rebuild the registry with an exploding search_logs tool:
    registry = ToolRegistry()
    registry.register(ExplodingLogsTool())
    from app.tools.deployments_tool import RecentDeploymentsTool
    from app.tools.metrics_tool import QueryMetricsTool

    # Add the *other* tools that the mock plan wants to call so it can
    # keep making progress after the failing one:
    registry.register(RecentDeploymentsTool())
    registry.register(QueryMetricsTool())

    llm = MockLLM()
    service = InvestigationService(llm=llm, registry=registry)
    alert = Alert.from_incoming(
        IncomingAlert(
            title="Rate spike",
            service="payments-service",
            severity=AlertSeverity.HIGH,
        )
    )
    context = service.normalizer.build(alert)

    from app.models.investigation import Investigation

    investigation = Investigation(alert=alert, context=context)

    agent = RCAAgent(llm=llm, registry=registry, max_iterations=8)
    result = await agent.investigate(investigation)

    # The exploding tool crashes the investigation deterministically,
    # but the agent must convert it into a structured 'failed' outcome,
    # NOT a raw traceback back to the caller.
    assert result.status.value == "failed"
    assert result.error and "search_logs" in result.error


def test_registry_raises_toolexecutionerror() -> None:
    """Direct-registry-level assertion of the wrapping behaviour."""
    reset_registry()
    reg = ToolRegistry()
    reg.register(ExplodingLogsTool())

    import asyncio

    with pytest.raises(ToolExecutionError):
        asyncio.get_event_loop().run_until_complete(
            reg.execute("search_logs", {})
        )
