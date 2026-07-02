"""End-to-end smoke tests using FastAPI's TestClient."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import reset_singletons
from app.main import create_app
from app.services.data_store import reset_store
from app.tools.registry import reset_registry


@pytest.fixture(scope="module")
def client() -> TestClient:
    reset_store()
    reset_registry()
    reset_singletons()
    app = create_app()
    return TestClient(app)


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_readiness(client: TestClient) -> None:
    r = client.get("/readiness")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["llm_provider"] == "mock"
    assert "search_logs" in body["tools"]
    assert "query_metrics" in body["tools"]


def test_ingest_alert_completes_full_investigation(client: TestClient) -> None:
    payload = {
        "title": "High error rate on payments-service",
        "description": "5xx errors spiked above threshold",
        "service": "payments-service",
        "environment": "production",
        "severity": "critical",
        "source": "prometheus",
        "metric_name": "error_rate",
        "metric_value": 0.19,
        "threshold": 0.05,
        "labels": {"team": "payments"},
    }
    r = client.post("/alerts", json=payload)
    assert r.status_code == 201, r.text

    inv = r.json()
    assert inv["status"] == "completed"
    assert inv["alert"]["service"] == "payments-service"
    assert inv["report"] is not None

    report = inv["report"]
    assert report["headline"]
    assert report["primary_hypothesis"]["confidence"] > 0
    assert len(report["remediation"]) >= 1

    # The agent should have used multiple tools:
    action_steps = [s for s in inv["steps"] if s["type"] == "action"]
    tool_names = {s["tool"] for s in action_steps}
    assert {"search_logs", "query_metrics", "recent_deployments"} <= tool_names

    # The normalized context should have been enriched:
    ctx = inv["context"]
    assert len(ctx["logs"]) > 0
    assert len(ctx["metrics"]) > 0
    assert len(ctx["deployments"]) >= 1
    assert "payments-service" in ctx["involved_services"]


def test_list_and_get_investigation(client: TestClient) -> None:
    r = client.get("/investigations")
    assert r.status_code == 200
    listing = r.json()
    assert len(listing) >= 1

    inv_id = listing[0]["id"]
    r = client.get(f"/investigations/{inv_id}")
    assert r.status_code == 200
    assert r.json()["id"] == inv_id


def test_get_missing_investigation_returns_404(client: TestClient) -> None:
    r = client.get("/investigations/does-not-exist")
    assert r.status_code == 404
