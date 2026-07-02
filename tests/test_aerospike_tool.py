"""Tests for the ``query_aerospike`` tool and the ``AerospikeClient``.

We never touch a real Aerospike cluster here (the official client is
a C extension whose wheels vary by OS). Instead we substitute a
:class:`FakeAerospikeClient` for the process-wide singleton, so the
tool exercises its full happy-path (bind key, execute, summarize,
serialize) and every failure branch (unconfigured, not connected,
invalid input, missing param, downstream error).

A single end-to-end test drives a full alert through the agent to
prove that the Aerospike step gets scheduled first when the alert
carries a ``transactionId`` label, and that the returned record
satisfies the hallucination guardrail.
"""
from __future__ import annotations

from typing import Any

import pytest

from app.agents.rca_agent import RCAAgent
from app.llm.mock import MockLLM
from app.models.alert import Alert, AlertSeverity, IncomingAlert
from app.models.investigation import Investigation, StepType
from app.services.aerospike_client import (
    AerospikeClient,
    AerospikeUnavailable,
    reset_aerospike_client,
    set_aerospike_client,
)
from app.services.data_store import reset_store
from app.services.database import reset_db_client, set_db_client
from app.services.investigation_service import InvestigationService
from app.tools.aerospike_tool import (
    ALLOWED_OPERATIONS,
    OperationValidationError,
    QueryAerospikeTool,
    _jsonable_record,
)
from app.tools.registry import get_registry, reset_registry
from tests.test_database_tool import FakeDatabaseClient


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeAerospikeClient(AerospikeClient):
    """Test double that never talks to a real Aerospike cluster.

    Records every lookup on ``self.calls`` and returns whatever the
    caller pre-populated in ``self.store``, keyed by
    ``(set_name, key)``.
    """

    def __init__(
        self,
        store: dict[tuple[str, str], dict[str, Any]] | None = None,
        *,
        raise_on_get: Exception | None = None,
        namespace: str = "insurance",
    ) -> None:
        # Skip parent __init__ - avoids any C-library import path.
        self.store = store if store is not None else {}
        self._raise = raise_on_get
        self.namespace = namespace
        self.calls: list[tuple[str, str, str | None]] = []  # (set, key, ns)

    @property
    def is_configured(self) -> bool:  # type: ignore[override]
        return True

    @property
    def is_connected(self) -> bool:  # type: ignore[override]
        return True

    async def connect(self) -> None:  # noqa: D401
        return None

    async def close(self) -> None:
        return None

    async def get(
        self,
        set_name: str,
        key: str,
        *,
        namespace: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any] | None:
        self.calls.append((set_name, key, namespace))
        if self._raise is not None:
            raise self._raise
        return self.store.get((set_name, key))


class UnconfiguredFakeAerospikeClient(AerospikeClient):
    """AEROSPIKE_HOSTS not set - the tool must short-circuit."""

    def __init__(self) -> None:
        pass

    @property
    def is_configured(self) -> bool:  # type: ignore[override]
        return False

    @property
    def is_connected(self) -> bool:  # type: ignore[override]
        return False

    async def get(self, *args, **kwargs):  # noqa: ANN201, ANN002, ANN003
        raise AssertionError("get() must not be called when unconfigured")


class DisconnectedFakeAerospikeClient(AerospikeClient):
    """Hosts + namespace set but the cluster never came up."""

    def __init__(self) -> None:
        pass

    @property
    def is_configured(self) -> bool:  # type: ignore[override]
        return True

    @property
    def is_connected(self) -> bool:  # type: ignore[override]
        return False

    async def get(self, *args, **kwargs):  # noqa: ANN201, ANN002, ANN003
        raise AssertionError("get() must not be called when disconnected")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolate_aerospike_singleton():
    """Every test starts with a clean process-wide Aerospike slot."""
    reset_aerospike_client()
    yield
    reset_aerospike_client()


# ---------------------------------------------------------------------------
# Row/record serialization
# ---------------------------------------------------------------------------


def test_jsonable_record_coerces_bytes_and_sets() -> None:
    raw = {
        "bins": {
            "status": "FAILED",
            "tags": {"a", "b"},                # set
            "payload": b"hello",                # bytes
            "nested": {"list": [1, 2, {"k": b"v"}]},
        },
        "meta": {"ttl": 300, "gen": 4},
    }
    clean = _jsonable_record(raw)
    assert clean["meta"] == {"ttl": 300, "gen": 4}
    assert clean["bins"]["status"] == "FAILED"
    assert sorted(clean["bins"]["tags"]) == ["a", "b"]
    assert clean["bins"]["payload"] == "hello"
    assert clean["bins"]["nested"]["list"][2]["k"] == "v"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_query_aerospike_advertises_all_allowlisted_operations() -> None:
    tool = QueryAerospikeTool()
    spec = tool.spec()

    assert spec["name"] == "query_aerospike"
    assert spec["description"]

    schema = spec["input_schema"]
    assert schema["type"] == "object"
    assert schema["required"] == ["operation"]

    op_prop = schema["properties"]["operation"]
    assert set(op_prop["enum"]) == set(ALLOWED_OPERATIONS.keys())
    for name in [
        "policy_cache_get",
        "transaction_state_get",
        "idempotency_get",
    ]:
        assert name in op_prop["enum"]


# ---------------------------------------------------------------------------
# Tool.run() - happy path
# ---------------------------------------------------------------------------


async def test_query_aerospike_returns_record_when_found() -> None:
    fake = FakeAerospikeClient(
        store={
            ("tx_state", "TX123"): {
                "bins": {
                    "state": "PENDING",
                    "attempts": 3,
                    "last_error": "UPSTREAM_TIMEOUT",
                },
                "meta": {"ttl": 180, "gen": 4},
            }
        }
    )
    set_aerospike_client(fake)

    tool = QueryAerospikeTool()
    result = await tool.run(
        operation="transaction_state_get",
        params={"transaction_id": "TX123"},
    )

    assert result.tool == "query_aerospike"
    assert result.data["available"] is True
    assert result.data["operation"] == "transaction_state_get"
    assert result.data["found"] is True
    assert result.data["set"] == "tx_state"
    assert result.data["key"] == "TX123"
    assert result.data["namespace"] == "insurance"
    assert result.data["record"]["bins"]["state"] == "PENDING"

    # Summary should surface the decision-relevant bins:
    assert "TX123" in result.summary
    assert "PENDING" in result.summary
    assert "attempts=3" in result.summary
    assert "UPSTREAM_TIMEOUT" in result.summary

    # And exactly one lookup was issued with the right set + key:
    assert fake.calls == [("tx_state", "TX123", None)]


async def test_query_aerospike_returns_found_false_for_missing_key() -> None:
    fake = FakeAerospikeClient(store={})
    set_aerospike_client(fake)

    tool = QueryAerospikeTool()
    result = await tool.run(
        operation="policy_cache_get", params={"policy_id": "POL-ghost"}
    )

    assert result.data["available"] is True
    assert result.data["found"] is False
    assert result.data["record"] is None
    assert "not found" in result.summary.lower()


# ---------------------------------------------------------------------------
# Tool.run() - unavailable branches
# ---------------------------------------------------------------------------


async def test_query_aerospike_returns_unavailable_when_not_configured() -> None:
    set_aerospike_client(UnconfiguredFakeAerospikeClient())

    tool = QueryAerospikeTool()
    result = await tool.run(
        operation="policy_cache_get", params={"policy_id": "POL-1"}
    )

    assert result.data["available"] is False
    assert result.data["reason"] == "not_connected"
    assert result.data["record"] is None
    assert "not configured" in result.summary.lower()


async def test_query_aerospike_returns_unavailable_when_disconnected() -> None:
    set_aerospike_client(DisconnectedFakeAerospikeClient())

    tool = QueryAerospikeTool()
    result = await tool.run(
        operation="policy_cache_get", params={"policy_id": "POL-1"}
    )

    assert result.data["available"] is False
    assert result.data["reason"] == "not_connected"
    assert "not connected" in result.summary.lower()


# ---------------------------------------------------------------------------
# Tool.run() - input validation
# ---------------------------------------------------------------------------


async def test_query_aerospike_rejects_unknown_operation() -> None:
    set_aerospike_client(FakeAerospikeClient(store={}))

    tool = QueryAerospikeTool()
    result = await tool.run(operation="drop_the_cluster", params={})

    assert result.data["available"] is False
    assert result.data["reason"] == "invalid_input"
    assert "drop_the_cluster" in result.data["error"]


async def test_query_aerospike_rejects_missing_operation() -> None:
    set_aerospike_client(FakeAerospikeClient(store={}))

    tool = QueryAerospikeTool()
    result = await tool.run(params={"policy_id": "P"})

    assert result.data["available"] is False
    assert result.data["reason"] == "invalid_input"


async def test_query_aerospike_rejects_missing_key_param() -> None:
    set_aerospike_client(FakeAerospikeClient(store={}))

    tool = QueryAerospikeTool()
    # policy_cache_get requires policy_id but caller sent transaction_id.
    result = await tool.run(
        operation="policy_cache_get", params={"transaction_id": "TX9"}
    )

    assert result.data["available"] is False
    assert result.data["reason"] == "invalid_input"
    assert "policy_id" in result.data["error"]


# ---------------------------------------------------------------------------
# Tool.run() - downstream failure
# ---------------------------------------------------------------------------


async def test_query_aerospike_wraps_get_error_as_lookup_error() -> None:
    fake = FakeAerospikeClient(
        raise_on_get=RuntimeError("cluster unreachable")
    )
    set_aerospike_client(fake)

    tool = QueryAerospikeTool()
    result = await tool.run(
        operation="policy_cache_get", params={"policy_id": "POL-1"}
    )

    assert result.data["available"] is True
    assert result.data["reason"] == "lookup_error"
    assert "cluster unreachable" in result.data["error"]
    assert "failed" in result.summary.lower()


async def test_query_aerospike_wraps_aerospike_unavailable_from_get() -> None:
    fake = FakeAerospikeClient(
        raise_on_get=AerospikeUnavailable("cluster went away")
    )
    set_aerospike_client(fake)

    tool = QueryAerospikeTool()
    result = await tool.run(
        operation="transaction_state_get",
        params={"transaction_id": "TX-x"},
    )

    # AerospikeUnavailable takes the not_connected path:
    assert result.data["available"] is False
    assert result.data["reason"] == "not_connected"


# ---------------------------------------------------------------------------
# End-to-end: alert.labels.transactionId schedules an Aerospike lookup
# FIRST (fastest signal), then Postgres, then telemetry.
# ---------------------------------------------------------------------------


async def test_alert_with_transaction_id_triggers_aerospike_then_db() -> None:
    reset_store()
    reset_registry()
    reset_db_client()

    # DB fake - the plan will call query_database(transaction_journey)
    # right after the cache lookup.
    db_fake = FakeDatabaseClient(
        rows=[
            {
                "transaction_id": "TX123",
                "transaction_state": "COMPLETED",
                "workflow_id": "WF-42",
                "amount": 1200.00,
                "transaction_created_at": None,
                "policy_id": "POL-ICICI-9",
                "policy_status": "FAILED",
                "provider": "icici",
                "provider_error": "UPSTREAM_TIMEOUT",
                "policy_updated_at": None,
            }
        ]
    )
    set_db_client(db_fake)

    # Aerospike fake - the tx_state cache says the transaction has been
    # retried 3 times and is still in flight.
    aero_fake = FakeAerospikeClient(
        store={
            ("tx_state", "TX123"): {
                "bins": {
                    "state": "PENDING",
                    "attempts": 3,
                    "last_error": "UPSTREAM_TIMEOUT",
                },
                "meta": {"ttl": 300, "gen": 4},
            }
        }
    )
    set_aerospike_client(aero_fake)

    incoming = IncomingAlert(
        title="Policy issuance failed for TX123",
        service="payments-service",
        severity=AlertSeverity.HIGH,
        labels={"transactionId": "TX123", "product": "travel_insurance"},
    )
    alert = Alert.from_incoming(incoming)

    llm = MockLLM()
    registry = get_registry()
    assert "query_aerospike" in registry.names()
    assert "query_database" in registry.names()

    service = InvestigationService(llm=llm, registry=registry)
    context = service.normalizer.build(alert)
    investigation = Investigation(alert=alert, context=context)

    # Budget: aerospike + db + 6 telemetry tools + FINALIZE = 9 iters.
    # Default agent_max_iterations is 8, so raise the ceiling here.
    agent = RCAAgent(
        llm=llm, registry=registry, max_iterations=12, timeout_seconds=30
    )
    result = await agent.investigate(investigation)

    assert result.status.value == "completed", (
        f"investigation did not complete: error={result.error}"
    )

    action_steps = [s for s in result.steps if s.type == StepType.ACTION]
    tool_names = [s.tool for s in action_steps]

    # Aerospike FIRST (fastest signal), Postgres SECOND (authoritative),
    # then generic telemetry.
    assert tool_names[0] == "query_aerospike", (
        f"expected aerospike first, got {tool_names}"
    )
    assert "query_database" in tool_names
    aero_index = tool_names.index("query_aerospike")
    db_index = tool_names.index("query_database")
    assert aero_index < db_index

    # Both fakes were hit with the right key.
    assert aero_fake.calls == [("tx_state", "TX123", None)]
    assert len(db_fake.calls) == 1

    # Context now carries an AerospikeRecord tied to the alert's tx id:
    assert len(result.context.aerospike_records) == 1
    aero_rec = result.context.aerospike_records[0]
    assert aero_rec.operation == "transaction_state_get"
    assert aero_rec.params.get("transaction_id") == "TX123"
    assert aero_rec.found is True
    assert aero_rec.record["bins"]["state"] == "PENDING"

    # Report is NOT insufficient-evidence:
    assert result.report is not None
    assert "Insufficient evidence" not in result.report.headline

    # Aerospike evidence is quoted in the primary hypothesis so the
    # on-call can trace the retry signal back to the cache.
    supporting = " ".join(
        result.report.primary_hypothesis.supporting_evidence
    )
    assert "Aerospike" in supporting
    assert "TX123" in supporting
    assert "attempts=3" in supporting


async def test_aerospike_record_alone_satisfies_guardrail() -> None:
    """Even without any log/metric telemetry for the alerting service,
    a matching Aerospike record for the alert's identifier is enough
    for the guardrail to allow a real (non-insufficient-evidence) RCA.
    """
    reset_store()
    reset_registry()
    reset_db_client()

    # DB is unconfigured for this test - only Aerospike has ground truth.
    from app.services.database import DatabaseClient

    class NoDb(DatabaseClient):
        def __init__(self) -> None:
            pass

        @property
        def is_configured(self) -> bool:  # type: ignore[override]
            return False

        @property
        def is_connected(self) -> bool:  # type: ignore[override]
            return False

    set_db_client(NoDb())

    aero_fake = FakeAerospikeClient(
        store={
            ("tx_state", "TX-ONLY-CACHE"): {
                "bins": {
                    "state": "STUCK",
                    "attempts": 7,
                    "last_error": "PROVIDER_ERROR",
                },
                "meta": {"ttl": 60, "gen": 8},
            }
        }
    )
    set_aerospike_client(aero_fake)

    # Point the alert at a service whose seeded telemetry is EMPTY so
    # the only guardrail-satisfying evidence is the Aerospike record.
    incoming = IncomingAlert(
        title="Policy stuck",
        service="ghost-service",
        severity=AlertSeverity.HIGH,
        labels={"transactionId": "TX-ONLY-CACHE"},
    )
    alert = Alert.from_incoming(incoming)

    llm = MockLLM()
    registry = get_registry()
    service = InvestigationService(llm=llm, registry=registry)
    context = service.normalizer.build(alert)
    investigation = Investigation(alert=alert, context=context)

    agent = RCAAgent(
        llm=llm, registry=registry, max_iterations=12, timeout_seconds=30
    )
    result = await agent.investigate(investigation)

    assert result.status.value == "completed"
    assert result.report is not None
    # Guardrail must NOT trigger, because we do have a matching cache row:
    assert "Insufficient evidence" not in result.report.headline
    assert result.report.confidence > 0.15
    supporting = " ".join(
        result.report.primary_hypothesis.supporting_evidence
    )
    assert "Aerospike" in supporting
    assert "TX-ONLY-CACHE" in supporting
