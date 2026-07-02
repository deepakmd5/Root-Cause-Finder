"""Tests for the ``query_database`` tool and the ``DatabaseClient``.

We never touch a real PostgreSQL instance here. Instead we substitute
a :class:`FakeDatabaseClient` for the process-wide singleton, so the
tool exercises its full happy-path (bind params, execute, summarize,
serialize) and every failure branch (unconfigured, not connected,
invalid input, missing param, downstream error).

A single end-to-end test drives a full alert through the agent to
prove that the DB step gets scheduled first when the alert carries a
``transactionId`` label, and that the resulting DB record satisfies
the hallucination guardrail.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import pytest

from app.agents.rca_agent import RCAAgent
from app.llm.mock import MockLLM
from app.models.alert import Alert, AlertSeverity, IncomingAlert
from app.models.investigation import Investigation, StepType
from app.services.data_store import reset_store
from app.services.database import (
    DatabaseClient,
    DatabaseUnavailable,
    reset_db_client,
    set_db_client,
)
from app.services.investigation_service import InvestigationService
from app.tools.database_tool import (
    ALLOWED_QUERIES,
    QueryDatabaseTool,
    QueryTemplate,
    QueryValidationError,
    _jsonable_rows,
)
from app.tools.registry import get_registry, reset_registry


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeDatabaseClient(DatabaseClient):
    """A :class:`DatabaseClient` that never touches asyncpg.

    Records the ``(sql, args)`` of every ``fetch()`` call so tests can
    assert on the exact positional bind values the tool computed.
    """

    def __init__(
        self,
        rows: list[dict[str, Any]] | None = None,
        *,
        raise_on_fetch: Exception | None = None,
    ) -> None:
        # Skip the parent __init__ - we don't need any asyncpg attributes.
        self._rows = rows if rows is not None else []
        self._raise = raise_on_fetch
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

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

    async def fetch(
        self,
        sql: str,
        *args: Any,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append((sql, args))
        if self._raise is not None:
            raise self._raise
        return list(self._rows)


class UnconfiguredFakeClient(DatabaseClient):
    """DATABASE_URL not set - the tool must short-circuit."""

    def __init__(self) -> None:
        pass

    @property
    def is_configured(self) -> bool:  # type: ignore[override]
        return False

    @property
    def is_connected(self) -> bool:  # type: ignore[override]
        return False

    async def fetch(self, sql, *args, timeout=None):  # noqa: ANN001, ANN201
        raise AssertionError("fetch() must not be called when unconfigured")


class ConfiguredButDisconnectedFakeClient(DatabaseClient):
    """DATABASE_URL is set but the pool never came up (Postgres down)."""

    def __init__(self) -> None:
        pass

    @property
    def is_configured(self) -> bool:  # type: ignore[override]
        return True

    @property
    def is_connected(self) -> bool:  # type: ignore[override]
        return False

    async def fetch(self, sql, *args, timeout=None):  # noqa: ANN001, ANN201
        raise AssertionError("fetch() must not be called when disconnected")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolate_db_singleton():
    """Every test starts with a clean process-wide DB client slot."""
    reset_db_client()
    yield
    reset_db_client()


# ---------------------------------------------------------------------------
# QueryTemplate + validation
# ---------------------------------------------------------------------------


def test_query_template_bind_returns_positional_args_in_order() -> None:
    tpl = QueryTemplate(
        description="test", sql="SELECT $1, $2", params=["a", "b"]
    )
    assert tpl.bind({"a": 1, "b": 2}) == [1, 2]
    # Order follows ``params``, not the caller's dict:
    assert tpl.bind({"b": 20, "a": 10}) == [10, 20]


def test_query_template_bind_raises_on_missing_param() -> None:
    tpl = QueryTemplate(
        description="t", sql="SELECT $1, $2", params=["a", "b"]
    )
    with pytest.raises(QueryValidationError) as exc:
        tpl.bind({"a": 1})  # b is missing
    assert "b" in str(exc.value)


# ---------------------------------------------------------------------------
# Row serialization
# ---------------------------------------------------------------------------


def test_jsonable_rows_serializes_datetime_uuid_and_decimal() -> None:
    ts = datetime(2026, 1, 1, 12, 30, tzinfo=timezone.utc)
    uid = uuid.uuid4()
    rows = [
        {
            "policy_id": "POL-1",
            "created_at": ts,
            "trace_id": uid,
            "amount": Decimal("123.45"),
            "provider": None,
            "count": 5,
        }
    ]
    clean = _jsonable_rows(rows)
    assert len(clean) == 1
    row = clean[0]
    assert row["policy_id"] == "POL-1"
    assert row["created_at"] == ts.isoformat()
    assert row["trace_id"] == str(uid)
    # Decimal is JSON-encodable via default=str; the round-trip check
    # accepts it as-is or stringifies it - both are acceptable so we
    # just assert we haven't lost the value:
    assert str(row["amount"]) == "123.45"
    assert row["provider"] is None
    assert row["count"] == 5


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------


def test_query_database_advertises_all_allowlisted_queries_in_schema() -> None:
    tool = QueryDatabaseTool()
    spec = tool.spec()

    assert spec["name"] == "query_database"
    assert spec["description"]

    schema = spec["input_schema"]
    assert schema["type"] == "object"
    assert schema["required"] == ["query_name"]

    query_name_prop = schema["properties"]["query_name"]
    assert set(query_name_prop["enum"]) == set(ALLOWED_QUERIES.keys())

    # Sanity: at least the four queries we ship exist.
    for name in [
        "policy_status",
        "transaction_journey",
        "recent_failed_policies_by_provider",
        "policy_failure_count_by_provider",
    ]:
        assert name in query_name_prop["enum"]


# ---------------------------------------------------------------------------
# Tool.run() - happy path
# ---------------------------------------------------------------------------


async def test_query_database_returns_rows_when_configured_and_connected() -> None:
    fake_rows = [
        {
            "transaction_id": "TX123",
            "transaction_state": "COMPLETED",
            "workflow_id": "WF-9",
            "amount": Decimal("500.00"),
            "transaction_created_at": datetime(
                2026, 1, 1, 10, 0, tzinfo=timezone.utc
            ),
            "policy_id": "POL-1",
            "policy_status": "FAILED",
            "provider": "icici",
            "provider_error": "TIMEOUT_UPSTREAM",
            "policy_updated_at": datetime(
                2026, 1, 1, 10, 1, tzinfo=timezone.utc
            ),
        }
    ]
    fake = FakeDatabaseClient(rows=fake_rows)
    set_db_client(fake)

    tool = QueryDatabaseTool()
    result = await tool.run(
        query_name="transaction_journey",
        params={"transaction_id": "TX123"},
    )

    assert result.tool == "query_database"
    assert result.data["available"] is True
    assert result.data["query_name"] == "transaction_journey"
    assert result.data["row_count"] == 1
    assert result.data["rows"][0]["transaction_id"] == "TX123"
    # datetime + Decimal survived the JSON-safe rewrite:
    assert result.data["rows"][0]["transaction_created_at"] == (
        "2026-01-01T10:00:00+00:00"
    )

    # Summary should surface the most decision-relevant fields:
    assert "TX123" in result.summary
    assert "COMPLETED" in result.summary
    assert "TIMEOUT_UPSTREAM" in result.summary

    # And the tool must have passed exactly one positional arg (TX123):
    assert len(fake.calls) == 1
    sent_sql, sent_args = fake.calls[0]
    assert "FROM transactions" in sent_sql
    assert sent_args == ("TX123",)


async def test_query_database_summary_reflects_zero_row_case() -> None:
    set_db_client(FakeDatabaseClient(rows=[]))

    tool = QueryDatabaseTool()
    result = await tool.run(
        query_name="policy_status", params={"policy_id": "POL-ghost"}
    )

    assert result.data["available"] is True
    assert result.data["row_count"] == 0
    assert result.data["rows"] == []
    assert "0 rows" in result.summary
    assert "policy_status" in result.summary


# ---------------------------------------------------------------------------
# Tool.run() - unavailable branches
# ---------------------------------------------------------------------------


async def test_query_database_returns_unavailable_when_not_configured() -> None:
    set_db_client(UnconfiguredFakeClient())

    tool = QueryDatabaseTool()
    result = await tool.run(
        query_name="policy_status", params={"policy_id": "POL-1"}
    )

    assert result.data["available"] is False
    assert result.data["reason"] == "not_connected"
    assert result.data["rows"] == []
    assert result.data["row_count"] == 0
    assert "not configured" in result.summary.lower()


async def test_query_database_returns_unavailable_when_disconnected() -> None:
    set_db_client(ConfiguredButDisconnectedFakeClient())

    tool = QueryDatabaseTool()
    result = await tool.run(
        query_name="policy_status", params={"policy_id": "POL-1"}
    )

    assert result.data["available"] is False
    assert result.data["reason"] == "not_connected"
    assert "not connected" in result.summary.lower()


# ---------------------------------------------------------------------------
# Tool.run() - input validation
# ---------------------------------------------------------------------------


async def test_query_database_rejects_unknown_query_name() -> None:
    # Even if the DB is up, the allowlist gate must block.
    set_db_client(FakeDatabaseClient(rows=[{"policy_id": "x"}]))

    tool = QueryDatabaseTool()
    result = await tool.run(query_name="drop_all_tables", params={})

    assert result.data["available"] is False
    assert result.data["reason"] == "invalid_input"
    assert "drop_all_tables" in result.data["error"]
    assert result.data["rows"] == []


async def test_query_database_rejects_missing_query_name() -> None:
    set_db_client(FakeDatabaseClient(rows=[]))

    tool = QueryDatabaseTool()
    # LLM forgot query_name entirely:
    result = await tool.run(params={"policy_id": "P"})

    assert result.data["available"] is False
    assert result.data["reason"] == "invalid_input"


async def test_query_database_rejects_missing_required_param() -> None:
    set_db_client(FakeDatabaseClient(rows=[]))

    tool = QueryDatabaseTool()
    result = await tool.run(
        query_name="recent_failed_policies_by_provider",
        params={"provider": "icici"},  # missing since_minutes
    )

    assert result.data["available"] is False
    assert result.data["reason"] == "invalid_input"
    assert "since_minutes" in result.data["error"]


# ---------------------------------------------------------------------------
# Tool.run() - downstream failure
# ---------------------------------------------------------------------------


async def test_query_database_wraps_fetch_error_as_query_error() -> None:
    fake = FakeDatabaseClient(
        raise_on_fetch=RuntimeError("connection reset by peer")
    )
    set_db_client(fake)

    tool = QueryDatabaseTool()
    result = await tool.run(
        query_name="policy_status", params={"policy_id": "POL-1"}
    )

    assert result.data["available"] is True  # DB *was* reachable, query failed
    assert result.data["reason"] == "query_error"
    assert "connection reset by peer" in result.data["error"]
    assert result.data["rows"] == []
    assert "failed" in result.summary.lower()


async def test_query_database_wraps_database_unavailable_from_fetch() -> None:
    fake = FakeDatabaseClient(
        raise_on_fetch=DatabaseUnavailable("pool went away")
    )
    set_db_client(fake)

    tool = QueryDatabaseTool()
    result = await tool.run(
        query_name="policy_status", params={"policy_id": "POL-1"}
    )

    # DatabaseUnavailable takes the "not_connected" path in the tool:
    assert result.data["available"] is False
    assert result.data["reason"] == "not_connected"


# ---------------------------------------------------------------------------
# End-to-end: alert.labels.transactionId schedules a DB lookup FIRST
# and the returned row satisfies the hallucination guardrail.
# ---------------------------------------------------------------------------


async def test_alert_with_transaction_id_triggers_db_lookup_and_evidence() -> None:
    reset_store()
    reset_registry()

    # Install a fake DB with a plausible ICICI/travel_insurance journey row.
    fake_row = {
        "transaction_id": "TX123",
        "transaction_state": "COMPLETED",
        "workflow_id": "WF-42",
        "amount": Decimal("1200.00"),
        "transaction_created_at": datetime(
            2026, 1, 1, 10, 0, tzinfo=timezone.utc
        ),
        "policy_id": "POL-ICICI-9",
        "policy_status": "FAILED",
        "provider": "icici",
        "provider_error": "UPSTREAM_TIMEOUT",
        "policy_updated_at": datetime(
            2026, 1, 1, 10, 1, tzinfo=timezone.utc
        ),
    }
    fake = FakeDatabaseClient(rows=[fake_row])
    set_db_client(fake)

    # Feed a payments-service alert (existing seeded telemetry works)
    # WITH a transactionId label so the mock plan puts DB lookup first.
    incoming = IncomingAlert(
        title="Policy issuance failed for TX123",
        service="payments-service",
        severity=AlertSeverity.HIGH,
        labels={"transactionId": "TX123", "product": "travel_insurance"},
    )
    alert = Alert.from_incoming(incoming)

    llm = MockLLM()
    registry = get_registry()
    assert "query_database" in registry.names()

    service = InvestigationService(llm=llm, registry=registry)
    context = service.normalizer.build(alert)
    investigation = Investigation(alert=alert, context=context)

    # Bump the iteration budget: DB + 6 telemetry tools + FINALIZE = 8+,
    # and the default is 8. Give it headroom.
    agent = RCAAgent(
        llm=llm, registry=registry, max_iterations=12, timeout_seconds=30
    )
    result = await agent.investigate(investigation)

    assert result.status.value == "completed", (
        f"investigation did not complete: error={result.error}"
    )

    # The DB tool was actually invoked. With Aerospike now wired in as
    # a peer tool, the mock plan puts the (fast) cache lookup FIRST and
    # the (authoritative) DB query second - so query_database is not
    # necessarily action_steps[0], but it MUST appear before any of the
    # generic telemetry tools (deployments, logs, metrics, traces).
    action_steps = [s for s in result.steps if s.type == StepType.ACTION]
    assert action_steps, "no actions were recorded"
    tool_names = [s.tool for s in action_steps]
    assert "query_database" in tool_names, (
        f"expected query_database in the plan, got {tool_names}"
    )
    db_index = tool_names.index("query_database")
    for later_tool in ("recent_deployments", "search_logs", "query_metrics"):
        if later_tool in tool_names:
            assert db_index < tool_names.index(later_tool), (
                f"expected query_database before {later_tool}, "
                f"got {tool_names}"
            )

    # The fake DB was hit exactly once with TX123 as the bind arg.
    assert len(fake.calls) == 1
    _, bind_args = fake.calls[0]
    assert bind_args == ("TX123",)

    # The context now carries a DbRecord tied to the alert's tx_id -
    # this is what the guardrail keys off.
    assert len(result.context.db_records) == 1
    rec = result.context.db_records[0]
    assert rec.query_name == "transaction_journey"
    assert rec.params.get("transaction_id") == "TX123"
    assert rec.row_count == 1
    assert rec.available is True

    # And crucially: the report is NOT the insufficient-evidence one,
    # because we have both telemetry AND a matching DB row.
    assert result.report is not None
    assert "Insufficient evidence" not in result.report.headline
    assert result.report.confidence > 0.3

    # DB evidence should be quoted in the primary hypothesis so the
    # on-call can trace it back to the source of truth.
    supporting = " ".join(result.report.primary_hypothesis.supporting_evidence)
    assert "TX123" in supporting


async def test_alert_without_business_ids_skips_db_step() -> None:
    """No transactionId/policyId/provider -> mock does not call the DB."""
    reset_store()
    reset_registry()

    fake = FakeDatabaseClient(rows=[])
    set_db_client(fake)

    incoming = IncomingAlert(
        title="High error rate on payments-service",
        service="payments-service",
        severity=AlertSeverity.HIGH,
    )
    alert = Alert.from_incoming(incoming)

    llm = MockLLM()
    registry = get_registry()
    service = InvestigationService(llm=llm, registry=registry)
    context = service.normalizer.build(alert)
    investigation = Investigation(alert=alert, context=context)

    agent = RCAAgent(
        llm=llm, registry=registry, max_iterations=10, timeout_seconds=30
    )
    result = await agent.investigate(investigation)

    assert result.status.value == "completed"
    # Fake DB was never called - no business identifiers to key off.
    assert fake.calls == []
    tool_names = {
        s.tool for s in result.steps if s.type == StepType.ACTION
    }
    assert "query_database" not in tool_names
