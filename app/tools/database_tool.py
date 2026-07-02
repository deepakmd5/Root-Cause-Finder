"""PostgreSQL query tool for the RCA agent.

Safety model
------------
The LLM never writes SQL. It picks a ``query_name`` from an
**allowlist** and supplies named parameters. The SQL bodies live here,
in code, and are versioned + reviewed like any other production code.
This eliminates the classic LLM-to-DB failure modes:

* SQL injection (all params go through ``$1``/``$2`` bind vars, and the
  SQL body itself is never user-controlled).
* Data exfiltration (LLM cannot select from arbitrary tables).
* Destructive statements (allowlist only exposes SELECTs; the
  ``DatabaseClient`` additionally forces every session to
  ``READ ONLY``).

Ship with an insurance-flavoured schema. Extend the allowlist by adding
entries to :data:`ALLOWED_QUERIES`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from app.core.exceptions import RCAError
from app.core.logging import get_logger
from app.services.database import DatabaseUnavailable, get_db_client
from app.tools.base import Tool, ToolResult

log = get_logger(__name__)


class QueryTemplate(BaseModel):
    """A single allowlisted, parameterized query."""

    description: str
    sql: str
    # Ordered list of parameter names. Positional binds ``$1..$N`` in
    # ``sql`` correspond to this list, in order.
    params: list[str]

    def bind(self, provided: dict[str, Any]) -> list[Any]:
        """Return positional args for the SQL from a name->value dict."""
        missing = [p for p in self.params if p not in provided]
        if missing:
            raise QueryValidationError(
                f"Missing required params: {missing}. "
                f"Expected: {self.params}"
            )
        return [provided[p] for p in self.params]


class QueryValidationError(RCAError):
    """Raised when the LLM's tool_input is malformed."""


# ---------------------------------------------------------------------------
# The allowlist. Add new entries here and they become instantly available
# to every LLM adapter.
# ---------------------------------------------------------------------------

ALLOWED_QUERIES: dict[str, QueryTemplate] = {
    "policy_status": QueryTemplate(
        description=(
            "Fetch the current state of a single policy by its "
            "policy_id. Returns status, provider, provider_error, and "
            "timestamps. Use when the alert carries a policy identifier."
        ),
        sql="""
            SELECT policy_id,
                   transaction_id,
                   status,
                   provider,
                   provider_error,
                   product,
                   created_at,
                   updated_at
            FROM policies
            WHERE policy_id = $1
            LIMIT 1
        """,
        params=["policy_id"],
    ),
    "transaction_journey": QueryTemplate(
        description=(
            "Return a transaction and the associated policy (if any). "
            "Useful when the alert only carries a transaction_id and you "
            "need to correlate payment state with policy issuance."
        ),
        sql="""
            SELECT t.transaction_id,
                   t.state          AS transaction_state,
                   t.workflow_id,
                   t.amount,
                   t.created_at     AS transaction_created_at,
                   p.policy_id,
                   p.status         AS policy_status,
                   p.provider,
                   p.provider_error,
                   p.updated_at     AS policy_updated_at
            FROM transactions t
            LEFT JOIN policies p
              ON p.transaction_id = t.transaction_id
            WHERE t.transaction_id = $1
            LIMIT 5
        """,
        params=["transaction_id"],
    ),
    "recent_failed_policies_by_provider": QueryTemplate(
        description=(
            "List recently failed policies for a given provider within "
            "the last N minutes. Useful for confirming that an incident "
            "is provider-wide rather than a single stuck transaction."
        ),
        sql="""
            SELECT policy_id,
                   transaction_id,
                   status,
                   provider_error,
                   created_at,
                   updated_at
            FROM policies
            WHERE provider = $1
              AND status = 'FAILED'
              AND created_at >= NOW() - make_interval(mins => $2::int)
            ORDER BY created_at DESC
            LIMIT 50
        """,
        params=["provider", "since_minutes"],
    ),
    "policy_failure_count_by_provider": QueryTemplate(
        description=(
            "Count failed vs total policies per provider over the last "
            "N minutes. Highlights whether one provider is disproportion"
            "ately failing right now."
        ),
        sql="""
            SELECT provider,
                   COUNT(*)                                       AS total,
                   COUNT(*) FILTER (WHERE status = 'FAILED')      AS failed,
                   COUNT(*) FILTER (WHERE status = 'SUCCESS')     AS succeeded
            FROM policies
            WHERE created_at >= NOW() - make_interval(mins => $1::int)
            GROUP BY provider
            ORDER BY failed DESC, total DESC
        """,
        params=["since_minutes"],
    ),
}


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class QueryDatabaseTool(Tool):
    name = "query_database"
    description = (
        "Run an allowlisted, read-only, parameterized query against the "
        "operational PostgreSQL database. The LLM does NOT write SQL - "
        "it selects a `query_name` from the allowlist and passes named "
        "`params`. Use this to fetch policy/transaction state, correlate "
        "an alert's transactionId or policyId with backend records, or "
        "quantify provider-wide failure rates."
    )
    # The input_schema is regenerated on init so it always reflects the
    # current allowlist.

    def __init__(self) -> None:
        self.input_schema = self._build_schema()

    def _build_schema(self) -> dict[str, Any]:
        query_descriptions = "\n".join(
            f"  - {name}: {tpl.description.strip()} params={tpl.params}"
            for name, tpl in ALLOWED_QUERIES.items()
        )
        return {
            "type": "object",
            "properties": {
                "query_name": {
                    "type": "string",
                    "enum": list(ALLOWED_QUERIES.keys()),
                    "description": (
                        "Name of the allowlisted query to run. "
                        "Available queries:\n" + query_descriptions
                    ),
                },
                "params": {
                    "type": "object",
                    "description": (
                        "Named parameters required by the chosen query. "
                        "Each query's required param names are listed in "
                        "the query_name enum's description above."
                    ),
                    "additionalProperties": True,
                },
            },
            "required": ["query_name"],
        }

    async def run(
        self,
        query_name: str | None = None,
        params: dict[str, Any] | None = None,
        **_: Any,
    ) -> ToolResult:
        params = params or {}
        try:
            template = self._resolve(query_name)
            bind_values = template.bind(params)
        except QueryValidationError as exc:
            return ToolResult(
                tool=self.name,
                input={"query_name": query_name, "params": params},
                summary=f"Invalid tool call: {exc}",
                data={
                    "available": False,
                    "reason": "invalid_input",
                    "error": str(exc),
                    "rows": [],
                    "row_count": 0,
                    "query_name": query_name,
                    "params": params,
                },
            )

        client = get_db_client()
        if not client.is_configured:
            summary = (
                f"Database not configured: cannot run `{query_name}`. "
                "Set DATABASE_URL in the environment."
            )
            return self._unavailable_result(query_name, params, summary)

        if not client.is_connected:
            summary = (
                f"Database configured but not connected: cannot run "
                f"`{query_name}`. Check DB availability."
            )
            return self._unavailable_result(query_name, params, summary)

        try:
            rows = await client.fetch(template.sql, *bind_values)
        except DatabaseUnavailable as exc:
            return self._unavailable_result(query_name, params, str(exc))
        except Exception as exc:  # noqa: BLE001
            log.exception("db.query_failed", query_name=query_name)
            return ToolResult(
                tool=self.name,
                input={"query_name": query_name, "params": params},
                summary=(
                    f"Database query `{query_name}` failed: "
                    f"{exc.__class__.__name__}: {exc}"
                ),
                data={
                    "available": True,
                    "reason": "query_error",
                    "error": str(exc),
                    "rows": [],
                    "row_count": 0,
                    "query_name": query_name,
                    "params": params,
                },
            )

        summary = self._summarize(query_name, params, rows)
        return ToolResult(
            tool=self.name,
            input={"query_name": query_name, "params": params},
            summary=summary,
            data={
                "available": True,
                "query_name": query_name,
                "params": params,
                "row_count": len(rows),
                "rows": _jsonable_rows(rows),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    # -- Helpers ---------------------------------------------------------

    def _resolve(self, query_name: str | None) -> QueryTemplate:
        if not query_name:
            raise QueryValidationError(
                "query_name is required. Choose one of: "
                + ", ".join(ALLOWED_QUERIES)
            )
        if query_name not in ALLOWED_QUERIES:
            raise QueryValidationError(
                f"Unknown query_name '{query_name}'. Allowed: "
                + ", ".join(ALLOWED_QUERIES)
            )
        return ALLOWED_QUERIES[query_name]

    def _unavailable_result(
        self,
        query_name: str | None,
        params: dict[str, Any],
        reason: str,
    ) -> ToolResult:
        return ToolResult(
            tool=self.name,
            input={"query_name": query_name, "params": params},
            summary=reason,
            data={
                "available": False,
                "reason": "not_connected",
                "error": reason,
                "rows": [],
                "row_count": 0,
                "query_name": query_name,
                "params": params,
            },
        )

    def _summarize(
        self,
        query_name: str,
        params: dict[str, Any],
        rows: list[dict[str, Any]],
    ) -> str:
        if not rows:
            return (
                f"Query `{query_name}` returned 0 rows for params={params}."
            )

        # Query-specific one-liners give the LLM a much stronger next-step
        # signal than a raw row count.
        if query_name == "policy_status":
            r = rows[0]
            return (
                f"Policy {r.get('policy_id')} status="
                f"{r.get('status')} provider={r.get('provider')} "
                f"provider_error={r.get('provider_error') or 'none'}."
            )
        if query_name == "transaction_journey":
            r = rows[0]
            return (
                f"Transaction {r.get('transaction_id')} state="
                f"{r.get('transaction_state')}; "
                f"policy_status={r.get('policy_status') or 'no-policy'}; "
                f"provider_error={r.get('provider_error') or 'none'}."
            )
        if query_name == "recent_failed_policies_by_provider":
            return (
                f"{len(rows)} failed policies for "
                f"provider={params.get('provider')} in the last "
                f"{params.get('since_minutes')}m."
            )
        if query_name == "policy_failure_count_by_provider":
            top = rows[0]
            return (
                f"{len(rows)} providers observed; worst offender: "
                f"{top.get('provider')} failed={top.get('failed')}"
                f"/{top.get('total')}."
            )
        return f"Query `{query_name}` returned {len(rows)} row(s)."


def _jsonable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Coerce non-JSON-native values (datetime, Decimal, UUID) to strings."""
    import json
    import uuid
    from decimal import Decimal

    out: list[dict[str, Any]] = []
    for row in rows:
        clean: dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, datetime):
                clean[k] = v.isoformat()
            elif isinstance(v, uuid.UUID):
                clean[k] = str(v)
            elif isinstance(v, Decimal):
                # Preserve exact value as a string; float() would risk
                # silent precision loss for currency-sized numbers.
                clean[k] = str(v)
            else:
                # int, float, str, bool, None all round-trip fine.
                try:
                    json.dumps(v)  # sanity check
                    clean[k] = v
                except (TypeError, ValueError):
                    clean[k] = str(v)
        out.append(clean)
    return out
