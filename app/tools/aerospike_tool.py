"""Aerospike (NoSQL) query tool for the RCA agent.

Safety model - matches :mod:`app.tools.database_tool`
-----------------------------------------------------
The LLM never constructs raw Aerospike keys. It selects an
``operation`` from an allowlist and supplies one named parameter that
becomes the record key. The set + namespace mapping lives in code
(:data:`ALLOWED_OPERATIONS`), so the LLM cannot:

* Read from arbitrary sets (data exfiltration).
* Trigger destructive operations (this tool is read-only end-to-end).
* Bypass the async timeout / total-timeout budgets.

Ships with three insurance-flavoured hot-cache lookups. Extend by
adding entries to :data:`ALLOWED_OPERATIONS`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from app.core.exceptions import RCAError
from app.core.logging import get_logger
from app.services.aerospike_client import (
    AerospikeUnavailable,
    get_aerospike_client,
)
from app.tools.base import Tool, ToolResult

log = get_logger(__name__)


class AerospikeOperation(BaseModel):
    """A single allowlisted, parameterized Aerospike lookup."""

    description: str
    aerospike_set: str
    key_param: str  # which named param becomes the record key
    # ``None`` means "use the client's default namespace" (from config).
    namespace: str | None = None


class OperationValidationError(RCAError):
    """Raised when the LLM's tool_input is malformed."""


# ---------------------------------------------------------------------------
# The allowlist. Every entry translates a named operation into
# (namespace, set, key(param)) that the async client can execute.
# ---------------------------------------------------------------------------

ALLOWED_OPERATIONS: dict[str, AerospikeOperation] = {
    "policy_cache_get": AerospikeOperation(
        description=(
            "Read the hot cache entry for a policy. Returns the last "
            "cached status/provider snapshot, useful when the alert "
            "references a policy_id and you want to know what the "
            "downstream consumers saw most recently."
        ),
        aerospike_set="policy_cache",
        key_param="policy_id",
    ),
    "transaction_state_get": AerospikeOperation(
        description=(
            "Read the in-flight transaction state cache. Useful when "
            "the alert carries a transaction_id: tells you whether the "
            "transaction is still PENDING (workflow stuck), whether "
            "retries are ongoing, or when the last state transition "
            "occurred."
        ),
        aerospike_set="tx_state",
        key_param="transaction_id",
    ),
    "idempotency_get": AerospikeOperation(
        description=(
            "Read the idempotency record for a client-supplied request "
            "key. Confirms whether a duplicate submission or a retry "
            "storm hit the same request; the ``attempts`` bin quantifies "
            "how badly."
        ),
        aerospike_set="idempotency",
        key_param="idempotency_key",
    ),
}


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class QueryAerospikeTool(Tool):
    name = "query_aerospike"
    description = (
        "Read from the operational Aerospike (NoSQL) cluster using an "
        "allowlisted, read-only, single-key operation. The LLM does NOT "
        "construct Aerospike keys - it selects an `operation` from the "
        "allowlist and passes one named `params` entry that becomes the "
        "record key. Use this for hot-cache lookups (policy cache, "
        "transaction state, idempotency records) when the alert carries "
        "a business identifier."
    )
    # input_schema is regenerated on init so it reflects the current
    # allowlist automatically.

    def __init__(self) -> None:
        self.input_schema = self._build_schema()

    def _build_schema(self) -> dict[str, Any]:
        op_descriptions = "\n".join(
            f"  - {name}: {op.description.strip()} "
            f"key_param={op.key_param}"
            for name, op in ALLOWED_OPERATIONS.items()
        )
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": list(ALLOWED_OPERATIONS.keys()),
                    "description": (
                        "Name of the allowlisted Aerospike operation. "
                        "Available operations:\n" + op_descriptions
                    ),
                },
                "params": {
                    "type": "object",
                    "description": (
                        "Named parameters. Each operation requires "
                        "exactly one key-defining param whose name is "
                        "shown in the `operation` enum description."
                    ),
                    "additionalProperties": True,
                },
            },
            "required": ["operation"],
        }

    async def run(
        self,
        operation: str | None = None,
        params: dict[str, Any] | None = None,
        **_: Any,
    ) -> ToolResult:
        params = params or {}
        try:
            op = self._resolve(operation)
            key = self._extract_key(op, params)
        except OperationValidationError as exc:
            return ToolResult(
                tool=self.name,
                input={"operation": operation, "params": params},
                summary=f"Invalid tool call: {exc}",
                data={
                    "available": False,
                    "reason": "invalid_input",
                    "error": str(exc),
                    "operation": operation,
                    "params": params,
                    "found": False,
                    "record": None,
                },
            )

        client = get_aerospike_client()
        if not client.is_configured:
            summary = (
                f"Aerospike not configured: cannot run `{operation}`. "
                "Set AEROSPIKE_HOSTS + AEROSPIKE_NAMESPACE in the "
                "environment."
            )
            return self._unavailable_result(operation, params, summary)

        if not client.is_connected:
            summary = (
                f"Aerospike configured but not connected: cannot run "
                f"`{operation}`. Check cluster availability."
            )
            return self._unavailable_result(operation, params, summary)

        try:
            record = await client.get(
                set_name=op.aerospike_set,
                key=key,
                namespace=op.namespace,
            )
        except AerospikeUnavailable as exc:
            return self._unavailable_result(operation, params, str(exc))
        except Exception as exc:  # noqa: BLE001
            log.exception("aerospike.lookup_failed", operation=operation)
            return ToolResult(
                tool=self.name,
                input={"operation": operation, "params": params},
                summary=(
                    f"Aerospike lookup `{operation}` failed: "
                    f"{exc.__class__.__name__}: {exc}"
                ),
                data={
                    "available": True,
                    "reason": "lookup_error",
                    "error": str(exc),
                    "operation": operation,
                    "params": params,
                    "found": False,
                    "record": None,
                },
            )

        found = record is not None
        summary = self._summarize(operation, key, record)
        return ToolResult(
            tool=self.name,
            input={"operation": operation, "params": params},
            summary=summary,
            data={
                "available": True,
                "operation": operation,
                "params": params,
                "namespace": op.namespace or client.namespace,
                "set": op.aerospike_set,
                "key": key,
                "found": found,
                "record": _jsonable_record(record) if found else None,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    # -- Helpers ---------------------------------------------------------

    def _resolve(self, operation: str | None) -> AerospikeOperation:
        if not operation:
            raise OperationValidationError(
                "operation is required. Choose one of: "
                + ", ".join(ALLOWED_OPERATIONS)
            )
        if operation not in ALLOWED_OPERATIONS:
            raise OperationValidationError(
                f"Unknown operation '{operation}'. Allowed: "
                + ", ".join(ALLOWED_OPERATIONS)
            )
        return ALLOWED_OPERATIONS[operation]

    def _extract_key(
        self, op: AerospikeOperation, params: dict[str, Any]
    ) -> str:
        raw = params.get(op.key_param)
        if raw is None or raw == "":
            raise OperationValidationError(
                f"Missing required param `{op.key_param}` for operation "
                f"`{[k for k, v in ALLOWED_OPERATIONS.items() if v is op][0]}`."
            )
        return str(raw)

    def _unavailable_result(
        self,
        operation: str | None,
        params: dict[str, Any],
        reason: str,
    ) -> ToolResult:
        return ToolResult(
            tool=self.name,
            input={"operation": operation, "params": params},
            summary=reason,
            data={
                "available": False,
                "reason": "not_connected",
                "error": reason,
                "operation": operation,
                "params": params,
                "found": False,
                "record": None,
            },
        )

    def _summarize(
        self,
        operation: str,
        key: str,
        record: dict[str, Any] | None,
    ) -> str:
        if record is None:
            return f"Aerospike `{operation}` key=`{key}`: not found."

        bins = record.get("bins") or {}
        meta = record.get("meta") or {}

        # Operation-specific one-liners give the LLM a stronger next-step
        # signal than a raw bin dump.
        if operation == "policy_cache_get":
            return (
                f"Aerospike.policy_cache[{key}] "
                f"status={bins.get('status')} "
                f"provider={bins.get('provider')} "
                f"provider_error={bins.get('provider_error') or 'none'} "
                f"ttl={meta.get('ttl')}s."
            )
        if operation == "transaction_state_get":
            return (
                f"Aerospike.tx_state[{key}] "
                f"state={bins.get('state')} "
                f"attempts={bins.get('attempts')} "
                f"last_error={bins.get('last_error') or 'none'} "
                f"ttl={meta.get('ttl')}s."
            )
        if operation == "idempotency_get":
            return (
                f"Aerospike.idempotency[{key}] "
                f"attempts={bins.get('attempts')} "
                f"in_flight={bins.get('in_flight')} "
                f"outcome={bins.get('outcome') or 'pending'}."
            )
        return f"Aerospike `{operation}` key=`{key}` found."


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _jsonable_record(record: dict[str, Any]) -> dict[str, Any]:
    """Coerce a raw Aerospike record into JSON-safe primitives.

    Aerospike bin values can be bytes, sets, or nested maps/lists. We
    project everything down to str/int/float/bool/None/list/dict so the
    :class:`ToolResult` can be persisted and re-serialized without
    surprises.
    """
    return {
        "bins": _jsonable(record.get("bins") or {}),
        "meta": _jsonable(record.get("meta") or {}),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
