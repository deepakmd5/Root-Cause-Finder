"""Deterministic mock LLM.

Implements the same ``decide()`` contract as a real LLM, but uses simple
heuristics against the running context so the agent behaves sensibly
without any external calls. This makes the prototype runnable with zero
configuration and keeps demos + tests reproducible.
"""
from __future__ import annotations

import json
import re
from typing import Any

from app.llm.base import AgentDecision, DecisionType, LLMMessage


class MockLLM:
    name = "mock"

    async def decide(
        self,
        messages: list[LLMMessage],
        available_tools: list[dict[str, Any]],
    ) -> AgentDecision:
        state = _parse_state(messages)
        used_keys = state["used_keys"]
        alert = state["alert"]
        service = alert.get("service", "unknown-service")
        labels = {k.lower(): v for k, v in (alert.get("labels") or {}).items()}
        tx_id = labels.get("transactionid") or labels.get("transaction_id")
        policy_id = labels.get("policyid") or labels.get("policy_id")
        provider = labels.get("provider")
        idem_key = labels.get("idempotencykey") or labels.get("idempotency_key")

        # Investigation plan - each step gathers a piece of the puzzle.
        # ``dedupe_key`` is the semantic identity of the step; two steps
        # with the same dedupe_key are considered "the same call".
        plan: list[tuple[str, dict[str, Any], str, str]] = []

        # Aerospike (hot cache / in-flight state) is the fastest signal
        # we can get and it often shows retry storms or expired sessions
        # that Postgres would not surface. Do the cache read FIRST when
        # the alert carries a business identifier.
        if tx_id:
            plan.append(
                (
                    "query_aerospike",
                    {
                        "operation": "transaction_state_get",
                        "params": {"transaction_id": tx_id},
                    },
                    f"Alert carries transactionId={tx_id}. Check the "
                    "in-flight transaction cache first - it will show "
                    "attempt counts and the last error before I even "
                    "touch the DB.",
                    f"query_aerospike:transaction_state_get:{tx_id}",
                )
            )
        if policy_id:
            plan.append(
                (
                    "query_aerospike",
                    {
                        "operation": "policy_cache_get",
                        "params": {"policy_id": policy_id},
                    },
                    f"Alert carries policyId={policy_id}. Read the "
                    "policy cache to see the last snapshot downstream "
                    "consumers observed.",
                    f"query_aerospike:policy_cache_get:{policy_id}",
                )
            )
        if idem_key:
            plan.append(
                (
                    "query_aerospike",
                    {
                        "operation": "idempotency_get",
                        "params": {"idempotency_key": idem_key},
                    },
                    f"Alert carries idempotencyKey={idem_key}. Check "
                    "for a duplicate submission / retry storm.",
                    f"query_aerospike:idempotency_get:{idem_key}",
                )
            )

        # DB lookups are the authoritative next step. They confirm
        # whatever the cache said (or reveal cache staleness).
        if tx_id:
            plan.append(
                (
                    "query_database",
                    {
                        "query_name": "transaction_journey",
                        "params": {"transaction_id": tx_id},
                    },
                    f"Now correlate the cache view with the DB: fetch "
                    f"the full transaction journey for {tx_id}.",
                    f"query_database:transaction_journey:{tx_id}",
                )
            )
        if policy_id:
            plan.append(
                (
                    "query_database",
                    {
                        "query_name": "policy_status",
                        "params": {"policy_id": policy_id},
                    },
                    f"Fetch the authoritative policy state for "
                    f"{policy_id} from the DB.",
                    f"query_database:policy_status:{policy_id}",
                )
            )
        if provider:
            plan.append(
                (
                    "query_database",
                    {
                        "query_name": "recent_failed_policies_by_provider",
                        "params": {
                            "provider": provider,
                            "since_minutes": 30,
                        },
                    },
                    f"Confirm whether provider `{provider}` is failing "
                    "widely or only for this one transaction.",
                    f"query_database:recent_failed_policies:{provider}",
                )
            )

        plan.extend(
            [
                (
                    "recent_deployments",
                    {"service": service, "since_minutes": 240},
                    f"A production alert fired for `{service}`. Recent "
                    "deploys are a top suspect - checking them next.",
                    f"recent_deployments:{service}",
                ),
                (
                    "search_logs",
                    {"services": [service], "levels": ["ERROR", "WARN"]},
                    "Now I need concrete error signatures. Pulling "
                    f"ERROR/WARN logs for {service} to see the failure "
                    "mode.",
                    f"search_logs:{service}",
                ),
                (
                    "query_metrics",
                    {"service": service, "metric": "error_rate"},
                    "Correlating logs with the error-rate metric to "
                    "confirm impact and timing.",
                    f"query_metrics:{service}:error_rate",
                ),
                (
                    "query_metrics",
                    {"service": service, "metric": "latency_p95_ms"},
                    "Checking latency to see if this is a hard failure "
                    "or a slow degradation.",
                    f"query_metrics:{service}:latency_p95_ms",
                ),
                (
                    "fetch_traces",
                    {"service": service, "error_only": True},
                    "Traces will show which downstream hop is actually "
                    "failing.",
                    f"fetch_traces:{service}",
                ),
                (
                    "get_service_dependencies",
                    {"service": service},
                    "Mapping the dependency graph so I know the blast "
                    "radius.",
                    f"get_service_dependencies:{service}",
                ),
                (
                    "find_similar_incidents",
                    {
                        "service": service,
                        "keywords": ["pool", "timeout", "deploy"],
                    },
                    "Checking whether we have seen this signature "
                    "before.",
                    f"find_similar_incidents:{service}",
                ),
            ]
        )

        for tool_name, tool_input, thought, dedupe_key in plan:
            if dedupe_key not in used_keys:
                return AgentDecision(
                    type=DecisionType.USE_TOOL,
                    thought=thought,
                    tool=tool_name,
                    tool_input=tool_input,
                )

        # All evidence gathered - synthesize the RCA.
        return AgentDecision(
            type=DecisionType.FINALIZE,
            thought=(
                "I have deployment history, error logs, degraded "
                "metrics, error traces, dependency topology, matching "
                "past incidents, plus (when available) direct DB state "
                "and hot-cache readings from Aerospike. Time to "
                "synthesize the RCA."
            ),
            final_answer=_synthesize_rca(state),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _observation_key(tool: str, tool_input: dict[str, Any]) -> str:
    """Semantic identity for an observed tool call (matches plan keys)."""
    if tool == "query_metrics":
        return f"{tool}:{tool_input.get('service')}:{tool_input.get('metric')}"
    if tool == "search_logs":
        services = tool_input.get("services") or []
        primary = services[0] if services else ""
        return f"{tool}:{primary}"
    if tool == "query_database":
        qn = tool_input.get("query_name", "")
        params = tool_input.get("params") or {}
        if qn == "transaction_journey":
            return f"{tool}:{qn}:{params.get('transaction_id', '')}"
        if qn == "policy_status":
            return f"{tool}:{qn}:{params.get('policy_id', '')}"
        if qn == "recent_failed_policies_by_provider":
            return f"{tool}:recent_failed_policies:{params.get('provider', '')}"
        # Fallback: dedupe purely by query name
        return f"{tool}:{qn}"
    if tool == "query_aerospike":
        op = tool_input.get("operation", "")
        params = tool_input.get("params") or {}
        if op == "transaction_state_get":
            return f"{tool}:{op}:{params.get('transaction_id', '')}"
        if op == "policy_cache_get":
            return f"{tool}:{op}:{params.get('policy_id', '')}"
        if op == "idempotency_get":
            return f"{tool}:{op}:{params.get('idempotency_key', '')}"
        return f"{tool}:{op}"
    return f"{tool}:{tool_input.get('service', '')}"


def _parse_state(messages: list[LLMMessage]) -> dict[str, Any]:
    """Extract accumulated tool observations from the message history."""
    alert: dict[str, Any] = {}
    used_keys: set[str] = set()
    observations: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == "user" and msg.content.startswith("ALERT_JSON::"):
            try:
                alert = json.loads(msg.content.removeprefix("ALERT_JSON::"))
            except json.JSONDecodeError:
                alert = {}
        elif msg.role == "user" and msg.content.startswith("OBSERVATION::"):
            payload = msg.content.removeprefix("OBSERVATION::")
            try:
                obs = json.loads(payload)
            except json.JSONDecodeError:
                continue
            observations.append(obs)
            used_keys.add(
                _observation_key(obs.get("tool", ""), obs.get("input", {}))
            )

    return {
        "alert": alert,
        "used_keys": used_keys,
        "observations": observations,
    }


def _synthesize_rca(state: dict[str, Any]) -> dict[str, Any]:
    """Build the FINALIZE payload from collected observations.

    Uses simple pattern-matching on the observation stream. In a real
    deployment the LLM would replace this with genuine reasoning.
    """
    alert = state["alert"]
    observations = state["observations"]
    service = alert.get("service", "unknown-service")

    deploy_evidence: list[str] = []
    log_evidence: list[str] = []
    metric_evidence: list[str] = []
    trace_evidence: list[str] = []
    similar_evidence: list[str] = []
    db_evidence: list[str] = []
    aerospike_evidence: list[str] = []
    impacted: set[str] = {service}
    timeline: list[str] = []

    for obs in observations:
        tool = obs.get("tool", "")
        output = obs.get("output", {})
        if tool == "recent_deployments":
            for dep in output.get("deployments", []):
                summary = (
                    f"{dep['service']} deployed {dep['version']} at "
                    f"{dep['timestamp']} by {dep['author']} - "
                    f"{dep['change_summary']}"
                )
                deploy_evidence.append(summary)
                timeline.append(f"[deploy] {summary}")
        elif tool == "search_logs":
            for log in output.get("logs", [])[:5]:
                log_evidence.append(
                    f"{log['timestamp']} [{log['service']}] "
                    f"{log['level']}: {log['message']}"
                )
                impacted.add(log["service"])
        elif tool == "query_metrics":
            samples = output.get("samples", [])
            if samples:
                values = [s["value"] for s in samples]
                avg = sum(values) / len(values)
                metric_evidence.append(
                    f"{samples[0]['metric']} on {samples[0]['service']} "
                    f"avg={avg:.3f} (n={len(values)})"
                )
        elif tool == "fetch_traces":
            for tr in output.get("traces", [])[:3]:
                trace_evidence.append(
                    f"{tr['service']}::{tr['operation']} "
                    f"{tr['duration_ms']}ms status={tr['status']} "
                    f"err={tr.get('error_message')}"
                )
                impacted.add(tr["service"])
        elif tool == "get_service_dependencies":
            dep = output.get("dependency", {})
            for c in dep.get("consumed_by", []):
                impacted.add(c)
        elif tool == "find_similar_incidents":
            for inc in output.get("incidents", [])[:2]:
                similar_evidence.append(
                    f"{inc['incident_id']}: {inc['title']} - "
                    f"{inc['root_cause']} (similarity="
                    f"{inc['similarity_score']:.2f})"
                )
        elif tool == "query_database":
            if not output.get("available"):
                continue
            qn = output.get("query_name")
            for row in output.get("rows", [])[:5]:
                if qn == "policy_status":
                    db_evidence.append(
                        f"DB.policies[{row.get('policy_id')}] "
                        f"status={row.get('status')} "
                        f"provider={row.get('provider')} "
                        f"error={row.get('provider_error')}"
                    )
                elif qn == "transaction_journey":
                    db_evidence.append(
                        f"DB.tx[{row.get('transaction_id')}] "
                        f"state={row.get('transaction_state')} "
                        f"policy={row.get('policy_status') or 'none'} "
                        f"err={row.get('provider_error')}"
                    )
                elif qn == "recent_failed_policies_by_provider":
                    db_evidence.append(
                        f"DB.failed_policy[{row.get('policy_id')}] "
                        f"tx={row.get('transaction_id')} "
                        f"err={row.get('provider_error')}"
                    )
                else:
                    db_evidence.append(f"DB.{qn}: {row}")
        elif tool == "query_aerospike":
            if not output.get("available"):
                continue
            op = output.get("operation")
            key = output.get("key")
            if not output.get("found"):
                aerospike_evidence.append(
                    f"Aerospike.{op}[{key}] not_found"
                )
                continue
            bins = ((output.get("record") or {}).get("bins")) or {}
            if op == "policy_cache_get":
                aerospike_evidence.append(
                    f"Aerospike.policy_cache[{key}] "
                    f"status={bins.get('status')} "
                    f"provider={bins.get('provider')} "
                    f"provider_error={bins.get('provider_error')}"
                )
            elif op == "transaction_state_get":
                aerospike_evidence.append(
                    f"Aerospike.tx_state[{key}] "
                    f"state={bins.get('state')} "
                    f"attempts={bins.get('attempts')} "
                    f"last_error={bins.get('last_error')}"
                )
            elif op == "idempotency_get":
                aerospike_evidence.append(
                    f"Aerospike.idempotency[{key}] "
                    f"attempts={bins.get('attempts')} "
                    f"in_flight={bins.get('in_flight')} "
                    f"outcome={bins.get('outcome')}"
                )
            else:
                aerospike_evidence.append(
                    f"Aerospike.{op}[{key}] bins={bins}"
                )

    # -- Guardrail: no service-specific evidence at all --------------------
    # ``similar_incidents`` and dependency data are excluded from this
    # check on purpose - they always return *something* and can't
    # substitute for direct observation of the alerting service.
    # DB and Aerospike evidence count because they're ground-truth about
    # the exact transaction/policy the alert references.
    if not (
        deploy_evidence
        or log_evidence
        or metric_evidence
        or trace_evidence
        or db_evidence
        or aerospike_evidence
    ):
        return _insufficient_evidence_payload(service, alert)

    # -- Heuristic hypothesis selection ------------------------------------
    pool_signal = any(
        _matches(text, r"pool|hikari|connection.*exhaust|too many connection")
        for text in log_evidence + deploy_evidence
    )
    recent_deploy = any(
        _matches(text, r"pool|retry|connection", flags=re.I)
        for text in deploy_evidence
    )

    if pool_signal and recent_deploy:
        primary_statement = (
            f"Recent deployment of `{service}` reduced the database "
            "connection pool size, causing pool exhaustion under normal "
            "load. Downstream callers time out and the api-gateway trips "
            "its circuit breaker."
        )
        primary_conf = 0.86
        remediation = [
            {
                "title": "Roll back the offending deployment",
                "description": (
                    f"Revert {service} to the previous stable version to "
                    "restore the prior connection pool sizing."
                ),
                "priority": "immediate",
                "owner_hint": "on-call for " + service,
            },
            {
                "title": "Add a guardrail on connection pool changes",
                "description": (
                    "Require a canary + load test whenever pool sizing "
                    "config changes."
                ),
                "priority": "short_term",
                "owner_hint": "platform",
            },
            {
                "title": "Backfill DB-pool SLO alerting",
                "description": (
                    "Alert when db_connection_pool_usage > 0.85 for 2m "
                    "and pair it with an auto-runbook."
                ),
                "priority": "long_term",
                "owner_hint": "observability",
            },
        ]
    else:
        primary_statement = (
            f"Elevated error rate and latency on `{service}` correlate "
            "with a recent change. Investigation should focus on the "
            "most recent deployment and downstream saturation."
        )
        primary_conf = 0.55
        remediation = [
            {
                "title": "Investigate most recent deployment",
                "description": (
                    "Correlate deploy time with the onset of alerts and "
                    "consider rollback."
                ),
                "priority": "immediate",
                "owner_hint": "on-call for " + service,
            }
        ]

    alternates = [
        {
            "statement": (
                "Upstream traffic spike overwhelmed the service without a "
                "code change."
            ),
            "confidence": 0.18,
            "supporting_evidence": [],
            "contradicting_evidence": [
                "Baseline RPS unchanged; degradation begins at deploy time.",
            ],
        },
        {
            "statement": (
                "Downstream database instance failed independently of the "
                "deploy."
            ),
            "confidence": 0.12,
            "supporting_evidence": [],
            "contradicting_evidence": [
                "DB-side metrics show pool saturation, not host failure.",
            ],
        },
    ]

    return {
        "headline": (
            f"Likely root cause: connection pool misconfiguration in "
            f"latest {service} deployment"
        ),
        "summary": (
            f"An alert for `{service}` fired at "
            f"{alert.get('fired_at', 'recently')}. Evidence across "
            "deployments, logs, metrics, and traces converges on a recent "
            "deploy that changed the DB connection pool. The pool "
            "saturated, cascading errors upstream."
        ),
        "primary_hypothesis": {
            "statement": primary_statement,
            "confidence": primary_conf,
            "supporting_evidence": (
                aerospike_evidence[:2]
                + db_evidence[:3]
                + deploy_evidence[:2]
                + log_evidence[:3]
                + trace_evidence[:2]
                + similar_evidence[:1]
            ),
            "contradicting_evidence": [],
        },
        "alternate_hypotheses": alternates,
        "impacted_services": sorted(impacted),
        "detection_signals": metric_evidence[:5],
        "timeline": timeline[:10],
        "remediation": remediation,
        "confidence": primary_conf,
        "references": [],
    }


def _matches(text: str, pattern: str, flags: int = re.IGNORECASE) -> bool:
    return re.search(pattern, text, flags) is not None


def _insufficient_evidence_payload(
    service: str, alert: dict[str, Any]
) -> dict[str, Any]:
    """FINALIZE payload used when tools returned no direct signals.

    Kept intentionally cautious - low confidence, empty supporting
    evidence, and remediation that steers the on-call toward
    investigating telemetry gaps rather than the (unknown) root cause.
    """
    return {
        "headline": f"Insufficient evidence for {service}",
        "summary": (
            f"Investigation ran to completion but no direct telemetry "
            f"(logs, metrics, deployments, or traces) was found for "
            f"`{service}` in the lookback window. Reporting "
            "insufficient evidence rather than guessing."
        ),
        "primary_hypothesis": {
            "statement": (
                f"Insufficient evidence to determine the root cause "
                f"of `{alert.get('title', 'the alert')}`."
            ),
            "confidence": 0.1,
            "supporting_evidence": [],
            "contradicting_evidence": [],
        },
        "alternate_hypotheses": [],
        "impacted_services": [service],
        "detection_signals": [],
        "timeline": [],
        "remediation": [
            {
                "title": "Verify telemetry pipeline for the service",
                "description": (
                    f"Confirm that logs, metrics, and traces are being "
                    f"emitted and indexed for `{service}`."
                ),
                "priority": "immediate",
                "owner_hint": "observability",
            },
            {
                "title": "Extend the lookback window",
                "description": (
                    "Retry the investigation with a wider time range "
                    "(e.g. 4h) in case the causal event predates the "
                    "default window."
                ),
                "priority": "short_term",
                "owner_hint": "on-call",
            },
        ],
        "confidence": 0.1,
        "references": [],
    }
