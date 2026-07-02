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

        # Investigation plan - each step gathers a piece of the puzzle.
        # ``dedupe_key`` is the semantic identity of the step; two steps
        # with the same dedupe_key are considered "the same call".
        plan: list[tuple[str, dict[str, Any], str, str]] = [
            (
                "recent_deployments",
                {"service": service, "since_minutes": 240},
                f"A production alert fired for `{service}`. Recent deploys "
                "are the highest-prior suspect - let me check them first.",
                f"recent_deployments:{service}",
            ),
            (
                "search_logs",
                {"services": [service], "levels": ["ERROR", "WARN"]},
                "Now I need concrete error signatures. Pulling ERROR/WARN "
                f"logs for {service} to see the failure mode.",
                f"search_logs:{service}",
            ),
            (
                "query_metrics",
                {"service": service, "metric": "error_rate"},
                "Correlating logs with the error-rate metric to confirm "
                "impact and timing.",
                f"query_metrics:{service}:error_rate",
            ),
            (
                "query_metrics",
                {"service": service, "metric": "latency_p95_ms"},
                "Checking latency to see if this is a hard failure or a "
                "slow degradation.",
                f"query_metrics:{service}:latency_p95_ms",
            ),
            (
                "fetch_traces",
                {"service": service, "error_only": True},
                "Traces will show which downstream hop is actually failing.",
                f"fetch_traces:{service}",
            ),
            (
                "get_service_dependencies",
                {"service": service},
                "Mapping the dependency graph so I know the blast radius.",
                f"get_service_dependencies:{service}",
            ),
            (
                "find_similar_incidents",
                {
                    "service": service,
                    "keywords": ["pool", "timeout", "deploy"],
                },
                "Checking whether we have seen this signature before.",
                f"find_similar_incidents:{service}",
            ),
        ]

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
                "I have deployment history, error logs, degraded metrics, "
                "error traces, dependency topology, and matching past "
                "incidents. Time to synthesize the RCA."
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
                deploy_evidence[:2]
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
