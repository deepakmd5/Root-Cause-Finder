"""ReAct-style Root Cause Analysis agent.

Loop:

    1. Ask the LLM for a decision given the current transcript.
    2. If it wants a tool -> execute the tool, append the observation,
       loop.
    3. If it wants to finalize -> parse the RCA report and stop.
    4. Cap iterations, capture every step for observability.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from app.agents.prompts import SYSTEM_PROMPT
from app.config import get_settings
from app.core.exceptions import AgentBudgetExceeded, LLMError
from app.core.logging import get_logger
from app.llm.base import AgentDecision, DecisionType, LLMAdapter, LLMMessage
from app.models.context import (
    AerospikeRecord,
    DbRecord,
    DeploymentEvent,
    LogEntry,
    MetricSample,
    NormalizedContext,
    ServiceDependency,
    SimilarIncident,
    TraceSpan,
)
from app.models.investigation import AgentStep, Investigation, StepType
from app.models.rca import (
    Hypothesis,
    RCAReport,
    RemediationAction,
    RemediationPriority,
)
from app.tools.registry import ToolRegistry

log = get_logger(__name__)


class RCAAgent:
    """Autonomous RCA agent using a ReAct loop over pluggable tools."""

    def __init__(
        self,
        llm: LLMAdapter,
        registry: ToolRegistry,
        max_iterations: int | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        settings = get_settings()
        self.llm = llm
        self.registry = registry
        self.max_iterations = max_iterations or settings.agent_max_iterations
        self.timeout_seconds = timeout_seconds or settings.agent_timeout_seconds

    async def investigate(self, investigation: Investigation) -> Investigation:
        """Run the agent loop against an already-normalized investigation."""
        assert investigation.context is not None, "context must be built first"

        investigation.mark_started()
        started = time.monotonic()
        log.info(
            "agent.start",
            investigation_id=investigation.id,
            alert_id=investigation.alert.id,
            llm=self.llm.name,
        )

        messages: list[LLMMessage] = [
            LLMMessage(role="system", content=SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=(
                    "Investigate the following incident. Use the tools to "
                    "gather evidence, then produce a structured RCA.\n\n"
                    f"CONTEXT_SUMMARY: {investigation.context.summary}"
                ),
            ),
            LLMMessage(
                role="user",
                content="ALERT_JSON::"
                + investigation.alert.model_dump_json(),
            ),
        ]

        try:
            for i in range(1, self.max_iterations + 1):
                if time.monotonic() - started > self.timeout_seconds:
                    raise AgentBudgetExceeded(
                        f"Timeout after {self.timeout_seconds}s"
                    )

                decision = await self._decide(messages)
                self._record_thought(investigation, i, decision)

                if decision.type == DecisionType.FINALIZE:
                    report = self._parse_report(decision, investigation.context)
                    report = self._apply_guardrails(report, investigation.context)
                    investigation.steps.append(
                        AgentStep(
                            index=len(investigation.steps) + 1,
                            type=StepType.FINAL,
                            content=report.headline,
                        )
                    )
                    investigation.mark_completed(report)
                    log.info(
                        "agent.completed",
                        investigation_id=investigation.id,
                        iterations=i,
                        confidence=report.confidence,
                    )
                    return investigation

                # Otherwise: USE_TOOL
                await self._execute_tool(
                    decision, investigation, messages
                )
            raise AgentBudgetExceeded(
                f"Exceeded max iterations ({self.max_iterations})"
            )

        except AgentBudgetExceeded as exc:
            log.warning("agent.budget_exceeded", error=str(exc))
            investigation.mark_failed(str(exc))
            return investigation
        except LLMError as exc:
            log.exception("agent.llm_error")
            investigation.mark_failed(f"LLM error: {exc}")
            return investigation
        except Exception as exc:  # noqa: BLE001
            log.exception("agent.unexpected_error")
            investigation.mark_failed(f"Unexpected error: {exc}")
            return investigation

    # -- Internals --------------------------------------------------------

    async def _decide(self, messages: list[LLMMessage]) -> AgentDecision:
        specs = self.registry.list_specs()
        try:
            return await asyncio.wait_for(
                self.llm.decide(messages, specs),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise LLMError("LLM decide() timed out") from exc

    def _record_thought(
        self,
        investigation: Investigation,
        iteration: int,
        decision: AgentDecision,
    ) -> None:
        investigation.steps.append(
            AgentStep(
                index=len(investigation.steps) + 1,
                type=StepType.THOUGHT,
                content=(
                    f"[iter {iteration}] {decision.thought or '(no thought)'}"
                ),
            )
        )

    async def _execute_tool(
        self,
        decision: AgentDecision,
        investigation: Investigation,
        messages: list[LLMMessage],
    ) -> None:
        if not decision.tool:
            raise LLMError("Decision USE_TOOL missing 'tool' field")

        started = time.monotonic()
        action_step = AgentStep(
            index=len(investigation.steps) + 1,
            type=StepType.ACTION,
            content=f"call {decision.tool}",
            tool=decision.tool,
            tool_input=decision.tool_input,
        )
        investigation.steps.append(action_step)

        result = await self.registry.execute(decision.tool, decision.tool_input)
        elapsed_ms = int((time.monotonic() - started) * 1000)

        # Merge tool output into the normalized context.
        self._merge_into_context(investigation.context, result.tool, result.data)

        observation = AgentStep(
            index=len(investigation.steps) + 1,
            type=StepType.OBSERVATION,
            content=result.summary,
            tool=result.tool,
            tool_input=result.input,
            tool_output_summary=result.summary,
            duration_ms=elapsed_ms,
        )
        investigation.steps.append(observation)

        # Feed the observation back into the LLM message history.
        messages.append(
            LLMMessage(
                role="assistant",
                content=(
                    f"THOUGHT: {decision.thought}\n"
                    f"ACTION: {decision.tool}({json.dumps(decision.tool_input)})"
                ),
            )
        )
        messages.append(
            LLMMessage(
                role="user",
                content="OBSERVATION::"
                + json.dumps(
                    {
                        "tool": result.tool,
                        "input": result.input,
                        "summary": result.summary,
                        "output": result.data,
                    },
                    default=str,
                ),
            )
        )

    def _merge_into_context(
        self,
        ctx: NormalizedContext | None,
        tool: str,
        data: dict[str, Any],
    ) -> None:
        if ctx is None:
            return
        if tool == "search_logs":
            ctx.logs.extend(LogEntry(**log_) for log_ in data.get("logs", []))
        elif tool == "query_metrics":
            ctx.metrics.extend(
                MetricSample(**s) for s in data.get("samples", [])
            )
        elif tool == "recent_deployments":
            ctx.deployments.extend(
                DeploymentEvent(**d) for d in data.get("deployments", [])
            )
        elif tool == "fetch_traces":
            ctx.traces.extend(
                TraceSpan(**t) for t in data.get("traces", [])
            )
        elif tool == "get_service_dependencies":
            dep = data.get("dependency")
            if dep:
                ctx.dependencies.append(ServiceDependency(**dep))
        elif tool == "find_similar_incidents":
            ctx.similar_incidents.extend(
                SimilarIncident(**i) for i in data.get("incidents", [])
            )
        elif tool == "query_database":
            ctx.db_records.append(
                DbRecord(
                    query_name=data.get("query_name") or "",
                    params=data.get("params") or {},
                    row_count=int(data.get("row_count", 0)),
                    rows=list(data.get("rows", [])),
                    fetched_at=None,  # kept as str inside data for provenance
                    available=bool(data.get("available", False)),
                    error=data.get("error"),
                )
            )
        elif tool == "query_aerospike":
            ctx.aerospike_records.append(
                AerospikeRecord(
                    operation=data.get("operation") or "",
                    params=data.get("params") or {},
                    namespace=data.get("namespace"),
                    set_name=data.get("set"),
                    key=data.get("key"),
                    found=bool(data.get("found", False)),
                    record=data.get("record"),
                    fetched_at=None,  # kept as str inside data for provenance
                    available=bool(data.get("available", False)),
                    error=data.get("error"),
                )
            )

    def _parse_report(
        self,
        decision: AgentDecision,
        context: NormalizedContext,
    ) -> RCAReport:
        payload = decision.final_answer or {}

        def as_hypothesis(raw: dict[str, Any]) -> Hypothesis:
            return Hypothesis(
                statement=raw.get("statement", "(no statement)"),
                confidence=float(raw.get("confidence", 0.0)),
                supporting_evidence=list(raw.get("supporting_evidence", [])),
                contradicting_evidence=list(
                    raw.get("contradicting_evidence", [])
                ),
            )

        primary = as_hypothesis(
            payload.get(
                "primary_hypothesis",
                {"statement": "unknown", "confidence": 0.0},
            )
        )
        alternates = [
            as_hypothesis(h) for h in payload.get("alternate_hypotheses", [])
        ]
        remediation = [
            RemediationAction(**r) for r in payload.get("remediation", [])
        ]

        impacted = payload.get("impacted_services") or [
            context.alert.service
        ]

        return RCAReport(
            headline=payload.get(
                "headline",
                f"Investigation of {context.alert.service}",
            ),
            summary=payload.get("summary", ""),
            primary_hypothesis=primary,
            alternate_hypotheses=alternates,
            impacted_services=list(impacted),
            detection_signals=list(payload.get("detection_signals", [])),
            timeline=list(payload.get("timeline", [])),
            remediation=remediation,
            confidence=float(
                payload.get("confidence", primary.confidence)
            ),
            references=list(payload.get("references", [])),
        )

    # -- Guardrails -------------------------------------------------------

    def _apply_guardrails(
        self,
        report: RCAReport,
        context: NormalizedContext,
    ) -> RCAReport:
        """Post-process the report to prevent evidence-free RCAs.

        The LLM's own answer is trusted only when *service-specific*
        telemetry actually landed in the context during the
        investigation. If nothing did (unknown service, silent pipeline,
        stale lookback, ...) we replace the primary hypothesis with an
        honest "insufficient evidence" verdict rather than emit a
        confident-sounding hallucination.
        """
        if _has_service_specific_evidence(context):
            return report

        service = context.alert.service
        insufficient = Hypothesis(
            statement=(
                "Insufficient evidence to determine root cause. "
                f"No service-specific telemetry (logs, metrics, traces, "
                f"deployments) was found for `{service}` within the "
                f"lookback window."
            ),
            confidence=0.1,
            supporting_evidence=[],
            contradicting_evidence=[],
        )
        remediation = [
            RemediationAction(
                title="Verify service identity and telemetry pipeline",
                description=(
                    f"Confirm that the alert's service name '{service}' "
                    "matches what the observability stack indexes. Check "
                    "log shipping, metric scraping, and trace ingestion "
                    "for the service."
                ),
                priority=RemediationPriority.IMMEDIATE,
                owner_hint="observability",
            ),
            RemediationAction(
                title="Widen the investigation lookback window",
                description=(
                    "Re-run the investigation with an extended lookback "
                    "(e.g. 4h) in case the causing event fell outside "
                    "the default 30m window."
                ),
                priority=RemediationPriority.SHORT_TERM,
                owner_hint="on-call",
            ),
        ]
        summary = (
            f"Investigation completed but no service-specific telemetry "
            f"was gathered for `{service}`. Reporting insufficient "
            "evidence rather than guessing a root cause."
        )
        headline = f"Insufficient evidence for {service}"

        # Downgrade any alternate hypotheses so they can't be confused
        # for a confident answer.
        downgraded_alternates = [
            Hypothesis(
                statement=alt.statement,
                confidence=min(alt.confidence, 0.15),
                supporting_evidence=alt.supporting_evidence,
                contradicting_evidence=alt.contradicting_evidence
                + ["Not supported by service-specific telemetry."],
            )
            for alt in report.alternate_hypotheses
        ]

        return report.model_copy(
            update={
                "headline": headline,
                "summary": summary,
                "primary_hypothesis": insufficient,
                "alternate_hypotheses": downgraded_alternates,
                "remediation": remediation,
                "confidence": 0.1,
            }
        )


def _has_service_specific_evidence(context: NormalizedContext) -> bool:
    """True iff at least one telemetry source has direct evidence.

    Direct evidence means:

    * A log/metric/trace/deploy line that names the alerting service, OR
    * A DB record fetched *for this alert's* transaction_id / policy_id
      identifiers (as declared in ``alert.labels``), OR
    * An Aerospike record fetched for the same alert-carried
      transaction_id / policy_id / idempotency_key.

    Similar-incidents and dependency-graph responses are excluded on
    purpose: the historical KB and topology map always return "something"
    for any query, so leaning on them would let a hallucinated RCA slip
    through the guardrail.
    """
    service = context.alert.service.lower()
    if any(log.service.lower() == service for log in context.logs):
        return True
    if any(m.service.lower() == service for m in context.metrics):
        return True
    if any(d.service.lower() == service for d in context.deployments):
        return True
    if any(t.service.lower() == service for t in context.traces):
        return True

    # DB rows only count when they were fetched for the *specific*
    # identifiers the alert already carries. Otherwise the LLM could
    # dredge up unrelated rows and claim them as evidence.
    labels = {k.lower(): v for k, v in (context.alert.labels or {}).items()}
    tx_id = labels.get("transactionid") or labels.get("transaction_id")
    policy_id = labels.get("policyid") or labels.get("policy_id")
    idem_key = labels.get("idempotencykey") or labels.get("idempotency_key")
    for rec in context.db_records:
        if not rec.available or rec.row_count == 0:
            continue
        p = {k.lower(): v for k, v in (rec.params or {}).items()}
        if tx_id and str(p.get("transaction_id", "")) == str(tx_id):
            return True
        if policy_id and str(p.get("policy_id", "")) == str(policy_id):
            return True

    # Aerospike records: same rule. The record must have been fetched
    # for one of the alert's own identifiers AND the key must have been
    # found (a not-found lookup is still valuable *context*, but it is
    # not evidence *of the service failing right now*).
    for rec in context.aerospike_records:
        if not rec.available or not rec.found:
            continue
        p = {k.lower(): v for k, v in (rec.params or {}).items()}
        if tx_id and str(p.get("transaction_id", "")) == str(tx_id):
            return True
        if policy_id and str(p.get("policy_id", "")) == str(policy_id):
            return True
        if idem_key and str(p.get("idempotency_key", "")) == str(idem_key):
            return True
    return False
