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
    DeploymentEvent,
    LogEntry,
    MetricSample,
    NormalizedContext,
    ServiceDependency,
    SimilarIncident,
    TraceSpan,
)
from app.models.investigation import AgentStep, Investigation, StepType
from app.models.rca import Hypothesis, RCAReport, RemediationAction
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
