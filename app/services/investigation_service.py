"""Investigation orchestration service.

The FastAPI routes call into this thin service which:
- normalizes the incoming alert into a canonical context
- launches the agent
- stores + retrieves investigations
"""
from __future__ import annotations

from asyncio import Lock
from typing import Iterable

from app.agents.rca_agent import RCAAgent
from app.context.normalizer import ContextNormalizer
from app.core.exceptions import InvestigationNotFoundError
from app.core.logging import get_logger
from app.llm.base import LLMAdapter
from app.models.alert import Alert, IncomingAlert
from app.models.investigation import Investigation
from app.tools.registry import ToolRegistry

log = get_logger(__name__)


class InvestigationService:
    def __init__(
        self,
        llm: LLMAdapter,
        registry: ToolRegistry,
        normalizer: ContextNormalizer | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.normalizer = normalizer or ContextNormalizer()
        self._store: dict[str, Investigation] = {}
        self._lock = Lock()

    async def start_investigation(
        self, incoming: IncomingAlert
    ) -> Investigation:
        alert = Alert.from_incoming(incoming)
        context = self.normalizer.build(alert)
        investigation = Investigation(alert=alert, context=context)

        async with self._lock:
            self._store[investigation.id] = investigation

        log.info(
            "investigation.created",
            investigation_id=investigation.id,
            service=alert.service,
        )

        agent = RCAAgent(llm=self.llm, registry=self.registry)
        return await agent.investigate(investigation)

    async def get(self, investigation_id: str) -> Investigation:
        async with self._lock:
            inv = self._store.get(investigation_id)
        if inv is None:
            raise InvestigationNotFoundError(
                f"No investigation with id={investigation_id}"
            )
        return inv

    async def list(self, limit: int = 50) -> list[Investigation]:
        async with self._lock:
            values: Iterable[Investigation] = self._store.values()
            ordered = sorted(
                values, key=lambda i: i.created_at, reverse=True
            )
        return ordered[:limit]
