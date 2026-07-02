"""FastAPI dependency wiring.

The service is intentionally stateless per-request but shares a small
number of process-scoped singletons: the LLM, the tool registry, and
the investigation store.
"""
from __future__ import annotations

from functools import lru_cache

from app.llm.base import LLMAdapter, build_llm
from app.services.investigation_service import InvestigationService
from app.tools.registry import get_registry


@lru_cache
def _shared_llm() -> LLMAdapter:
    return build_llm()


@lru_cache
def _shared_service() -> InvestigationService:
    return InvestigationService(llm=_shared_llm(), registry=get_registry())


def get_llm() -> LLMAdapter:
    return _shared_llm()


def get_investigation_service() -> InvestigationService:
    return _shared_service()


def reset_singletons() -> None:
    """For tests."""
    _shared_llm.cache_clear()
    _shared_service.cache_clear()
