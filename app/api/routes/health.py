"""Health and readiness endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app import __version__
from app.api.dependencies import get_llm
from app.config import Settings, get_settings
from app.llm.base import LLMAdapter
from app.tools.registry import get_registry

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readiness")
def readiness(
    settings: Settings = Depends(get_settings),
    llm: LLMAdapter = Depends(get_llm),
) -> dict[str, object]:
    return {
        "status": "ready",
        "version": __version__,
        "environment": settings.app_env,
        "llm_provider": llm.name,
        "tools": get_registry().names(),
    }
