"""Investigation retrieval endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.dependencies import get_investigation_service
from app.core.exceptions import InvestigationNotFoundError
from app.models.investigation import Investigation
from app.services.investigation_service import InvestigationService

router = APIRouter(prefix="/investigations", tags=["investigations"])


@router.get(
    "",
    response_model=list[Investigation],
    summary="List recent investigations.",
)
async def list_investigations(
    limit: int = Query(default=25, ge=1, le=200),
    service: InvestigationService = Depends(get_investigation_service),
) -> list[Investigation]:
    return await service.list(limit=limit)


@router.get(
    "/{investigation_id}",
    response_model=Investigation,
    summary="Fetch a single investigation.",
)
async def get_investigation(
    investigation_id: str,
    service: InvestigationService = Depends(get_investigation_service),
) -> Investigation:
    try:
        return await service.get(investigation_id)
    except InvestigationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
