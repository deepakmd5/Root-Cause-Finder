"""Alert ingestion + investigation kickoff."""
from __future__ import annotations

from fastapi import APIRouter, Depends, status

from app.api.dependencies import get_investigation_service
from app.models.alert import IncomingAlert
from app.models.investigation import Investigation
from app.services.investigation_service import InvestigationService

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.post(
    "",
    response_model=Investigation,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest an alert and run an RCA investigation.",
    response_description="The completed (or failed) investigation.",
)
async def ingest_alert(
    incoming: IncomingAlert,
    service: InvestigationService = Depends(get_investigation_service),
) -> Investigation:
    """Accept an alert, normalize it, and run the agent synchronously.

    The response includes:
    - The alert (with server-assigned id)
    - The normalized context that was built
    - The full agent step-by-step trace
    - The final RCA report (if the investigation completed)
    """
    return await service.start_investigation(incoming)
