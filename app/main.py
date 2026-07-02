"""FastAPI application factory and middleware."""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__
from app.api.dependencies import get_llm
from app.api.routes.alerts import router as alerts_router
from app.api.routes.health import router as health_router
from app.api.routes.investigations import router as investigations_router
from app.config import get_settings
from app.core.exceptions import RCAError
from app.core.logging import configure_logging, get_logger
from app.services.data_store import get_store
from app.tools.registry import get_registry

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()

    # Warm all singletons so failures surface at startup.
    llm = get_llm()
    registry = get_registry()
    store = get_store()

    log.info(
        "app.startup",
        env=settings.app_env,
        version=__version__,
        llm=llm.name,
        tools=registry.names(),
        seeded_logs=len(store.logs),
        seeded_metrics=len(store.metrics),
    )

    yield

    log.info("app.shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Agentic AI Root Cause Finder",
        description=(
            "Production-style prototype of an autonomous RCA agent. "
            "Accepts alerts, builds a normalized incident context, then "
            "runs a ReAct-style loop over investigative tools "
            "(logs, metrics, deployments, traces, dependencies, and "
            "similar-incident search) to produce a structured RCA "
            "report."
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    @app.middleware("http")
    async def add_request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            path=request.url.path,
            method=request.method,
        )
        start = time.monotonic()
        try:
            response = await call_next(request)
        finally:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            log.info("http.request", elapsed_ms=elapsed_ms)
        response.headers["x-request-id"] = request_id
        return response

    @app.exception_handler(RCAError)
    async def rca_error_handler(_: Request, exc: RCAError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": "validation_error", "detail": exc.errors()},
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(
        _: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code, content={"error": exc.detail}
        )

    @app.get("/", tags=["meta"], include_in_schema=False)
    async def index() -> dict[str, str]:
        return {
            "name": settings.app_name,
            "version": __version__,
            "docs": "/docs",
        }

    app.include_router(health_router)
    app.include_router(alerts_router)
    app.include_router(investigations_router)

    return app


app = create_app()
