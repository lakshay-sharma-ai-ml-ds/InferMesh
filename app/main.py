"""
InferMesh FastAPI Application
==============================
REST API server exposing inference endpoints, health checks,
and Prometheus metrics.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import Optional

import structlog
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.api.middleware import RateLimitMiddleware, RequestIDMiddleware
from app.api.routes import router
from app.config import get_config
from app.models import InferenceRequest, InferenceResponse
from app.orchestrator import InferMeshOrchestrator

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def configure_logging(log_level: str = "INFO") -> None:
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level),
    )


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

_orchestrator: Optional[InferMeshOrchestrator] = None


def get_orchestrator() -> InferMeshOrchestrator:
    if _orchestrator is None:
        raise RuntimeError("Orchestrator not initialized")
    return _orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    global _orchestrator
    config = get_config()
    configure_logging(config.log_level)

    logger = structlog.get_logger("app.main")
    logger.info("InferMesh starting up", version=config.version, env=config.env)

    _orchestrator = InferMeshOrchestrator(config)
    app.state.orchestrator = _orchestrator
    await _orchestrator.start()

    yield  # Application runs here

    logger.info("InferMesh shutting down")
    await _orchestrator.stop()


def create_app() -> FastAPI:
    config = get_config()

    app = FastAPI(
        title="InferMesh",
        description="Distributed LLM Inference Orchestrator",
        version=config.version,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(RateLimitMiddleware, max_requests=1000, window_s=1.0)

    # Include routers
    app.include_router(router)

    # Prometheus metrics endpoint
    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        data = generate_latest()
        return PlainTextResponse(data, media_type=CONTENT_TYPE_LATEST)

    # Health check
    @app.get("/health", tags=["health"])
    async def health():
        orch = get_orchestrator()
        cluster = orch.health_monitor.get_cluster_health()
        return {
            "status": cluster.overall_status,
            "healthy_workers": cluster.healthy_workers,
            "total_workers": cluster.total_workers,
            "version": config.version,
        }

    # Readiness probe
    @app.get("/ready", tags=["health"])
    async def ready():
        orch = get_orchestrator()
        workers = orch.gpu_manager.get_healthy_workers()
        if not workers:
            raise HTTPException(status_code=503, detail="No healthy workers")
        return {"ready": True, "workers": len(workers)}

    return app


app = create_app()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def cli_entry():
    import uvicorn
    config = get_config()
    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        workers=config.workers,
        loop="asyncio",
        log_level=config.log_level.lower(),
    )


if __name__ == "__main__":
    cli_entry()
