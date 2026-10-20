"""FastAPI application entry point."""

import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.cache.redis_client import close_redis
from app.config import settings
from app.services.checkpoint import close_checkpointer, init_checkpointer
from app.services.database import close_database, init_database
from app.services.session_manager import reset_session_manager
from app.services.operation_store import get_operation_store
from app.services.model_capabilities import get_model_capabilities, probe_model_capabilities
from app.errors import PublicAppError
from app.security import validate_auth_configuration
from app.logging_security import install_safe_access_logging
from app.upload_limits import PreparationUploadLimit

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
install_safe_access_logging()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    validate_auth_configuration()
    capability_probe = None
    try:
        await init_database()
        await init_checkpointer()
        await get_operation_store().recover_running()
        reset_session_manager()
        capability_probe = asyncio.create_task(
            probe_model_capabilities(), name="model-capability-probe"
        )
        yield
    finally:
        if capability_probe is not None and not capability_probe.done():
            capability_probe.cancel()
            await asyncio.gather(capability_probe, return_exceptions=True)
        await close_checkpointer()
        await close_database()
        await close_redis()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Mockview API",
        description="JD and resume driven AI interview practice platform",
        version="0.2.0",
        lifespan=lifespan,
    )

    cors_origins = settings.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=cors_origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*", "Authorization"],
    )

    app.include_router(api_router, prefix="/api")
    app.add_middleware(PreparationUploadLimit)

    @app.exception_handler(PublicAppError)
    async def public_app_error_handler(_request: Request, exc: PublicAppError):
        headers = (
            {"Retry-After": str(exc.retry_after_seconds)}
            if exc.retry_after_seconds is not None else None
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.public().model_dump(mode="json")},
            headers=headers,
        )

    @app.get("/health")
    async def health_check():
        return {
            "status": "ok",
            "model": settings.llm_model,
            "debug": settings.debug,
            "redis_enabled": settings.redis_enabled,
            "capabilities": get_model_capabilities(),
        }

    return app


app = create_app()
