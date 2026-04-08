from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.db.session import initialize_database
from app.services.engine_factory import create_face_engine
from app.services.face_service import FaceService

settings = get_settings()


def _configure_pipeline_timing_logs() -> None:
    """Route app.* INFO logs to stderr when pipeline timing is enabled."""
    if not settings.pipeline_timing:
        return
    app_logger = logging.getLogger("app")
    if app_logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    app_logger.addHandler(handler)
    app_logger.setLevel(logging.INFO)
    app_logger.propagate = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_pipeline_timing_logs()
    initialize_database()
    face_engine = create_face_engine(settings)
    app.state.face_service = FaceService(settings=settings, face_engine=face_engine)
    yield


app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(router, prefix=settings.api_v1_prefix)
