from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from qdrant_client.http.exceptions import UnexpectedResponse

from app.api.routes import router
from app.core.config import get_settings
from app.services.engine_factory import create_face_engine
from app.services.faiss_sample_index import FaissFromQdrantSearch
from app.services.face_service import FaceService
from app.services.qdrant_store import QdrantDirectSearch, QdrantStore
from app.services.vector_types import VectorSearch

settings = get_settings()


def _configure_pipeline_timing_logs() -> None:
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


def _build_vector_search(store: QdrantStore) -> VectorSearch:
    if settings.effective_vector_search_backend == "faiss":
        faiss_search = FaissFromQdrantSearch(settings, store)
        faiss_search.rebuild()
        return faiss_search
    return QdrantDirectSearch(store)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_pipeline_timing_logs()
    face_engine = create_face_engine(settings)
    qdrant_store = QdrantStore(settings)
    try:
        qdrant_store.ensure_collection()
    except UnexpectedResponse as exc:
        if exc.status_code == 401:
            raise RuntimeError(
                "Qdrant returned 401 Unauthorized: the API key does not match this server. "
                f"Set QDRANT_API_KEY to the same value as Qdrant's QDRANT__SERVICE__API_KEY "
                f"(for the instance at {settings.qdrant_url!r}). "
                "If Qdrant was started without an API key, leave QDRANT_API_KEY empty."
            ) from exc
        raise
    vector_search = _build_vector_search(qdrant_store)
    app.state.face_service = FaceService(
        settings=settings,
        face_engine=face_engine,
        qdrant_store=qdrant_store,
        vector_search=vector_search,
    )
    yield


app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(router, prefix=settings.api_v1_prefix)
