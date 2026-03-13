from __future__ import annotations

from app.core.config import Settings
from app.services.face_engine import BaseFaceEngine
from app.services.insightface_engine import InsightFaceEngine
from app.services.opencv_engine import OpenCVFaceEngine


def create_face_engine(settings: Settings) -> BaseFaceEngine:
    backend = settings.face_backend.lower()
    if backend == "insightface":
        return InsightFaceEngine(settings)
    if backend == "opencv":
        return OpenCVFaceEngine(settings)
    raise RuntimeError(
        f"Unsupported FACE_BACKEND '{settings.face_backend}'. "
        "Expected 'insightface' or 'opencv'."
    )
