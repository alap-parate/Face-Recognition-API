from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from app.core.config import Settings
from app.core.pipeline_timing import PipelineTimer
from app.services.face_engine import BaseFaceEngine, FaceVectorSample, InvalidFaceImageError
from app.services.qdrant_store import QdrantStore
from app.services.vector_types import RankedPerson, VectorSearch


@dataclass(slots=True)
class EnrollmentResult:
    external_id: str
    name: str
    created: bool
    submitted_image_count: int
    stored_sample_count: int
    dropped_sample_count: int
    samples: list[FaceVectorSample]


@dataclass(slots=True)
class RecognitionCandidate:
    external_id: str
    name: str
    distance: float
    similarity: float
    sample_count: int


@dataclass(slots=True)
class RecognitionResult:
    matched: bool
    threshold: float
    query_detection_score: float
    query_quality_score: float
    candidates: list[RecognitionCandidate]


class FaceService:
    def __init__(
        self,
        settings: Settings,
        face_engine: BaseFaceEngine,
        qdrant_store: QdrantStore,
        vector_search: VectorSearch,
    ) -> None:
        self.settings = settings
        self.face_engine = face_engine
        self._qdrant = qdrant_store
        self._vector_search = vector_search

    @property
    def model_name(self) -> str:
        return self.face_engine.model_name

    def qdrant_health(self) -> bool:
        return self._qdrant.health_check()

    def enroll(
        self,
        org_id: str,
        external_id: str,
        name: str,
        is_active: bool,
        image_payloads: Sequence[bytes],
    ) -> EnrollmentResult:
        org_id = org_id.strip()
        external_id = external_id.strip()
        name = name.strip()

        if not org_id:
            raise ValueError("org_id is required.")
        if not external_id:
            raise ValueError("external_id is required.")
        if not name:
            raise ValueError("name is required.")
        if not (
            self.settings.min_enrollment_images
            <= len(image_payloads)
            <= self.settings.max_enrollment_images
        ):
            raise InvalidFaceImageError(
                f"Provide exactly {self.settings.min_enrollment_images} images for enrollment."
            )

        samples = [
            self.face_engine.extract_sample(payload, sample_index=index)
            for index, payload in enumerate(image_payloads, start=1)
        ]
        svc_timer = PipelineTimer(self.settings.pipeline_timing)
        t = svc_timer.start()
        _aggregate, kept_samples = self.face_engine.aggregate_samples(samples)
        t = svc_timer.record("aggregate_ms", t)
        dropped_sample_count = len(samples) - len(kept_samples)

        existing_pid = self._qdrant.find_existing_person_id(org_id, external_id)
        created = existing_pid is None
        person_id = existing_pid if existing_pid is not None else str(uuid.uuid4())

        self._qdrant.delete_identity_points(org_id, external_id)
        self._qdrant.upsert_enrollment(
            org_id=org_id,
            external_id=external_id,
            name=name,
            is_active=is_active,
            person_id=person_id,
            kept_samples=kept_samples,
            sample_count=len(kept_samples),
        )
        self._vector_search.sync_after_enroll()
        svc_timer.record("qdrant_ms", t)
        svc_timer.log(
            "face_service.enroll",
            external_id=external_id,
            org_id=org_id,
            images=len(image_payloads),
        )
        return EnrollmentResult(
            external_id=external_id,
            name=name,
            created=created,
            submitted_image_count=len(image_payloads),
            stored_sample_count=len(kept_samples),
            dropped_sample_count=dropped_sample_count,
            samples=kept_samples,
        )

    def recognize(
        self,
        org_id: str,
        image_payload: bytes,
        top_k: int | None = None,
    ) -> RecognitionResult:
        org_id = org_id.strip()
        if not org_id:
            raise ValueError("org_id is required.")

        requested_top_k = top_k or self.settings.recognition_top_k_default
        limited_top_k = min(
            max(1, requested_top_k),
            self.settings.recognition_top_k_max,
        )

        query_sample = self.face_engine.extract_sample(image_payload, sample_index=1)
        probe_k = self.settings.faiss_sample_probe_count(limited_top_k)
        svc_timer = PipelineTimer(self.settings.pipeline_timing)
        t = svc_timer.start()
        ranked = self._vector_search.search_ranked_persons(
            np.asarray(query_sample.embedding, dtype=np.float32),
            org_id,
            limited_top_k,
            probe_k,
        )
        label = (
            "faiss_search_ms"
            if self.settings.effective_vector_search_backend == "faiss"
            else "qdrant_search_ms"
        )
        svc_timer.record(label, t)
        svc_timer.log(
            "face_service.recognize",
            top_k=limited_top_k,
            probe_k=probe_k,
        )

        candidates = [
            RecognitionCandidate(
                external_id=rp.external_id,
                name=rp.name,
                distance=float(rp.distance),
                similarity=max(0.0, 1.0 - float(rp.distance)),
                sample_count=rp.sample_count,
            )
            for rp in ranked
        ]
        matched = bool(candidates) and (
            candidates[0].distance <= self.settings.recognition_match_threshold
        )
        return RecognitionResult(
            matched=matched,
            threshold=self.settings.recognition_match_threshold,
            query_detection_score=query_sample.detection_score,
            query_quality_score=query_sample.quality_score,
            candidates=candidates,
        )
