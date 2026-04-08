from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.pipeline_timing import PipelineTimer
from app.db.models import FaceSample, Person
from app.services.face_engine import BaseFaceEngine, FaceVectorSample, InvalidFaceImageError


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
    def __init__(self, settings: Settings, face_engine: BaseFaceEngine) -> None:
        self.settings = settings
        self.face_engine = face_engine

    @property
    def model_name(self) -> str:
        return self.face_engine.model_name

    def enroll(
        self,
        db: Session,
        external_id: str,
        name: str,
        image_payloads: Sequence[bytes],
    ) -> EnrollmentResult:
        external_id = external_id.strip()
        name = name.strip()

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
                f"Provide between {self.settings.min_enrollment_images} and "
                f"{self.settings.max_enrollment_images} images for enrollment."
            )

        samples = [
            self.face_engine.extract_sample(payload, sample_index=index)
            for index, payload in enumerate(image_payloads, start=1)
        ]
        svc_timer = PipelineTimer(self.settings.pipeline_timing)
        t = svc_timer.start()
        aggregate_embedding, kept_samples = self.face_engine.aggregate_samples(samples)
        t = svc_timer.record("aggregate_ms", t)
        dropped_sample_count = len(samples) - len(kept_samples)
        person = db.scalar(select(Person).where(Person.external_id == external_id))
        created = person is None
        now = datetime.now(timezone.utc)

        if person is None:
            person = Person(
                external_id=external_id,
                name=name,
                embedding=aggregate_embedding.tolist(),
                sample_count=len(kept_samples),
                last_enrolled_at=now,
            )
            db.add(person)
            db.flush()
        else:
            person.name = name
            person.embedding = aggregate_embedding.tolist()
            person.sample_count = len(kept_samples)
            person.last_enrolled_at = now
            db.execute(delete(FaceSample).where(FaceSample.person_id == person.id))
            db.flush()

        for sample_index, sample in enumerate(kept_samples, start=1):
            db.add(
                FaceSample(
                    person_id=person.id,
                    sample_index=sample_index,
                    detection_score=sample.detection_score,
                    blur_score=sample.blur_score,
                    quality_score=sample.quality_score,
                    bbox=sample.bbox,
                    embedding=sample.embedding.tolist(),
                )
            )

        db.commit()
        svc_timer.record("db_ms", t)
        svc_timer.log(
            "face_service.enroll",
            external_id=external_id,
            images=len(image_payloads),
        )
        return EnrollmentResult(
            external_id=person.external_id,
            name=person.name,
            created=created,
            submitted_image_count=len(image_payloads),
            stored_sample_count=len(kept_samples),
            dropped_sample_count=dropped_sample_count,
            samples=kept_samples,
        )

    def recognize(
        self,
        db: Session,
        image_payload: bytes,
        top_k: int | None = None,
    ) -> RecognitionResult:
        requested_top_k = top_k or self.settings.recognition_top_k_default
        limited_top_k = min(
            max(1, requested_top_k),
            self.settings.recognition_top_k_max,
        )

        query_sample = self.face_engine.extract_sample(image_payload, sample_index=1)
        distance = Person.embedding.cosine_distance(query_sample.embedding.tolist()).label(
            "distance"
        )
        svc_timer = PipelineTimer(self.settings.pipeline_timing)
        t = svc_timer.start()
        rows = db.execute(
            select(Person, distance).order_by(distance).limit(limited_top_k)
        ).all()
        svc_timer.record("pg_vector_search_ms", t)
        svc_timer.log(
            "face_service.recognize",
            top_k=limited_top_k,
        )

        candidates = [
            RecognitionCandidate(
                external_id=person.external_id,
                name=person.name,
                distance=float(person_distance),
                similarity=max(0.0, 1.0 - float(person_distance)),
                sample_count=person.sample_count,
            )
            for person, person_distance in rows
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
