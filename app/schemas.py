from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_name: str
    database_ready: bool


class EnrollmentSampleResponse(BaseModel):
    sample_index: int
    detection_score: float
    blur_score: float
    quality_score: float


class EnrollmentResponse(BaseModel):
    external_id: str
    name: str
    created: bool
    submitted_image_count: int
    stored_sample_count: int
    dropped_sample_count: int
    samples: list[EnrollmentSampleResponse]


class RecognitionCandidateResponse(BaseModel):
    external_id: str
    name: str
    distance: float
    similarity: float
    sample_count: int


class RecognitionResponse(BaseModel):
    matched: bool
    threshold: float
    query_detection_score: float
    query_quality_score: float
    best_match: RecognitionCandidateResponse | None
    candidates: list[RecognitionCandidateResponse]

