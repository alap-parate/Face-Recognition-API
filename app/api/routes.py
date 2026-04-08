from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.schemas import (
    EnrollmentResponse,
    EnrollmentSampleResponse,
    HealthResponse,
    RecognitionCandidateResponse,
    RecognitionResponse,
)
from app.services.face_engine import FaceProcessingError
from app.services.face_service import FaceService

router = APIRouter()


def get_face_service(request: Request) -> FaceService:
    return request.app.state.face_service


def read_upload_bytes(upload: UploadFile) -> bytes:
    if upload.content_type and not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported.")
    payload = upload.file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return payload


def _parse_bool_form(raw: str | bool) -> bool:
    if isinstance(raw, bool):
        return raw
    s = str(raw).strip().lower()
    return s in ("1", "true", "yes", "on")


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    face_service = get_face_service(request)
    ready = face_service.qdrant_health()
    return HealthResponse(
        status="ok" if ready else "degraded",
        model_name=face_service.model_name,
        database_ready=ready,
    )


@router.post("/faces/enroll", response_model=EnrollmentResponse)
def enroll_face(
    request: Request,
    org_id: str = Form(...),
    external_id: str = Form(...),
    name: str = Form(...),
    is_active: str = Form("true"),
    images: list[UploadFile] = File(...),
) -> EnrollmentResponse:
    face_service = get_face_service(request)
    try:
        result = face_service.enroll(
            org_id=org_id,
            external_id=external_id,
            name=name,
            is_active=_parse_bool_form(is_active),
            image_payloads=[read_upload_bytes(image) for image in images],
        )
    except FaceProcessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Vector store error while storing the enrollment.",
        ) from exc

    return EnrollmentResponse(
        external_id=result.external_id,
        name=result.name,
        created=result.created,
        submitted_image_count=result.submitted_image_count,
        stored_sample_count=result.stored_sample_count,
        dropped_sample_count=result.dropped_sample_count,
        samples=[
            EnrollmentSampleResponse(
                sample_index=sample.sample_index,
                detection_score=sample.detection_score,
                blur_score=sample.blur_score,
                quality_score=sample.quality_score,
            )
            for sample in result.samples
        ],
    )


@router.post("/faces/recognize", response_model=RecognitionResponse)
def recognize_face(
    request: Request,
    org_id: str = Form(...),
    image: UploadFile = File(...),
    top_k: int = Form(3),
) -> RecognitionResponse:
    face_service = get_face_service(request)
    try:
        result = face_service.recognize(
            org_id=org_id,
            image_payload=read_upload_bytes(image),
            top_k=top_k,
        )
    except FaceProcessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Vector store error while searching for a face match.",
        ) from exc

    candidates = [
        RecognitionCandidateResponse(
            external_id=candidate.external_id,
            name=candidate.name,
            distance=candidate.distance,
            similarity=candidate.similarity,
            sample_count=candidate.sample_count,
        )
        for candidate in result.candidates
    ]
    best_match = candidates[0] if candidates else None
    return RecognitionResponse(
        matched=result.matched,
        threshold=result.threshold,
        query_detection_score=result.query_detection_score,
        query_quality_score=result.query_quality_score,
        best_match=best_match,
        candidates=candidates,
    )
