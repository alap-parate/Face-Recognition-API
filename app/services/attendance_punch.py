from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from app.core.config import Settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AttendancePunchOutcome:
    """Result of calling the attendance device-punch API (skipped = dispatch returned None)."""

    success: bool
    error: dict[str, Any] | None = None


def _utc_iso_millis_z() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _error_from_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    err = payload.get("error")
    if err is None:
        return None
    if isinstance(err, dict):
        return err
    return {"message": str(err)}


def dispatch_attendance_punch(
    settings: Settings,
    *,
    matched: bool,
    device_identifier: str | None,
    employee_id: str | None,
    face_match_score: float | None,
) -> AttendancePunchOutcome | None:
    """POST /mobile/attendance/device-punch. None = skipped; else outcome with success and optional API error."""
    if not matched:
        return None
    base = (settings.attendance_api_base_url or "").strip()
    if not base:
        return None
    emp = (employee_id or "").strip()
    if not device_identifier or not emp:
        logger.warning(
            "attendance punch skipped: matched but device_identifier or employee_id missing "
            "(enroll with employee_id metadata for punches)"
        )
        return None
    key = (settings.attendance_api_key or "").strip()
    if not key:
        logger.warning("attendance punch skipped: ATTENDANCE_API_KEY not set")
        return None

    url = base.rstrip("/") + "/mobile/attendance/device-punch"
    score = float(face_match_score) if face_match_score is not None else 0.0
    body = {
        "deviceIdentifier": device_identifier,
        "employeeId": emp,
        "punchedAt": _utc_iso_millis_z(),
        "idempotencyKey": str(uuid.uuid4()),
        "metadata": {
            "faceMatchScore": round(score, 4),
            "appVersion": settings.attendance_app_version,
        },
    }
    timeout = settings.attendance_punch_timeout_seconds
    headers = {"x-api-key": key}

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=body, headers=headers)
    except httpx.RequestError as exc:
        logger.warning("attendance punch transport error: %s", exc, extra={"url": url})
        return AttendancePunchOutcome(
            success=False,
            error={
                "code": "TRANSPORT_ERROR",
                "message": str(exc),
                "context": {},
            },
        )

    payload: Any = None
    try:
        payload = response.json()
    except Exception:
        payload = None

    api_err = _error_from_payload(payload)
    if api_err is not None:
        logger.warning(
            "attendance punch API error: %s",
            api_err.get("code", api_err),
            extra={"employee_id": emp},
        )
        return AttendancePunchOutcome(success=False, error=api_err)

    if response.is_error:
        msg = response.reason_phrase or "Request failed"
        logger.warning(
            "attendance punch HTTP %s: %s",
            response.status_code,
            msg,
            extra={"url": url, "employee_id": emp},
        )
        return AttendancePunchOutcome(
            success=False,
            error={
                "code": "HTTP_ERROR",
                "message": msg,
                "context": {"statusCode": response.status_code},
            },
        )

    logger.info("attendance punch ok employee_id=%s", emp)
    return AttendancePunchOutcome(success=True, error=None)
