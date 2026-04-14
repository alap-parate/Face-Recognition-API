from __future__ import annotations

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import get_settings

_api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


async def require_x_api_key(
    x_api_key: str | None = Security(_api_key_header),
) -> None:
    """Reject requests when API_KEY is set in settings but header is missing or wrong."""
    expected = (get_settings().api_key or "").strip()
    if not expected:
        return
    if not x_api_key or x_api_key.strip() != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing x-api-key",
        )
