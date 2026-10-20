"""Small admission guard for the supported single-worker preparation service."""
import asyncio
from weakref import WeakValueDictionary
from fastapi import HTTPException, UploadFile

from app.cache.rate_limiter import check_rate_limit
from app.config import settings

_active: dict[str, str] = {}
_admissions: WeakValueDictionary = WeakValueDictionary()


def admission(user_id: str, operation_id: str) -> asyncio.Lock:
    """Serialize only simultaneous duplicate commands while their ID is accepted."""
    key = (user_id, operation_id)
    lock = _admissions.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _admissions[key] = lock
    return lock


def reserve(user_id: str, operation_id: str) -> None:
    # No await between checking and reserving: admission precedes background work.
    if user_id in _active:
        raise HTTPException(429, "A preparation task is already running for this user", headers={"Retry-After": "2"})
    _active[user_id] = operation_id


def release(user_id: str, operation_id: str) -> None:
    if _active.get(user_id) == operation_id:
        _active.pop(user_id, None)


async def enforce(name: str, user_id: str, limit: int, window: int) -> None:
    result = await check_rate_limit(name, f"user:{user_id}", limit, window, require_backend=True)
    if not result.allowed:
        raise HTTPException(429, "Preparation request limit reached or limiter unavailable", headers={"Retry-After": str(result.retry_after_seconds)})


async def read_resume(upload: UploadFile) -> tuple[bytes, str]:
    from pathlib import Path
    suffix = Path(upload.filename or "").suffix.lower()
    mime_types = {".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    if suffix not in mime_types or upload.content_type not in {mime_types.get(suffix), "application/octet-stream", None}:
        raise HTTPException(415, "Allowed resume types: PDF, PNG, JPG, JPEG")
    limit = settings.resume_upload_max_bytes
    if limit <= 0 or (upload.size is not None and upload.size > limit):
        raise HTTPException(413, "Resume exceeds the upload size limit")
    content = await upload.read(limit + 1)
    if len(content) > limit:
        raise HTTPException(413, "Resume exceeds the upload size limit")
    if not content:
        raise HTTPException(400, "Resume file is empty")
    valid = (content.startswith(b"%PDF-") if suffix == ".pdf" else
             content.startswith(b"\x89PNG\r\n\x1a\n") if suffix == ".png" else content.startswith(b"\xff\xd8\xff"))
    if not valid:
        raise HTTPException(415, "Resume content does not match its file type")
    return content, suffix
