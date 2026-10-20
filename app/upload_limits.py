"""Bound preparation multipart traffic before Starlette spools the whole body."""
from fastapi import HTTPException
from starlette.responses import JSONResponse

from app.config import settings


class PreparationUploadLimit:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        if scope["type"] != "http" or not (path.startswith("/api/preparation/cases/") and path.endswith("/resume")):
            return await self.app(scope, receive, send)
        limit = settings.resume_upload_max_bytes + 128 * 1024
        headers = dict(scope.get("headers", []))
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)(scope, receive, send)
        if declared > limit:
            return await JSONResponse({"detail": "Upload request too large"}, status_code=413)(scope, receive, send)
        received = 0

        async def bounded_receive():
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > limit:
                raise HTTPException(413, "Upload request too large")
            return message

        await self.app(scope, bounded_receive, send)
