"""Minimal administrator knowledge review and release API."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.models.knowledge import KnowledgeCandidateCreate
from app.models.user import UserInDB
from app.security import get_current_user
from app.services.knowledge_service import get_knowledge_service
from app.services.knowledge_repository import load_active_release, save_document

router = APIRouter()


def _admin(user: UserInDB) -> None:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Administrator access required")


class ReviewRequest(BaseModel):
    status: Literal["approved", "rejected", "withdrawn"]


@router.get("/candidates")
async def list_candidates(current_user: Annotated[UserInDB, Depends(get_current_user)]):
    _admin(current_user)
    service = get_knowledge_service()
    await service.hydrate()
    return service.list_candidates()


@router.post("/candidates", status_code=201)
async def add_candidate(
    request: KnowledgeCandidateCreate,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    _admin(current_user)
    service = get_knowledge_service()
    await service.hydrate()
    item = service.add_candidate(request)
    await save_document(item)
    return item


@router.post("/candidates/{document_id}/review")
async def review_candidate(
    document_id: str,
    request: ReviewRequest,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    _admin(current_user)
    try:
        service = get_knowledge_service()
        await service.hydrate()
        item = service.review(document_id, request.status)
        await save_document(item)
        return item
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Knowledge document not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/releases", status_code=202)
async def publish_release(current_user: Annotated[UserInDB, Depends(get_current_user)]):
    _admin(current_user)
    service = get_knowledge_service()
    await service.hydrate()
    return await service.publish()


@router.get("/releases/active")
async def active_release(current_user: Annotated[UserInDB, Depends(get_current_user)]):
    _admin(current_user)
    service = get_knowledge_service()
    await service.hydrate()
    return service.active_release() or await load_active_release()
