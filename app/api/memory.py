"""User-visible observation dispute and fact-correction endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.memory.models import MemoryType
from app.memory.service import get_memory_service
from app.models.user import UserInDB
from app.security import get_current_user

router = APIRouter()


class FactCorrection(BaseModel):
    content: str = Field(min_length=1)
    replaces_memory_id: str | None = None


@router.get("/observations")
async def list_observations(
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    service = get_memory_service()
    if hasattr(service.store, "alist_memory_items"):
        return await service.store.alist_memory_items(
            current_user.id,
            memory_types=[MemoryType.INTERVIEW_EPISODE.value],
            limit=100,
        )
    return service.store.list_memory_items(
        current_user.id,
        memory_types=[MemoryType.INTERVIEW_EPISODE.value],
        limit=100,
    )


@router.post("/observations/{observation_id}/dispute")
async def dispute_observation(
    observation_id: str,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    try:
        return await get_memory_service().dispute_observation(
            current_user.id, observation_id
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Observation not found") from exc


@router.post("/facts", status_code=201)
async def correct_fact(
    request: FactCorrection,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    service = get_memory_service()
    try:
        return await service.correct_fact(
            current_user.id,
            request.content,
            request.replaces_memory_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Memory to replace not found") from exc
