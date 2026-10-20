"""API router registration."""

from fastapi import APIRouter

from app.api.auth import router as auth_router
from app.api.interview_rest import router as interview_rest_router
from app.api.interview_ws import router as interview_ws_router
from app.api.preparation import router as preparation_router
from app.api.knowledge import router as knowledge_router
from app.api.memory import router as memory_router

api_router = APIRouter()
api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
api_router.include_router(interview_rest_router, prefix="/interview", tags=["interview"])
api_router.include_router(interview_ws_router, prefix="/ws", tags=["websocket"])
api_router.include_router(preparation_router, prefix="/preparation", tags=["preparation"])
api_router.include_router(knowledge_router, prefix="/knowledge", tags=["knowledge"])
api_router.include_router(memory_router, prefix="/memory", tags=["memory"])
