import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

logging.basicConfig(level=logging.INFO, format="%(message)s")

from src.api.routes.admin_knowledge import router as admin_knowledge_router
from src.api.routes.chat import router as chat_router
from src.api.routes.health import router as health_router
from src.services.embedding_service import get_embedding_model


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_embedding_model()
    yield


app = FastAPI(
    title="RAG Backend USTA",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(admin_knowledge_router)
app.include_router(chat_router)
app.include_router(health_router)
