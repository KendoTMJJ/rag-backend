from fastapi import FastAPI
from src.api.routes.admin_knowledge import router as admin_knowledge_router
from src.api.routes.chat import router as chat_router

app = FastAPI(
    title="RAG Backend USTA",
    version="1.0.0"
)

app.include_router(admin_knowledge_router)
app.include_router(chat_router)
