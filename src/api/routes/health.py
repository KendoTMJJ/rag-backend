import urllib.request

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from src.core.config import Config
from src.database.config import SessionLocal
from src.services.embedding_service import _instance as _embedding_instance

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check():
    # embedding_model
    embedding_status = "loaded" if _embedding_instance is not None else "not_loaded"

    # database
    try:
        session = SessionLocal()
        session.execute(text("SELECT 1"))
        session.close()
        database_status = "connected"
    except Exception:
        database_status = "disconnected"

    # ollama
    try:
        url = (Config.OLLAMA_BASE_URL or "").rstrip("/")
        req = urllib.request.urlopen(url, timeout=3)
        req.close()
        ollama_status = "reachable"
    except Exception:
        ollama_status = "unreachable"

    all_ok = (
        embedding_status == "loaded"
        and database_status == "connected"
        and ollama_status == "reachable"
    )

    payload = {
        "status": "ok" if all_ok else "degraded",
        "embedding_model": embedding_status,
        "database": database_status,
        "ollama": ollama_status,
        "version": "1.0.0",
    }
    return JSONResponse(content=payload, status_code=200 if all_ok else 503)
