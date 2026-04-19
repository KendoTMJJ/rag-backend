import json
import logging
import time
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.database.config import get_db
from src.models.helpdesk import HelpdeskCategory
from src.services.helpdesk_service import HelpdeskService
from src.services.llm_service import LLMService
from src.core.config import Config as settings

router = APIRouter(prefix="/helpdesk", tags=["Helpdesk"])
logger = logging.getLogger(__name__)

_SALUDO_MSG = (
    "¡Hola! Soy el asistente de la mesa de ayuda. "
    "Puedes preguntarme sobre inscripciones, fechas importantes, "
    "costos, requisitos de admisión o trámites académicos. ¿En qué puedo ayudarte?"
)

# Instancia única igual que rag = RAGPipeline() en chat.py
_llm = LLMService(
    base_url=settings.OLLAMA_BASE_URL,
    model=settings.OLLAMA_MODEL,
    temperature=0.0,
)


# ── Modelos ───────────────────────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    question:      str
    chatSessionId: str


class ClassifyResponse(BaseModel):
    intent:  str
    message: str | None = None


class HelpdeskCategoryOut(BaseModel):
    intent:      str
    label:       str
    pdf_url:     str | None
    description: str | None

    class Config:
        from_attributes = True


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/classify", response_model=ClassifyResponse)
def classify_intent(
    body: ClassifyRequest,
    db:   Session = Depends(get_db),
):
    t0 = time.monotonic()
    service = HelpdeskService(db=db, llm=_llm)
    intent = service.classify_intent(body.question)
    ms = int((time.monotonic() - t0) * 1000)

    log_line = json.dumps({
        "ts":      datetime.now().isoformat(timespec="milliseconds"),
        "session": body.chatSessionId[:12],
        "q":       body.question[:60],
        "intent":  intent,
        "ms":      ms,
    }, ensure_ascii=False)

    if intent == "desconocida":
        logger.warning(log_line)
        message = service.generate_orientation_message(body.question)
        return ClassifyResponse(intent=intent, message=message)

    if intent == "saludo":
        logger.info(log_line)
        return ClassifyResponse(intent=intent, message=_SALUDO_MSG)

    logger.info(log_line)
    return ClassifyResponse(intent=intent)


@router.get("/category/{intent}", response_model=HelpdeskCategoryOut)
def get_category(
    intent: str,
    db:     Session = Depends(get_db),
):
    t0 = time.monotonic()
    row = (
        db.query(HelpdeskCategory)
        .filter(HelpdeskCategory.intent == intent.lower())
        .first()
    )
    ms = int((time.monotonic() - t0) * 1000)

    if not row:
        logger.warning(json.dumps({
            "ts":     datetime.now().isoformat(timespec="milliseconds"),
            "op":     "get_category",
            "intent": intent,
            "found":  False,
            "ms":     ms,
        }, ensure_ascii=False))
        raise HTTPException(status_code=404, detail="Categoría no encontrada")

    logger.info(json.dumps({
        "ts":     datetime.now().isoformat(timespec="milliseconds"),
        "op":     "get_category",
        "intent": intent,
        "found":  True,
        "ms":     ms,
    }, ensure_ascii=False))

    return row
