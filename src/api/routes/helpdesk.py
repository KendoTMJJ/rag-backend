import json
import logging
import time
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.database.config import get_db
from src.models.helpdesk import HelpdeskCategory, intent_to_display_label
from src.services.helpdesk_service import HelpdeskService, _RESERVED_INTENTS, _ALWAYS_VALID
from src.services.llm_service import LLMService
from src.core.config import Config as settings

router = APIRouter(prefix="/helpdesk", tags=["Helpdesk"])
logger = logging.getLogger(__name__)

_SALUDO_EXCLUDED = _RESERVED_INTENTS | _ALWAYS_VALID

_llm = LLMService(
    base_url=settings.OLLAMA_BASE_URL,
    model=settings.OLLAMA_MODEL,
    temperature=0.0,
)

_MIME_TYPES: dict[str, str] = {
    "pdf":  "application/pdf",
    "ppt":  "application/vnd.ms-powerpoint",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "doc":  "application/msword",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def _load_active_categories(db: Session) -> list[HelpdeskCategory]:
    return (
        db.query(HelpdeskCategory)
        .filter(HelpdeskCategory.intent.notin_(_SALUDO_EXCLUDED))
        .order_by(HelpdeskCategory.intent)
        .all()
    )


def _build_saludo_msg(db: Session) -> str:
    rows = _load_active_categories(db)
    if rows:
        labels = ", ".join(intent_to_display_label(r.intent) for r in rows)
        return (
            f"¡Hola! Soy el asistente de la mesa de ayuda. "
            f"Puedo orientarte con: {labels}. ¿En qué puedo ayudarte?"
        )
    return "¡Hola! Soy el asistente de la mesa de ayuda. ¿En qué puedo ayudarte?"


# ── Modelos ───────────────────────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    question:      str
    chatSessionId: str


class ClassifyResponse(BaseModel):
    intent:  str
    message: str | None = None


class PublicCategoryOut(BaseModel):
    intent:        str
    display_label: str
    description:   str | None
    has_document:  bool
    document_url:  str | None
    pdf_url:       str | None  # alias para compatibilidad con n8n

    class Config:
        from_attributes = True


def _public_out(r: HelpdeskCategory) -> PublicCategoryOut:
    has_doc = bool(r.document_data)
    relative = f"/helpdesk/document/{r.intent}" if has_doc else None
    base = settings.PUBLIC_BASE_URL
    absolute = f"{base}{relative}" if relative and base else relative
    return PublicCategoryOut(
        intent=r.intent,
        display_label=intent_to_display_label(r.intent),
        description=r.description,
        has_document=has_doc,
        document_url=absolute,
        pdf_url=absolute,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/categories", response_model=list[PublicCategoryOut])
def list_categories_public(db: Session = Depends(get_db)):
    rows = _load_active_categories(db)
    return [_public_out(r) for r in rows]


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
        return ClassifyResponse(intent=intent, message=_build_saludo_msg(db))

    if intent == "despedida":
        logger.info(log_line)
        return ClassifyResponse(
            intent="saludo",
            message="¡De nada! Si tienes otra consulta, aquí estoy para ayudarte. 😊",
        )

    logger.info(log_line)
    return ClassifyResponse(intent=intent)


@router.get("/category/{intent}", response_model=PublicCategoryOut)
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

    return _public_out(row)


@router.get("/document/{intent}")
def download_document(intent: str, db: Session = Depends(get_db)):
    row = (
        db.query(HelpdeskCategory)
        .filter(HelpdeskCategory.intent == intent.lower())
        .first()
    )
    if not row or not row.document_data:
        raise HTTPException(status_code=404, detail="Documento no disponible")

    filename = row.document_filename or f"{intent}.bin"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    content_type = _MIME_TYPES.get(ext, "application/octet-stream")

    logger.info(json.dumps({
        "ts":       datetime.now().isoformat(timespec="milliseconds"),
        "op":       "download_document",
        "intent":   intent,
        "filename": filename,
        "bytes":    len(row.document_data),
    }))

    return Response(
        content=row.document_data,
        media_type=content_type,
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )
