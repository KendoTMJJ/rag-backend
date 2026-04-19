import json
import logging
import time
from datetime import datetime

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.orm import Session

from src.core.config import Config
from src.database.config import get_db
from src.models.helpdesk import HelpdeskCategory
from src.api.schemas.helpdesk import (
    HelpdeskCategoryOut,
    HelpdeskCategoryCreate,
    HelpdeskCategoryUpdate,
)

router = APIRouter(prefix="/helpdesk/admin", tags=["Helpdesk Admin"])
logger = logging.getLogger(__name__)


# ── Auth ──────────────────────────────────────────────────────────────────────

def verify_internal_key(x_internal_key: str = Header(...)):
    if not Config.RAG_INTERNAL_API_KEY:
        raise HTTPException(
            status_code=500, detail="RAG_INTERNAL_API_KEY no configurada")
    if x_internal_key != Config.RAG_INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="No autorizado")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get(
    "/categories",
    response_model=list[HelpdeskCategoryOut],
    dependencies=[Depends(verify_internal_key)],
)
def list_categories(db: Session = Depends(get_db)):
    t0 = time.monotonic()
    rows = db.query(HelpdeskCategory).order_by(HelpdeskCategory.intent).all()
    ms = int((time.monotonic() - t0) * 1000)
    logger.info(json.dumps({
        "ts":    datetime.now().isoformat(timespec="milliseconds"),
        "op":    "list_categories",
        "count": len(rows),
        "ms":    ms,
    }))
    return rows


@router.get(
    "/categories/{id}",
    response_model=HelpdeskCategoryOut,
    dependencies=[Depends(verify_internal_key)],
)
def get_category(id: int, db: Session = Depends(get_db)):
    row = db.query(HelpdeskCategory).filter(HelpdeskCategory.id == id).first()
    if not row:
        raise HTTPException(status_code=404, detail="No encontrado")
    return row


@router.post(
    "/categories",
    response_model=HelpdeskCategoryOut,
    status_code=201,
    dependencies=[Depends(verify_internal_key)],
)
def create_category(body: HelpdeskCategoryCreate, db: Session = Depends(get_db)):
    existing = (
        db.query(HelpdeskCategory)
        .filter(HelpdeskCategory.intent == body.intent)
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Ya existe una categoría para intent='{body.intent}'",
        )

    row = HelpdeskCategory(
        **{**body.model_dump(), "intent": body.intent.lower()})

    db.add(row)
    db.commit()
    db.refresh(row)

    logger.info(json.dumps({
        "ts":     datetime.now().isoformat(timespec="milliseconds"),
        "op":     "create_category",
        "id":     row.id,
        "intent": row.intent,
    }))
    return row


@router.patch(
    "/categories/{id}",
    response_model=HelpdeskCategoryOut,
    dependencies=[Depends(verify_internal_key)],
)
def update_category(id: int, body: HelpdeskCategoryUpdate, db: Session = Depends(get_db)):
    row = db.query(HelpdeskCategory).filter(HelpdeskCategory.id == id).first()
    if not row:
        raise HTTPException(status_code=404, detail="No encontrado")

    for k, v in body.model_dump(exclude_unset=True).items():
        setattr(row, k, v)

    db.commit()
    db.refresh(row)

    logger.info(json.dumps({
        "ts":     datetime.now().isoformat(timespec="milliseconds"),
        "op":     "update_category",
        "id":     row.id,
        "intent": row.intent,
    }))
    return row


@router.delete(
    "/categories/{id}",
    dependencies=[Depends(verify_internal_key)],
)
def delete_category(id: int, db: Session = Depends(get_db)):
    row = db.query(HelpdeskCategory).filter(HelpdeskCategory.id == id).first()
    if not row:
        raise HTTPException(status_code=404, detail="No encontrado")

    db.delete(row)
    db.commit()

    logger.info(json.dumps({
        "ts": datetime.now().isoformat(timespec="milliseconds"),
        "op": "delete_category",
        "id": id,
    }))
    return {"deleted": id}


@router.get(
    "/intents",
    dependencies=[Depends(verify_internal_key)],
)
def list_intents(db: Session = Depends(get_db)):
    t0 = time.monotonic()
    rows = db.query(HelpdeskCategory).order_by(HelpdeskCategory.intent).all()
    ms = int((time.monotonic() - t0) * 1000)
    result = [{"intent": r.intent, "label": r.label, "count": 1} for r in rows]
    logger.info(json.dumps({
        "ts":    datetime.now().isoformat(timespec="milliseconds"),
        "op":    "list_intents",
        "count": len(result),
        "ms":    ms,
    }))
    return result
