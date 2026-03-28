import json
import logging
import time
from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel
from src.services.rag_pipeline import RAGPipeline

router = APIRouter(prefix="/chat", tags=["Chat"])

rag = RAGPipeline()
logger = logging.getLogger(__name__)


class QuestionRequest(BaseModel):
    question: str
    chatSessionId: str


@router.post("")
def chat(request: QuestionRequest):
    t0 = time.monotonic()
    result = rag.ask(request.question, chat_session_id=request.chatSessionId)
    ms = int((time.monotonic() - t0) * 1000)

    route = result.get("route", "")
    data = result.get("data") or {}

    entry = {
        "ts": datetime.now().isoformat(timespec="milliseconds"),
        "session": request.chatSessionId[:12],
        "q": request.question[:60],
        "route": route,
        "ms": ms,
        "resolved": data.get("resolved"),
        "source": data.get("source"),
        "snies": data.get("snies"),
    }

    log_line = json.dumps(entry, ensure_ascii=False)

    if route in ("LOW_CONFIDENCE", "NOT_FOUND"):
        logger.warning(log_line)
    elif route == "LLM_ERROR":
        logger.error(log_line)
    else:
        logger.info(log_line)

    return result
