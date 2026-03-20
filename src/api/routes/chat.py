from fastapi import APIRouter
from pydantic import BaseModel
from src.services.rag_pipeline import RAGPipeline

router = APIRouter(prefix="/chat", tags=["Chat"])

rag = RAGPipeline()


class QuestionRequest(BaseModel):
    question: str
    chatSessionId: str


@router.post("")
def chat(request: QuestionRequest):
    return rag.ask(request.question, chat_session_id=request.chatSessionId)
