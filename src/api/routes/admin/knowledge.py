from fastapi import APIRouter, Depends, Header, HTTPException, UploadFile, File
from src.core.config import Config
from src.services.knowledge_ingestion_service import ingest_knowledge_file

router = APIRouter(prefix="/admin/knowledge", tags=["Admin Knowledge"])


def verify_internal_key(x_internal_key: str = Header(...)):
    if not Config.RAG_INTERNAL_API_KEY:
        raise HTTPException(
            status_code=500, detail="RAG_INTERNAL_API_KEY no configurada")
    if x_internal_key != Config.RAG_INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="No autorizado")


@router.post("/upload", dependencies=[Depends(verify_internal_key)])
async def upload_knowledge(file: UploadFile = File(...)):
    if not file.filename.endswith(".xlsx"):
        raise HTTPException(
            status_code=400, detail="Solo se permiten archivos .xlsx")

    result = await ingest_knowledge_file(file)
    return result
