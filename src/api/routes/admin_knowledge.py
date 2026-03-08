from fastapi import APIRouter, UploadFile, File, HTTPException
from src.services.knowledge_ingestion_service import ingest_knowledge_file

router = APIRouter(prefix="/admin/knowledge", tags=["Admin Knowledge"])


@router.post("/upload")
async def upload_knowledge(file: UploadFile = File(...)):
    if not file.filename.endswith(".xlsx"):
        raise HTTPException(
            status_code=400, detail="Solo se permiten archivos .xlsx")

    result = await ingest_knowledge_file(file)
    return result
