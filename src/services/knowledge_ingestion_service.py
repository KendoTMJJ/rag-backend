from pathlib import Path
from fastapi import UploadFile
import shutil
import uuid

from src.extractors.program_excel_parser import ProgramExcelParser

TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)


async def ingest_knowledge_file(file: UploadFile) -> dict:
    unique_name = f"{uuid.uuid4()}_{file.filename}"
    temp_path = TEMP_DIR / unique_name

    with temp_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    parser = ProgramExcelParser(str(temp_path))
    parser.load_and_sync()

    return {
        "message": "Archivo procesado correctamente",
        "filename": file.filename
    }
