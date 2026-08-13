"""
Upload endpoint.
Accepts a document file, saves it to disk, runs it through the ingestion
pipeline (load -> chunk -> embed -> index), and reports how many chunks
were added to the vector store.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import shutil

from app.config import settings
from app.ingestion.loader_factory import load_document, UnsupportedFileTypeError
from app.vectorstore.chroma_store import add_documents

router = APIRouter()


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    extension = Path(file.filename).suffix.lower()

    upload_path = Path(settings.upload_dir) / file.filename
    upload_path.parent.mkdir(parents=True, exist_ok=True)

    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        documents = load_document(str(upload_path))
    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse file: {e}")

    if not documents:
        raise HTTPException(
            status_code=422,
            detail="No extractable text found in the uploaded file.",
        )

    chunk_count = add_documents(documents)

    return {
        "filename": file.filename,
        "file_type": extension,
        "raw_documents_extracted": len(documents),
        "chunks_indexed": chunk_count,
        "status": "success",
    }