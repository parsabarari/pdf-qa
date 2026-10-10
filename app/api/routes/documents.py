import logging
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.deps import get_vector_store
from app.core.config import Settings, get_settings
from app.schemas.documents import DocumentResponse, UploadResponse
from app.services import chunker, embeddings, pdf_parser
from app.services.embeddings import EmbeddingError
from app.services.pdf_parser import NoExtractableTextError, PDFParseError
from app.services.vector_store import VectorStore, VectorStoreError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
def upload_document(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    store: VectorStore = Depends(get_vector_store),
) -> UploadResponse:
    filename = Path(file.filename or "").name  # strip any client-supplied directories
    logger.info("upload_started filename=%s", filename)

    # --- 1. Validate -----------------------------------------------------------------
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Only .pdf files are supported")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = file.file.read(max_bytes + 1)
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file is empty")
    if len(data) > max_bytes:
        raise HTTPException(
            413,
            f"File is larger than {settings.max_upload_mb} MB",
        )
    if b"%PDF-" not in data[:1024]:  # check content, not just the extension
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "File is not a valid PDF")

    # --- 2. Save to disk -------------------------------------------------------------
    document_id = str(uuid.uuid4())
    os.makedirs(settings.upload_dir, exist_ok=True)
    pdf_path = Path(settings.upload_dir) / f"{document_id}.pdf"  # server-generated name
    pdf_path.write_bytes(data)

    try:
        # --- 3. Parse (page by page) -------------------------------------------------
        parsed = pdf_parser.parse_pdf(data)

        # --- 4. Chunk ----------------------------------------------------------------
        chunks = chunker.chunk_pages(parsed.pages, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            raise NoExtractableTextError("PDF contains no extractable text")

        # --- 5. Embed (one vector per chunk) -----------------------------------------
        vectors = embeddings.embed_texts([c.text for c in chunks], settings)

        # --- 6. Store ----------------------------------------------------------------
        store.add_chunks(document_id, filename, parsed.num_pages, chunks, vectors)

    except NoExtractableTextError as exc:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(422, str(exc)) from exc
    except PDFParseError as exc:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except EmbeddingError as exc:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Embedding service failed") from exc
    except VectorStoreError as exc:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Vector store failed") from exc

    logger.info(
        "pdf_processed document_id=%s pages=%d chunks=%d",
        document_id,
        parsed.num_pages,
        len(chunks),
    )
    return UploadResponse(
        document_id=document_id,
        filename=filename,
        num_pages=parsed.num_pages,
        num_chunks=len(chunks),
        status="processed",
    )


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: uuid.UUID, store: VectorStore = Depends(get_vector_store)
) -> DocumentResponse:
    try:
        info = store.get_document(str(document_id))
    except VectorStoreError as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Vector store failed") from exc
    if info is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return DocumentResponse(**info.__dict__)
