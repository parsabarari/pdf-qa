import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_vector_store
from app.core.config import Settings, get_settings
from app.schemas.qa import AskRequest, AskResponse, SourceChunk
from app.services import qa
from app.services.embeddings import EmbeddingError
from app.services.qa import LLMError
from app.services.vector_store import VectorStore, VectorStoreError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["question answering"])


@router.post("/{document_id}/ask", response_model=AskResponse)
def ask_document(
    document_id: uuid.UUID,
    body: AskRequest,
    settings: Settings = Depends(get_settings),
    store: VectorStore = Depends(get_vector_store),
) -> AskResponse:
    doc_id = str(document_id)
    try:
        if store.get_document(doc_id) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
        answer, chunks = qa.answer_question(store, doc_id, body.question, settings)
    except EmbeddingError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Embedding service failed") from exc
    except LLMError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "LLM service failed") from exc
    except VectorStoreError as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Vector store failed") from exc

    return AskResponse(
        document_id=doc_id,
        question=body.question,
        answer=answer,
        sources=[
            SourceChunk(page=c.page, chunk_index=c.chunk_index, text=c.text, score=c.score)
            for c in chunks
        ],
    )
