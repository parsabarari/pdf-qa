"""Step 5 and 7 of the pipeline: persist chunk vectors and search them (ChromaDB)."""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.services.chunker import Chunk

logger = logging.getLogger(__name__)

COLLECTION_NAME = "pdf_chunks"


class VectorStoreError(Exception):
    """Vector database operation failed."""


@dataclass(frozen=True)
class RetrievedChunk:
    page: int
    chunk_index: int
    text: str
    score: float  # cosine similarity, higher = more similar


@dataclass(frozen=True)
class DocumentInfo:
    document_id: str
    filename: str
    num_pages: int
    num_chunks: int
    created_at: str


class VectorStore:
    """One Chroma collection holds chunks from ALL documents.

    Every chunk is one vector and carries `document_id` in its metadata, so each
    query can be restricted with `where={"document_id": ...}`.
    """

    def __init__(self, path: str) -> None:
        try:
            os.makedirs(path, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=path, settings=ChromaSettings(anonymized_telemetry=False)
            )
            # Cosine distance. We always pass embeddings explicitly, so Chroma never
            # needs (or downloads) its own embedding model.
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
                embedding_function=None,
            )
        except Exception as exc:
            logger.error("vector_store_init_failed error_type=%s", type(exc).__name__)
            raise VectorStoreError("Could not open vector store") from exc

    def add_chunks(
        self,
        document_id: str,
        filename: str,
        num_pages: int,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise VectorStoreError("chunks and embeddings must have the same length")
        created_at = datetime.now(timezone.utc).isoformat()
        try:
            self._collection.upsert(
                ids=[f"{document_id}:{c.chunk_index}" for c in chunks],
                embeddings=embeddings,
                documents=[c.text for c in chunks],
                metadatas=[
                    {
                        "document_id": document_id,
                        "filename": filename,
                        "page": c.page,
                        "chunk_index": c.chunk_index,
                        "num_pages": num_pages,
                        "created_at": created_at,
                    }
                    for c in chunks
                ],
            )
        except Exception as exc:
            logger.error("vector_store_add_failed error_type=%s", type(exc).__name__)
            raise VectorStoreError("Could not store chunks") from exc
        logger.info("vector_store_added document_id=%s chunks=%d", document_id, len(chunks))

    def query(self, document_id: str, embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        try:
            result = self._collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                where={"document_id": document_id},  # never search other documents
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error("vector_store_query_failed error_type=%s", type(exc).__name__)
            raise VectorStoreError("Could not search chunks") from exc

        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        return [
            RetrievedChunk(
                page=int(meta["page"]),
                chunk_index=int(meta["chunk_index"]),
                text=text,
                score=round(1.0 - float(dist), 4),
            )
            for text, meta, dist in zip(documents, metadatas, distances)
        ]

    def get_document(self, document_id: str) -> DocumentInfo | None:
        try:
            result = self._collection.get(where={"document_id": document_id}, include=["metadatas"])
        except Exception as exc:
            logger.error("vector_store_get_failed error_type=%s", type(exc).__name__)
            raise VectorStoreError("Could not read document") from exc
        metadatas = result["metadatas"] or []
        if not metadatas:
            return None
        first = metadatas[0]
        return DocumentInfo(
            document_id=document_id,
            filename=str(first["filename"]),
            num_pages=int(first["num_pages"]),
            num_chunks=len(metadatas),
            created_at=str(first["created_at"]),
        )


@lru_cache
def _store_for_path(path: str) -> VectorStore:
    return VectorStore(path)


def get_vector_store_for_path(path: str) -> VectorStore:
    """Return one shared VectorStore per path (Chroma clients should be reused)."""
    return _store_for_path(path)
