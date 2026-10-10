"""Step 4 of the pipeline: text -> embedding vectors (one vector per text)."""

import logging
from functools import lru_cache

from openai import OpenAI

from app.core.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Embedding provider failed or is misconfigured. Message is safe to show clients."""


@lru_cache
def _get_client(api_key: str, base_url: str | None) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url, timeout=60.0, max_retries=2)


def embed_texts(
    texts: list[str],
    settings: Settings,
) -> list[list[float]]:
    """Embed many texts, batching requests. Output order matches input order."""
    if not settings.effective_embedding_api_key:
        raise EmbeddingError("Embedding service is not configured")
    if not texts:
        return []

    client = _get_client(settings.effective_embedding_api_key, settings.effective_embedding_base_url)
    vectors: list[list[float]] = []
    batch_size = settings.embedding_batch_size

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            response = client.embeddings.create(model=settings.embedding_model, input=batch)
        except Exception as exc:
            # Log only the exception type: provider messages can contain request details.
            logger.error("embedding_request_failed error_type=%s", type(exc).__name__)
            raise EmbeddingError("Embedding request failed") from exc
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors.extend(item.embedding for item in ordered)

    if len(vectors) != len(texts):
        raise EmbeddingError("Embedding provider returned an unexpected number of vectors")
    logger.info("embeddings_created count=%d model=%s", len(vectors), settings.embedding_model)
    return vectors


def embed_query(
    question: str,
    settings: Settings,
) -> list[float]:
    """Embed a question. Uses the SAME model as embed_texts, which is required for search."""
    return embed_texts([question], settings)[0]
