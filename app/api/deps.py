from fastapi import Depends

from app.core.config import Settings, get_settings
from app.services.vector_store import VectorStore, get_vector_store_for_path


def get_vector_store(settings: Settings = Depends(get_settings)) -> VectorStore:
    return get_vector_store_for_path(settings.chroma_path)
