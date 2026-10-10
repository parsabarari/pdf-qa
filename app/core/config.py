"""Application settings, loaded from environment variables (and an optional .env file)."""

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Provider credentials (any OpenAI-compatible API) ---
    openai_api_key: str = ""
    openai_base_url: str | None = None
    # Optional: use a different provider/key for embeddings. Falls back to the values above.
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None

    # --- Models ---
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # --- Storage ---
    chroma_path: str = "data/chroma"
    upload_dir: str = "data/uploads"

    # --- RAG parameters ---
    chunk_size: int = Field(default=1000, gt=0)  # characters
    chunk_overlap: int = Field(default=200, ge=0)  # characters
    top_k: int = Field(default=4, gt=0)

    # --- Misc ---
    max_upload_mb: int = Field(default=20, gt=0)
    embedding_batch_size: int = Field(default=64, gt=0)
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _check_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    # Empty strings in .env (e.g. `OPENAI_BASE_URL=`) are treated as "not set".
    @property
    def llm_base_url(self) -> str | None:
        return self.openai_base_url or None

    @property
    def effective_embedding_api_key(self) -> str:
        return self.embedding_api_key or self.openai_api_key

    @property
    def effective_embedding_base_url(self) -> str | None:
        return self.embedding_base_url or self.openai_base_url or None


@lru_cache
def get_settings() -> Settings:
    return Settings()
