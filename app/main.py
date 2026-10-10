import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import documents, qa
from app.core.config import get_settings

settings = get_settings()
logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.chroma_path, exist_ok=True)
    if not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY is not set; upload and ask endpoints will fail")
    logger.info("app_started embedding_model=%s llm_model=%s", settings.embedding_model, settings.llm_model)
    yield


app = FastAPI(
    title="PDF Q&A",
    description="Upload a PDF, then ask questions answered only from its content (RAG).",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(documents.router)
app.include_router(qa.router)


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}
