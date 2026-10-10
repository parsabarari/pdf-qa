import re
import zlib

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import app
from app.services import embeddings, qa


def make_pdf(pages: list[str]) -> bytes:
    """Build a small real PDF. An empty string produces a blank page."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((50, 72), text, fontsize=8)
    data = doc.tobytes()
    doc.close()
    return data


def fake_vector(text: str) -> list[float]:
    """Deterministic bag-of-words vector, so tests need no embedding API."""
    vec = [0.01] * 16
    for word in re.findall(r"\w+", text.lower()):
        vec[zlib.crc32(word.encode()) % 16] += 1.0
    return vec


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="test-key",
        chroma_path=str(tmp_path / "chroma"),
        upload_dir=str(tmp_path / "uploads"),
        chunk_size=200,
        chunk_overlap=40,
        top_k=3,
    )


@pytest.fixture
def client(settings):
    app.dependency_overrides[get_settings] = lambda: settings
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def mock_ai(monkeypatch):
    """Replace the embedding and LLM calls. Returns a list that records LLM prompts."""
    llm_calls: list[list[dict]] = []
    monkeypatch.setattr(embeddings, "embed_texts", lambda texts: [fake_vector(t) for t in texts])
    monkeypatch.setattr(embeddings, "embed_query", lambda q: fake_vector(q))

    def fake_llm(messages, settings):
        llm_calls.append(messages)
        return "Mock answer"

    monkeypatch.setattr(qa, "call_llm", fake_llm)
    return llm_calls
