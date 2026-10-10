"""Optional integration test that calls the REAL embedding and LLM APIs (costs a little).

Skipped by default. Run with:  RUN_INTEGRATION=1 OPENAI_API_KEY=sk-... pytest -m integration
"""

import os

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import app
from tests.conftest import make_pdf

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.getenv("RUN_INTEGRATION"), reason="set RUN_INTEGRATION=1 to run"),
]


def test_live_round_trip(tmp_path):
    settings = Settings(
        chroma_path=str(tmp_path / "chroma"), upload_dir=str(tmp_path / "uploads")
    )
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        client = TestClient(app)
        pdf = make_pdf(["The project codename is Falcon. It launches in March 2027."])
        up = client.post("/documents/upload", files={"file": ("t.pdf", pdf, "application/pdf")})
        assert up.status_code == 201
        doc_id = up.json()["document_id"]
        res = client.post(f"/documents/{doc_id}/ask", json={"question": "What is the codename?"})
        assert res.status_code == 200
        assert "Falcon" in res.json()["answer"]
    finally:
        app.dependency_overrides.clear()
