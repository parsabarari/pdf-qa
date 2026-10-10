from tests.conftest import make_pdf

GOOD = "Apples are red fruit that grow on trees in orchards around the world."


def upload(client, name, content, ctype="application/pdf"):
    return client.post("/documents/upload", files={"file": (name, content, ctype)})


def test_rejects_non_pdf_extension(client, mock_ai):
    assert upload(client, "notes.txt", b"hello", "text/plain").status_code == 415


def test_rejects_empty_file(client, mock_ai):
    assert upload(client, "empty.pdf", b"").status_code == 400


def test_rejects_fake_pdf_content(client, mock_ai):
    assert upload(client, "fake.pdf", b"this is not a pdf at all").status_code == 415


def test_rejects_corrupt_pdf(client, mock_ai):
    assert upload(client, "bad.pdf", b"%PDF-1.4 garbage garbage").status_code == 400


def test_rejects_pdf_without_text(client, mock_ai):
    response = upload(client, "blank.pdf", make_pdf(["", ""]))
    assert response.status_code == 422
    assert "extractable text" in response.json()["detail"]


def test_upload_success(client, mock_ai):
    pdf = make_pdf([GOOD, "", GOOD.replace("Apples", "Pears")])
    response = upload(client, "fruit.pdf", pdf)
    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "fruit.pdf"
    assert body["num_pages"] == 3  # blank page still counts
    assert body["num_chunks"] >= 2
    assert body["status"] == "processed"

    info = client.get(f"/documents/{body['document_id']}")
    assert info.status_code == 200
    assert info.json()["num_chunks"] == body["num_chunks"]


def test_embedding_failure_returns_502(client, monkeypatch):
    from app.services import embeddings
    from app.services.embeddings import EmbeddingError

    def boom(texts):
        raise EmbeddingError("secret provider detail")

    monkeypatch.setattr(embeddings, "embed_texts", boom)
    response = upload(client, "fruit.pdf", make_pdf([GOOD]))
    assert response.status_code == 502
    assert "secret" not in response.text
