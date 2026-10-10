from pydantic import BaseModel


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    num_pages: int
    num_chunks: int
    status: str


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    num_pages: int
    num_chunks: int
    created_at: str
