from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, examples=["What is the main conclusion?"])

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class SourceChunk(BaseModel):
    page: int
    chunk_index: int
    text: str
    score: float | None = None  # cosine similarity to the question (higher = closer)


class AskResponse(BaseModel):
    document_id: str
    question: str
    answer: str
    sources: list[SourceChunk]
