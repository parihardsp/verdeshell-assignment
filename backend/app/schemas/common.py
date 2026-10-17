"""Common utility and health check schemas."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check endpoint response schema."""

    status: str = Field(default="healthy")
    app_name: str
    version: str
    embedding_provider: str
    llm_provider: str
    embedding_provider_mode: str = "gemini"
    llm_provider_mode: str = "gemini"
    vector_store_provider: str = "ChromaVectorStore"
    total_documents: int
    total_chunks: int


class ErrorDetail(BaseModel):
    """RFC 7807 problem detail payload."""

    type: str
    message: str
    status_code: int
    details: dict[str, str] = Field(default_factory=dict)
    path: str
