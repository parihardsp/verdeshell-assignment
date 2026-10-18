"""Pydantic v2 schemas for question answering and RAG grounding."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AskRequest(BaseModel):
    """Schema for POST /ask request body."""

    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(
        ...,
        min_length=3,
        max_length=2_000,
        description="Natural language question to ask the knowledge assistant",
        examples=["How long do I have to request a refund?"],
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session identifier to scope Q&A to documents uploaded in this session",
        examples=["sess_20261016_a1b2c3d4"],
    )
    user_id: str | None = Field(
        default=None,
        description="Optional user identifier to scope Q&A strictly to this user's documents",
        examples=["usr_1001"],
    )

    @field_validator("question")
    @classmethod
    def validate_question_not_empty(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Question cannot be empty or purely whitespace.")
        return clean


class SourceCitation(BaseModel):
    """Schema for grounded source references attached to an answer."""

    model_config = ConfigDict(from_attributes=True)

    document_id: str = Field(..., description="Referenced document identifier")
    title: str = Field(..., description="Title of the source document")
    chunk_index: int = Field(..., description="Index of the cited chunk")
    snippet: str = Field(..., description="Relevant text snippet used for answer synthesis")
    similarity_score: float = Field(
        ..., description="Cosine similarity score indicating retrieval relevance"
    )
    session_id: str | None = Field(default=None, description="Session identifier")
    user_id: str | None = Field(default="default_user", description="User identifier")
    file_name: str | None = Field(default=None, description="Original uploaded filename")
    page_number: int | None = Field(default=None, description="Source page number in document")
    retrieval_channel: str | None = Field(
        default="dense", description="Origin channel: dense, bm25, or hybrid"
    )
    reranker_score: float | None = Field(
        default=None, description="Neural/LLM reranker precision score"
    )


class AskResponse(BaseModel):
    """Schema for POST /ask response containing synthesized answer, citations, and telemetry."""

    model_config = ConfigDict(from_attributes=True)

    answer: str = Field(
        ...,
        description="Grounded response synthesized strictly from retrieved knowledge",
    )
    sources: list[SourceCitation] = Field(
        default_factory=list,
        description="List of source document citations that informed the answer",
    )
    session_id: str | None = Field(
        default=None,
        description="Session identifier if query was session-scoped",
    )
    user_id: str | None = Field(
        default=None,
        description="User identifier if query was user-scoped",
    )
    latency_ms: int | None = Field(
        default=None,
        description="Execution latency in milliseconds",
    )
    confidence_score: float | None = Field(
        default=None,
        description="Grounded confidence score between 0.0 and 1.0",
    )


class GroundedSynthesisOutput(BaseModel):
    """Structured generation schema for zero-hallucination grounded LLM synthesis."""

    model_config = ConfigDict(str_strip_whitespace=True)

    answer: str = Field(
        ...,
        description=(
            "Helpful, structured grounded answer based strictly on verified context: "
            "start with a quick concise summary paragraph (1-2 sentences) directly answering the query, "
            "followed by clear bullet points or numbered steps detailing key facts, conditions, and procedures. "
            "Do not include conversational filler or meta-commentary."
        ),
    )
    has_sufficient_context: bool = Field(
        default=True,
        description="True if context contains enough factual information to answer the question; False if facts are missing.",
    )
    cited_chunk_indices: list[int] = Field(
        default_factory=list,
        description="List of 0-indexed candidate chunk numbers [0, 1, 2] directly used as factual evidence for this answer.",
    )
    confidence_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Grounded confidence score between 0.0 and 1.0 indicating degree of factual support in context.",
    )
