"""Application configuration management using Pydantic Settings v2.

Loads environment variables from `.env` with fallback defaults for local
zero-dependency development.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Core Application Settings
    APP_NAME: str = Field(default="Mini Knowledge Assistant", description="Application Title")
    APP_VERSION: str = Field(default="0.1.0", description="API Version")
    APP_ENV: Literal["development", "production", "test"] = Field(
        default="development", description="Runtime environment"
    )
    DEBUG: bool = Field(default=True, description="Debug mode")
    API_HOST: str = Field(default="0.0.0.0", alias="HOST", description="API Host")
    API_PORT: int = Field(default=8000, alias="PORT", description="API Port")
    CORS_ORIGINS: list[str] = Field(
        default=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
        ],
        description="Allowed CORS origins",
    )

    # Database Configuration
    DATABASE_URL: str = Field(
        default=f"sqlite+aiosqlite:///{DEFAULT_DATA_DIR}/knowledge_base.db",
        description="SQLAlchemy async connection string",
    )

    # Provider Selection (Standardized)
    EMBEDDING_PROVIDER: Literal["gemini", "mock"] = Field(
        default="gemini",
        description="Standardized embedding provider ('gemini' or 'mock')",
    )
    LLM_PROVIDER: Literal["gemini", "groq", "mock"] = Field(
        default="gemini",
        description="Standardized LLM generation provider ('gemini', 'groq', or 'mock')",
    )
    VECTOR_STORE_PROVIDER: Literal["chroma"] = Field(
        default="chroma",
        description="Vector store backend ('chroma')",
    )
    CHROMA_PERSIST_DIR: str = Field(
        default=str(DEFAULT_DATA_DIR / "chroma_db"),
        description="Persistent directory for ChromaDB storage",
    )
    UPLOAD_DIR: str = Field(
        default=str(DEFAULT_DATA_DIR / "uploads"),
        description="Root directory for storing session uploads",
    )
    MAX_FILES_PER_UPLOAD: int = Field(
        default=5,
        description="Maximum number of files allowed in a single multi-file upload",
    )

    # Google Gemini Settings
    GEMINI_API_KEY: str | None = Field(default=None, description="Google Gemini API Key")
    GEMINI_EMBEDDING_MODEL: str = Field(
        default="models/gemini-embedding-2", description="Gemini embedding model"
    )
    GEMINI_MODEL: str = Field(
        default="gemini-flash-lite-latest",
        alias="GEMINI_LLM_MODEL",
        description="Gemini text model",
    )

    # Groq Settings
    GROQ_API_KEY: str | None = Field(default=None, description="Groq API Key")
    GROQ_MODEL: str = Field(
        default="openai/gpt-oss-120b",
        alias="GROQ_LLM_MODEL",
        description="Groq completion model",
    )
    GROQ_RERANKER_MODEL: str = Field(
        default="openai/gpt-oss-120b",
        description="Groq model used for fast LLM precision reranking",
    )
    GROQ_BASE_URL: str = Field(default="https://api.groq.com", description="Groq API root")

    # DeepEval Evaluation Framework (Powered by Groq)
    EVALUATION_MODE: Literal["live", "mock"] = Field(
        default="live",
        validation_alias=AliasChoices("EVALUATION_MODE", "EVAL_MODE"),
        description="DeepEval benchmark execution mode: 'live' (real Groq/Gemini calls) or 'mock' (offline deterministic)",
    )
    DEEPEVAL_MODEL: str = Field(
        default="qwen/qwen3.8-27b",
        validation_alias=AliasChoices("DEEPEVAL_MODEL", "EVAL_MODEL", "EVALUATION_MODEL_NAME"),
        description="Unified model identifier used for both extractor synthesis and DeepEval benchmark judging",
    )
    OPENAI_API_KEY: str | None = Field(default=None, description="Optional legacy API key")
    OPENAI_BASE_URL: str | None = Field(default=None, description="Optional legacy base URL")

    # RAG Tuning Parameters
    CHUNK_SIZE: int = Field(default=1000, description="Chunk target character length")
    CHUNK_OVERLAP: int = Field(default=150, description="Character overlap between chunks")
    RETRIEVAL_TOP_K: int = Field(
        default=15,
        description="Initial candidate pool size fetched from each retrieval channel (Dense & BM25)",
    )
    FINAL_TOP_N: int = Field(
        default=3,
        description="Final precision chunks fed into synthesis after reranking",
    )
    TOP_K_RESULTS: int = Field(
        default=3,
        description="Legacy alias for FINAL_TOP_N",
    )
    RERANKER_TYPE: str = Field(
        default="groq",
        description="Reranker algorithm: 'groq' (Fast LLM), 'rrf' (Reciprocal Rank Fusion), or 'mock'",
    )
    SIMILARITY_THRESHOLD: float = Field(
        default=0.20, description="Minimum cosine similarity required for relevance"
    )

    # Rate Limiting Guardrails
    RATE_LIMIT_REQUESTS: int = Field(default=20, description="Max allowed requests per window")
    RATE_LIMIT_WINDOW_SECONDS: int = Field(
        default=60, description="Sliding window duration in seconds"
    )

    # File Upload Guardrails
    MAX_FILE_SIZE_BYTES: int = Field(
        default=5 * 1024 * 1024, description="Maximum upload size (5MB)"
    )
    ALLOWED_EXTENSIONS: list[str] = Field(
        default=[".txt", ".md", ".pdf", ".docx", ".doc"],
        description="Permitted file extensions for document upload",
    )

    # LangChain & LangSmith Observability & Tracing
    LANGCHAIN_TRACING_V2: bool = Field(
        default=True,
        description="Enable LangSmith tracing for LangChain components",
    )
    LANGCHAIN_ENDPOINT: str = Field(
        default="https://aws.api.smith.langchain.com",
        description="LangSmith API endpoint",
    )
    LANGCHAIN_API_KEY: str | None = Field(
        default=None,
        description="LangSmith API key",
    )
    LANGCHAIN_PROJECT: str = Field(
        default="Portfolio-Analytics-Agent-V1",
        description="LangSmith project name for traces",
    )

    def model_post_init(self, __context) -> None:
        """Export LangChain & LangSmith settings to os.environ so the LangSmith SDK activates globally."""
        import os

        if self.LANGCHAIN_TRACING_V2:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
        else:
            os.environ["LANGCHAIN_TRACING_V2"] = "false"

        if self.LANGCHAIN_ENDPOINT:
            os.environ["LANGCHAIN_ENDPOINT"] = self.LANGCHAIN_ENDPOINT
        if self.LANGCHAIN_API_KEY:
            os.environ["LANGCHAIN_API_KEY"] = self.LANGCHAIN_API_KEY
        if self.LANGCHAIN_PROJECT:
            os.environ["LANGCHAIN_PROJECT"] = self.LANGCHAIN_PROJECT

    @field_validator("CORS_ORIGINS", "ALLOWED_EXTENSIONS", mode="before")
    @classmethod
    def assemble_string_lists(cls, v):
        if isinstance(v, str):
            import json

            try:
                return json.loads(v)
            except Exception:
                return [i.strip() for i in v.split(",") if i.strip()]
        return v

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def resolve_database_url(cls, v):
        if isinstance(v, str) and v.startswith("sqlite+aiosqlite:///"):
            raw_path = v.replace("sqlite+aiosqlite:///", "")
            p = Path(raw_path)
            if not p.is_absolute():
                resolved = (PROJECT_ROOT / raw_path).resolve()
                return f"sqlite+aiosqlite:///{resolved}"
        return v

    @field_validator("CHROMA_PERSIST_DIR", "UPLOAD_DIR", mode="before")
    @classmethod
    def resolve_relative_data_dirs(cls, v):
        if isinstance(v, str):
            p = Path(v)
            if not p.is_absolute():
                return str((PROJECT_ROOT / v).resolve())
        return v

    @field_validator("EMBEDDING_PROVIDER", mode="before")
    @classmethod
    def normalize_embedding_provider(cls, v):
        if isinstance(v, str):
            clean = v.strip().lower()
            if clean in ("google", "gemini", "google_gemini", "google-gemini"):
                return "gemini"
            if clean in (
                "mock",
                "mock_embeddings",
                "mock_embedding",
                "mock-embeddings",
            ):
                return "mock"
            return clean
        return v

    @field_validator("LLM_PROVIDER", mode="before")
    @classmethod
    def normalize_llm_provider(cls, v):
        if isinstance(v, str):
            clean = v.strip().lower()
            if clean in ("google", "gemini", "google_gemini", "google-gemini"):
                return "gemini"
            if clean in ("groq", "groq_llm"):
                return "groq"
            if clean in ("mock", "mock_llm"):
                return "mock"
            return clean
        return v

    @field_validator("GROQ_MODEL", mode="before")
    @classmethod
    def fallback_groq_model(cls, v):
        if not v or not str(v).strip():
            return "openai/gpt-oss-120b"
        return str(v).strip()

    @field_validator("DEEPEVAL_MODEL", mode="before")
    @classmethod
    def fallback_deepeval_model(cls, v):
        if not v or not str(v).strip():
            return "qwen/qwen3.8-27b"
        return str(v).strip()

    @field_validator("GEMINI_MODEL", mode="before")
    @classmethod
    def fallback_gemini_model(cls, v):
        if not v or not str(v).strip():
            return "gemini-flash-lite-latest"
        return str(v).strip()

    @field_validator("RERANKER_TYPE", mode="before")
    @classmethod
    def normalize_reranker_type(cls, v):
        if isinstance(v, str):
            clean = v.strip().lower()
            if clean in ("groq", "fast_llm", "llm", "groq_llm"):
                return "groq"
            if clean in ("rrf", "reciprocal_rank", "reciprocal"):
                return "rrf"
            if clean in ("mock", "passthrough"):
                return "mock"
            return clean
        return v

    @field_validator("EVALUATION_MODE", mode="before")
    @classmethod
    def normalize_evaluation_mode(cls, v):
        if isinstance(v, str):
            clean = v.strip().lower()
            if clean in ("mock", "offline", "false", "0"):
                return "mock"
            if clean in ("live", "online", "true", "1", "real"):
                return "live"
            return clean
        return "live"

    @property
    def EMBEDDING_MODEL(self) -> str:
        return self.GEMINI_EMBEDDING_MODEL

    @property
    def LLM_MODEL(self) -> str:
        return self.GEMINI_MODEL


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return Settings()
