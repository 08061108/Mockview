import sys
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.7
    # Network timeouts apply to one physical request. Business-operation
    # deadlines below include queueing, validation and bounded repairs.
    model_connect_timeout_seconds: float = 10.0
    model_read_timeout_seconds: float = 90.0
    model_write_timeout_seconds: float = 30.0
    model_pool_timeout_seconds: float = 10.0
    model_timeout_seconds: float = 90.0  # legacy aggregate fallback
    model_max_concurrency: int = 2
    llm_transport_retries: int = 0
    llm_request_attempts: int = 2
    llm_retry_backoff_seconds: float = 0.5
    llm_max_retry_after_seconds: float = 10.0
    structured_repair_attempts: int = 1
    interview_start_timeout_seconds: float = 240.0
    question_generation_timeout_seconds: float = 120.0
    answer_processing_timeout_seconds: float = 240.0
    interview_start_max_attempts: int = 2
    question_generation_max_attempts: int = 2
    # Per-attempt physical-request budgets. Explicit user retries create a new
    # operation and budget; reconnects never do.
    interview_start_model_request_budget: int = 4
    question_generation_model_request_budget: int = 4
    answer_processing_model_request_budget: int = 12
    question_prefetch_enabled: bool = True
    question_prefetch_global_limit: int = 1
    llm_simple_enable_thinking: bool | None = False
    llm_complex_enable_thinking: bool | None = True
    llm_complex_thinking_budget: int | None = 2048
    llm_max_output_tokens: int = 4096
    model_force_ipv4: bool = sys.platform == "win32"
    model_trust_env: bool = False
    # Optional model-only proxy. Prefer this over exposing all outbound traffic
    # to ambient HTTP_PROXY/HTTPS_PROXY. Docker should use host.docker.internal.
    model_proxy_url: str = ""

    # Vision (multi-modal) — falls back to llm_* if not set
    vision_api_key: str = ""
    vision_base_url: str = ""
    vision_model: str = "qwen-vl-ocr"

    # Embedding
    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    embedding_query_batch_size: int = 10
    embedding_query_max_concurrency: int = 2
    rag_query_strategy: Literal[
        "legacy", "target_expansion_v1", "target_expansion_v2"
    ] = "legacy"
    rag_query_expansion_limit: int = 2
    rag_ranking_strategy: Literal["legacy", "bounded", "normalized"] = "legacy"
    rag_query_strategy_version: str = "deterministic-v1"

    # Bounded public-knowledge correction. Disabled unless explicitly enabled.
    public_search_enabled: bool = False
    tavily_api_key: str = ""
    search_timeout_seconds: float = 20.0
    extract_timeout_seconds: float = 30.0
    public_search_max_results: int = 3
    public_extract_max_urls: int = 3
    public_snapshot_max_chars: int = 6000
    public_snapshot_total_max_chars: int = 12000
    public_transport_retries: int = 2
    assessment_max_decisions: int = 8
    assessment_model_request_budget: int = 4
    assessment_search_budget: int = 6
    assessment_extract_budget: int = 12
    assessment_semantic_budget: int = 8
    assessment_semantic_enabled: bool = False
    assessment_autonomous_loop_enabled: bool = False
    question_semantic_validation_enabled: bool = False
    # Shadow validation is observational only: it must never alter questions,
    # citations, coverage or the interview state machine.
    question_semantic_shadow_enabled: bool = False

    # RAG and memory persistence
    chroma_persist_dir: str = "./chroma_data"
    memory_db_path: str = "./memory_data/memory.db"
    memory_store_backend: str = "sqlite"
    memory_vector_backend: str = "chroma"
    upload_dir: str = "./uploads"
    export_dir: str = "./exports"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_allowed_origins: str = "*"
    auth_secret_key: str = ""
    auth_token_expire_minutes: int = 60 * 24 * 7
    ws_auth_timeout_seconds: float = 5.0
    resume_upload_max_bytes: int = 10 * 1024 * 1024
    preparation_timeout_seconds: float = 240.0
    resume_parse_timeout_seconds: float = 60.0
    resume_model_request_budget: int = 2

    # Redis runtime enhancement layer. Disabled by default so local tests and
    # demos can run without a Redis server.
    redis_enabled: bool = False
    redis_url: str = "redis://localhost:6379/0"
    redis_socket_timeout_seconds: float = 3.0
    rate_limit_start_per_minute: int = 3
    rate_limit_answer_per_minute: int = 20
    rate_limit_resume_per_hour: int = 10
    rate_limit_vision_per_day: int = 5
    redis_lock_ttl_ms: int = 60000
    session_cache_ttl_seconds: int = 86400
    report_cache_ttl_seconds: int = 604800
    ws_presence_ttl_seconds: int = 90

    # LangGraph checkpointer. Keep "memory" for local development and tests;
    # use "postgres" in Docker/production so graph state survives restarts.
    checkpointer_backend: str = "memory"
    postgres_url: str = ""
    postgres_host: str = "localhost"
    postgres_host_port: int = 5432
    postgres_db: str = "mockview"
    postgres_user: str = "mockview"
    postgres_password: str = ""

    def model_post_init(self, __context) -> None:
        # Explicit legacy URLs still win. Local examples derive one consistent
        # connection string from the same credentials used by Compose.
        if not self.postgres_url:
            from urllib.parse import quote
            self.postgres_url = (
                f"postgresql://{quote(self.postgres_user, safe='')}:"
                f"{quote(self.postgres_password, safe='')}@{self.postgres_host}:"
                f"{self.postgres_host_port}/{quote(self.postgres_db, safe='')}"
            )

    # Business session metadata persistence. The graph checkpoint alone stores
    # LangGraph state, while this store preserves session metadata such as JD,
    # mode, user_id, resume parse result, and graph_started.
    session_store_backend: str = "memory"
    postgres_pool_min_size: int = 1
    postgres_pool_max_size: int = 5

    @property
    def cors_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.cors_allowed_origins.split(",")
            if origin.strip()
        ] or ["*"]

    @property
    def effective_embedding_api_key(self) -> str:
        return self.embedding_api_key or self.llm_api_key

    @property
    def effective_embedding_base_url(self) -> str:
        return self.embedding_base_url or self.llm_base_url

    @property
    def effective_vision_api_key(self) -> str:
        return self.vision_api_key or self.llm_api_key

    @property
    def effective_vision_base_url(self) -> str:
        return self.vision_base_url or self.llm_base_url

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    def resolve_project_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.project_root / path

    @property
    def memory_db_file(self) -> Path:
        return self.resolve_project_path(self.memory_db_path)

    @property
    def upload_root(self) -> Path:
        return self.resolve_project_path(self.upload_dir)

    @property
    def export_root(self) -> Path:
        return self.resolve_project_path(self.export_dir)


settings = Settings()
