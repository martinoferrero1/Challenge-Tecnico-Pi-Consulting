from pydantic import AliasChoices
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración central cargada desde variables de entorno."""

    app_name: str = Field(default="Challenge AI RAG API", alias="APP_NAME")
    app_version: str = Field(default="0.1.0", alias="APP_VERSION")
    api_prefix: str = Field(default="/api", alias="API_PREFIX")
    source_document_path: str = Field(
        default="data/original_document.docx",
        alias="SOURCE_DOCUMENT_PATH",
    )
    source_document_is_default: bool = Field(
        default=True,
        alias="SOURCE_DOCUMENT_IS_DEFAULT",
    )
    chroma_persist_dir: str = Field(default=".chroma", alias="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field(
        default="challenge_ai_documents",
        alias="CHROMA_COLLECTION_NAME",
    )
    text_chunk_size: int = Field(default=800, alias="TEXT_CHUNK_SIZE")
    text_chunk_overlap: int = Field(default=120, alias="TEXT_CHUNK_OVERLAP")
    rag_retrieval_limit: int = Field(default=3, alias="RAG_RETRIEVAL_LIMIT")
    conversation_context_mode: str = Field(
        default="disabled",
        alias="CONVERSATION_CONTEXT_MODE",
    )
    answer_cache_mode: str = Field(
        default="document_context",
        alias="ANSWER_CACHE_MODE",
    )
    conversation_history_limit: int = Field(
        default=10,
        ge=0,
        alias="CONVERSATION_HISTORY_LIMIT",
    )
    language_confidence_threshold: float = Field(
        default=0.5,
        ge=0,
        le=1,
        alias="LANGUAGE_CONFIDENCE_THRESHOLD",
    )
    answer_validation_retries: int = Field(
        default=1,
        ge=0,
        le=1,
        alias="ANSWER_VALIDATION_RETRIES",
    )
    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
    llm_temperature: float = Field(
        default=0.0,
        ge=0,
        le=2,
        alias="LLM_TEMPERATURE",
    )
    judge_llm_provider: str | None = Field(default=None, alias="JUDGE_LLM_PROVIDER")
    judge_llm_model: str | None = Field(default=None, alias="JUDGE_LLM_MODEL")
    embedding_provider: str = Field(default="openai", alias="EMBEDDING_PROVIDER")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_llm_model: str = Field(default="gpt-5.5", alias="OPENAI_LLM_MODEL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        alias="OPENAI_EMBEDDING_MODEL",
    )
    cohere_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("COHERE_API_KEY", "CO_API_KEY"),
    )
    cohere_llm_model: str = Field(
        default="command-a-plus-05-2026",
        alias="COHERE_LLM_MODEL",
    )
    cohere_embedding_model: str = Field(
        default="embed-v4.0",
        alias="COHERE_EMBEDDING_MODEL",
    )
    cohere_embedding_input_type: str = Field(
        default="search_document",
        alias="COHERE_EMBEDDING_INPUT_TYPE",
    )
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_llm_model: str = Field(
        default="gemini-3.5-flash",
        alias="GEMINI_LLM_MODEL",
    )
    gemini_embedding_model: str = Field(
        default="gemini-embedding-2",
        alias="GEMINI_EMBEDDING_MODEL",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


settings = Settings()
