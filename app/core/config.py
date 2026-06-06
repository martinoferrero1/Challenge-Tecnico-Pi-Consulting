from pydantic import AliasChoices
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="Challenge AI RAG API", alias="APP_NAME")
    app_version: str = Field(default="0.1.0", alias="APP_VERSION")
    api_prefix: str = Field(default="/api", alias="API_PREFIX")
    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
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
