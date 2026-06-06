from typing import Protocol

from app.application.ports.embedding_model import EmbeddingModelPort
from app.infrastructure.embedding_models.cohere_embedding_model import (
    CohereEmbeddingModel,
)
from app.infrastructure.embedding_models.gemini_embedding_model import (
    GeminiEmbeddingModel,
)
from app.infrastructure.embedding_models.openai_embedding_model import (
    OpenAIEmbeddingModel,
)


class EmbeddingModelSettings(Protocol):
    embedding_provider: str
    openai_api_key: str | None
    openai_embedding_model: str
    cohere_api_key: str | None
    cohere_embedding_model: str
    cohere_embedding_input_type: str
    gemini_api_key: str | None
    gemini_embedding_model: str


def create_embedding_model(settings: EmbeddingModelSettings) -> EmbeddingModelPort:
    provider = settings.embedding_provider.strip().casefold()

    if provider == "openai":
        return OpenAIEmbeddingModel(
            api_key=_required_key(settings.openai_api_key, "OPENAI_API_KEY"),
            model=settings.openai_embedding_model, # le paso este parametro extra ya que cohere tiene distintos tipos de inputs para embeddings, pueden ser search_document o search_query
        )

    if provider == "cohere":
        return CohereEmbeddingModel(
            api_key=_required_key(settings.cohere_api_key, "COHERE_API_KEY"),
            model=settings.cohere_embedding_model,
            input_type=settings.cohere_embedding_input_type,
        )

    if provider == "gemini":
        return GeminiEmbeddingModel(
            api_key=_required_key(settings.gemini_api_key, "GEMINI_API_KEY"),
            model=settings.gemini_embedding_model,
        )

    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")


def _required_key(value: str | None, env_name: str) -> str:
    if not value or not value.strip():
        raise ValueError(f"{env_name} must be configured")

    return value
