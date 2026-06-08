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
    """Subset de settings requerido para crear embeddings.

    Atributos:
        embedding_provider: Provider seleccionado para embeddings.
        openai_api_key: API key usada por OpenAI.
        openai_embedding_model: Modelo de embeddings de OpenAI.
        cohere_api_key: API key usada por Cohere.
        cohere_embedding_model: Modelo de embeddings de Cohere.
        cohere_embedding_input_type: Tipo de input enviado a Cohere.
        gemini_api_key: API key usada por Gemini.
        gemini_embedding_model: Modelo de embeddings de Gemini.
    """

    embedding_provider: str
    openai_api_key: str | None
    openai_embedding_model: str
    cohere_api_key: str | None
    cohere_embedding_model: str
    cohere_embedding_input_type: str
    gemini_api_key: str | None
    gemini_embedding_model: str


def create_embedding_model(settings: EmbeddingModelSettings) -> EmbeddingModelPort:
    """Crea el adaptador de embeddings según el provider configurado.

    Args:
        settings: Configuración con provider, modelos y API keys.

    Returns:
        Adapter que implementa ``EmbeddingModelPort``.

    Raises:
        ValueError: Si ``settings.embedding_provider`` no está soportado o falta
            la API key requerida.
    """
    provider = _normalized_provider(settings.embedding_provider)

    if provider == "openai":
        return OpenAIEmbeddingModel(
            api_key=_required_key(settings.openai_api_key, "OPENAI_API_KEY"),
            model=settings.openai_embedding_model,
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
    """Valida que una API key requerida esté configurada.

    Args:
        value: Valor leído desde settings.
        env_name: Nombre de la variable de entorno usada para el error.

    Returns:
        API key no vacía.

    Raises:
        ValueError: Si ``value`` es ``None`` o queda vacío.
    """
    if not value or not value.strip():
        raise ValueError(f"{env_name} must be configured")

    return value


def _normalized_provider(provider: str) -> str:
    """Normaliza provider, tolerando comillas externas de env vars."""
    return provider.strip().strip("\"'").casefold()
