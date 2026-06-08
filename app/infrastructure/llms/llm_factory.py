from typing import Protocol

from app.application.ports.llm import LLMPort
from app.infrastructure.llms.cohere_llm import CohereLLM
from app.infrastructure.llms.gemini_llm import GeminiLLM
from app.infrastructure.llms.openai_llm import OpenAILLM


class LLMSettings(Protocol):
    """Subset de settings requerido para crear LLMs.

    Atributos:
        llm_provider: Provider seleccionado para generación.
        openai_api_key: API key usada por OpenAI.
        openai_llm_model: Modelo de chat de OpenAI.
        cohere_api_key: API key usada por Cohere.
        cohere_llm_model: Modelo de chat de Cohere.
        gemini_api_key: API key usada por Gemini.
        gemini_llm_model: Modelo de chat de Gemini.
    """

    llm_provider: str
    llm_temperature: float
    judge_llm_provider: str | None
    judge_llm_model: str | None
    openai_api_key: str | None
    openai_llm_model: str
    cohere_api_key: str | None
    cohere_llm_model: str
    gemini_api_key: str | None
    gemini_llm_model: str


def create_llm(
    settings: LLMSettings,
    *,
    provider_override: str | None = None,
    model_override: str | None = None,
) -> LLMPort:
    """Crea el adaptador LLM según el provider configurado.

    Args:
        settings: Configuración con provider, modelos y API keys.

    Returns:
        Adapter que implementa ``LLMPort``.

    Raises:
        ValueError: Si ``settings.llm_provider`` no está soportado o falta la
            API key requerida.
    """
    provider = _selected_provider(settings.llm_provider, provider_override)

    if provider == "openai":
        return OpenAILLM(
            api_key=_required_key(settings.openai_api_key, "OPENAI_API_KEY"),
            model=_selected_model(settings.openai_llm_model, model_override),
            temperature=_llm_temperature(settings),
        )

    if provider == "cohere":
        return CohereLLM(
            api_key=_required_key(settings.cohere_api_key, "COHERE_API_KEY"),
            model=_selected_model(settings.cohere_llm_model, model_override),
            temperature=_llm_temperature(settings),
        )

    if provider == "gemini":
        return GeminiLLM(
            api_key=_required_key(settings.gemini_api_key, "GEMINI_API_KEY"),
            model=_selected_model(settings.gemini_llm_model, model_override),
            temperature=_llm_temperature(settings),
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")


def _required_key(value: str | None, env_name: str) -> str:
    """Valida que una API key requerida esté configurada.

    Args:
        value: Valor leído desde settings.
        env_name: Nombre de la variable de entorno usada en el error.

    Returns:
        API key no vacía.

    Raises:
        ValueError: Si ``value`` es ``None`` o queda vacío.
    """
    if not value or not value.strip():
        raise ValueError(f"{env_name} must be configured")

    return value


def _selected_provider(default_provider: str, provider_override: str | None) -> str:
    """Devuelve el provider override si existe, o el provider default."""
    if provider_override and provider_override.strip():
        return _normalized_provider(provider_override)

    return _normalized_provider(default_provider)


def _normalized_provider(provider: str) -> str:
    """Normaliza provider, tolerando comillas externas de env vars."""
    return provider.strip().strip("\"'").casefold()


def _selected_model(default_model: str, model_override: str | None) -> str:
    """Devuelve el modelo override si existe, o el modelo default."""
    if model_override and model_override.strip():
        return model_override.strip()

    return default_model


def _llm_temperature(settings: LLMSettings) -> float:
    """Devuelve la temperatura configurada, con default compatible con tests."""
    return float(getattr(settings, "llm_temperature", 0.0))
