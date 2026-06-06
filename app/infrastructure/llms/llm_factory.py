from typing import Protocol

from app.application.ports.llm import LLMPort
from app.infrastructure.llms.cohere_llm import CohereLLM
from app.infrastructure.llms.gemini_llm import GeminiLLM
from app.infrastructure.llms.openai_llm import OpenAILLM


class LLMSettings(Protocol):
    llm_provider: str
    openai_api_key: str | None
    openai_llm_model: str
    cohere_api_key: str | None
    cohere_llm_model: str
    gemini_api_key: str | None
    gemini_llm_model: str


def create_llm(settings: LLMSettings) -> LLMPort:
    provider = settings.llm_provider.strip().casefold()

    if provider == "openai":
        return OpenAILLM(
            api_key=_required_key(settings.openai_api_key, "OPENAI_API_KEY"),
            model=settings.openai_llm_model,
        )

    if provider == "cohere":
        return CohereLLM(
            api_key=_required_key(settings.cohere_api_key, "COHERE_API_KEY"),
            model=settings.cohere_llm_model,
        )

    if provider == "gemini":
        return GeminiLLM(
            api_key=_required_key(settings.gemini_api_key, "GEMINI_API_KEY"),
            model=settings.gemini_llm_model,
        )

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


def _required_key(value: str | None, env_name: str) -> str:
    if not value or not value.strip():
        raise ValueError(f"{env_name} must be configured")

    return value
