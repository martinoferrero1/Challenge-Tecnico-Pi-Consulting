from types import SimpleNamespace

import pytest

from app.infrastructure.llms.cohere_llm import CohereLLM
from app.infrastructure.llms.gemini_llm import GeminiLLM
from app.infrastructure.llms.llm_factory import create_llm
from app.infrastructure.llms.openai_llm import OpenAILLM


def test_llm_factory_creates_openai_adapter() -> None:
    model = create_llm(
        build_settings(
            llm_provider="openai",
            openai_api_key="openai-key",
        )
    )

    assert isinstance(model, OpenAILLM)


def test_llm_factory_creates_cohere_adapter() -> None:
    model = create_llm(
        build_settings(
            llm_provider="cohere",
            cohere_api_key="cohere-key",
        )
    )

    assert isinstance(model, CohereLLM)


def test_llm_factory_creates_gemini_adapter() -> None:
    model = create_llm(
        build_settings(
            llm_provider="gemini",
            gemini_api_key="gemini-key",
        )
    )

    assert isinstance(model, GeminiLLM)


def test_llm_factory_rejects_missing_api_key() -> None:
    with pytest.raises(ValueError):
        create_llm(build_settings(llm_provider="openai"))


def test_llm_factory_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError):
        create_llm(
            build_settings(
                llm_provider="unknown",
                openai_api_key="openai-key",
            )
        )


def build_settings(**overrides: object) -> SimpleNamespace:
    defaults = {
        "llm_provider": "openai",
        "openai_api_key": None,
        "openai_llm_model": "gpt-5.5",
        "cohere_api_key": None,
        "cohere_llm_model": "command-a-plus-05-2026",
        "gemini_api_key": None,
        "gemini_llm_model": "gemini-3.5-flash",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)
