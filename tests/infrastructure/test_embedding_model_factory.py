from types import SimpleNamespace

import pytest

from app.infrastructure.embedding_models.cohere_embedding_model import (
    CohereEmbeddingModel,
)
from app.infrastructure.embedding_models.embedding_model_factory import (
    create_embedding_model,
)
from app.infrastructure.embedding_models.gemini_embedding_model import (
    GeminiEmbeddingModel,
)
from app.infrastructure.embedding_models.openai_embedding_model import (
    OpenAIEmbeddingModel,
)


def test_embedding_model_factory_creates_openai_adapter() -> None:
    model = create_embedding_model(
        build_settings(
            embedding_provider="openai",
            openai_api_key="openai-key",
        )
    )

    assert isinstance(model, OpenAIEmbeddingModel)


def test_embedding_model_factory_creates_cohere_adapter() -> None:
    model = create_embedding_model(
        build_settings(
            embedding_provider="cohere",
            cohere_api_key="cohere-key",
        )
    )

    assert isinstance(model, CohereEmbeddingModel)


def test_embedding_model_factory_creates_gemini_adapter() -> None:
    model = create_embedding_model(
        build_settings(
            embedding_provider="gemini",
            gemini_api_key="gemini-key",
        )
    )

    assert isinstance(model, GeminiEmbeddingModel)


def test_embedding_model_factory_rejects_missing_api_key() -> None:
    with pytest.raises(ValueError):
        create_embedding_model(build_settings(embedding_provider="openai"))


def test_embedding_model_factory_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError):
        create_embedding_model(
            build_settings(
                embedding_provider="unknown",
                openai_api_key="openai-key",
            )
        )


def build_settings(**overrides: object) -> SimpleNamespace:
    defaults = {
        "embedding_provider": "openai",
        "openai_api_key": None,
        "openai_embedding_model": "text-embedding-3-small",
        "cohere_api_key": None,
        "cohere_embedding_model": "embed-v4.0",
        "cohere_embedding_input_type": "search_document",
        "gemini_api_key": None,
        "gemini_embedding_model": "gemini-embedding-2",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)
