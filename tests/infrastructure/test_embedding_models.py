import asyncio
from types import SimpleNamespace

from app.infrastructure.embedding_models.cohere_embedding_model import (
    CohereEmbeddingModel,
)
from app.infrastructure.embedding_models.gemini_embedding_model import (
    GeminiEmbeddingModel,
)
from app.infrastructure.embedding_models.openai_embedding_model import (
    OpenAIEmbeddingModel,
)


class FakeOpenAIEmbeddings:
    def __init__(self) -> None:
        self.payload: dict[str, object] | None = None

    async def create(self, model: str, input: list[str]) -> SimpleNamespace:
        self.payload = {"model": model, "input": input}
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[0.3, 0.4]),
                SimpleNamespace(index=0, embedding=[0.1, 0.2]),
            ]
        )


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = FakeOpenAIEmbeddings()


class FakeCohereClient:
    def __init__(self) -> None:
        self.payload: dict[str, object] | None = None

    def embed(
        self,
        inputs: list[dict[str, object]],
        model: str,
        input_type: str,
        embedding_types: list[str],
    ) -> SimpleNamespace:
        self.payload = {
            "inputs": inputs,
            "model": model,
            "input_type": input_type,
            "embedding_types": embedding_types,
        }
        return SimpleNamespace(
            embeddings=SimpleNamespace(float=[[0.1, 0.2], [0.3, 0.4]])
        )


class FakeGeminiModels:
    def __init__(self) -> None:
        self.payload: dict[str, object] | None = None

    def embed_content(self, model: str, contents: list[str]) -> SimpleNamespace:
        self.payload = {"model": model, "contents": contents}
        return SimpleNamespace(
            embeddings=[
                SimpleNamespace(values=[0.1, 0.2]),
                SimpleNamespace(values=[0.3, 0.4]),
            ]
        )


class FakeGeminiClient:
    def __init__(self) -> None:
        self.models = FakeGeminiModels()


def test_openai_embedding_model_embeds_batch() -> None:
    client = FakeOpenAIClient()
    model = OpenAIEmbeddingModel(
        api_key="openai-key",
        model="text-embedding-3-small",
        client=client,
    )

    embeddings = asyncio.run(model.embed_batch(["first", "second"]))

    assert client.embeddings.payload == {
        "model": "text-embedding-3-small",
        "input": ["first", "second"],
    }
    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]


def test_cohere_embedding_model_embeds_batch() -> None:
    client = FakeCohereClient()
    model = CohereEmbeddingModel(
        api_key="cohere-key",
        model="embed-v4.0",
        input_type="search_document",
        client=client,
    )

    embeddings = asyncio.run(model.embed_batch(["first", "second"]))

    assert client.payload == {
        "inputs": [
            {"content": [{"type": "text", "text": "first"}]},
            {"content": [{"type": "text", "text": "second"}]},
        ],
        "model": "embed-v4.0",
        "input_type": "search_document",
        "embedding_types": ["float"],
    }
    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]


def test_gemini_embedding_model_embeds_batch() -> None:
    client = FakeGeminiClient()
    model = GeminiEmbeddingModel(
        api_key="gemini-key",
        model="gemini-embedding-2",
        client=client,
    )

    embeddings = asyncio.run(model.embed_batch(["first", "second"]))

    assert client.models.payload == {
        "model": "gemini-embedding-2",
        "contents": ["first", "second"],
    }
    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
