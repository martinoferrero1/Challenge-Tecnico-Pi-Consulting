import asyncio

from app.infrastructure.embedding_models.cohere_embedding_model import (
    CohereEmbeddingModel,
)
from app.infrastructure.embedding_models.gemini_embedding_model import (
    GeminiEmbeddingModel,
)
from app.infrastructure.embedding_models.openai_embedding_model import (
    OpenAIEmbeddingModel,
)


class FakeEmbeddings:
    def __init__(self) -> None:
        self.batch_texts: list[str] = []
        self.query_text: str | None = None

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.batch_texts = texts
        return [[0.1, 0.2], [0.3, 0.4]]

    async def aembed_query(self, text: str) -> list[float]:
        self.query_text = text
        return [0.5, 0.6]


def test_openai_embedding_model_embeds_batch() -> None:
    client = FakeEmbeddings()
    model = OpenAIEmbeddingModel(
        api_key="openai-key",
        model="text-embedding-3-small",
        client=client,
    )

    embeddings = asyncio.run(model.embed_batch(["first", "second"]))

    assert client.batch_texts == ["first", "second"]
    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]


def test_cohere_embedding_model_embeds_batch() -> None:
    client = FakeEmbeddings()
    model = CohereEmbeddingModel(
        api_key="cohere-key",
        model="embed-v4.0",
        input_type="search_document",
        client=client,
    )

    embeddings = asyncio.run(model.embed_batch(["first", "second"]))

    assert client.batch_texts == ["first", "second"]
    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]


def test_gemini_embedding_model_embeds_batch() -> None:
    client = FakeEmbeddings()
    model = GeminiEmbeddingModel(
        api_key="gemini-key",
        model="gemini-embedding-2",
        client=client,
    )

    embeddings = asyncio.run(model.embed_batch(["first", "second"]))

    assert client.batch_texts == ["first", "second"]
    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]


def test_openai_embedding_model_embeds_query_text() -> None:
    client = FakeEmbeddings()
    model = OpenAIEmbeddingModel(
        api_key="openai-key",
        model="text-embedding-3-small",
        client=client,
    )

    embedding = asyncio.run(model.embed_text("question"))

    assert client.query_text == "question"
    assert embedding == [0.5, 0.6]
