import asyncio

import pytest

from app.application.use_cases.index_document import IndexDocumentUseCase
from app.application.use_cases.ingest_document import IngestDocumentUseCase
from app.domain.entities.document import Document
from app.domain.entities.document_chunk import DocumentChunk


class FakeDocumentLoader:
    async def load(self, source: str) -> Document:
        return Document(id="doc", content="one two three four five six")


class FakeDocumentChunker:
    def chunk(self, document: Document) -> tuple[DocumentChunk, ...]:
        return (
            DocumentChunk(id="chunk-1", content="one two three"),
            DocumentChunk(id="chunk-2", content="three four five"),
            DocumentChunk(id="chunk-3", content="five six"),
        )


class FakeEmbeddingModel:
    def __init__(self, embeddings: list[list[float]] | None = None) -> None:
        self.embeddings = embeddings
        self.texts: list[str] = []

    async def embed_text(self, text: str) -> list[float]:
        return [float(len(text))]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.texts = list(texts)
        if self.embeddings is not None:
            return self.embeddings

        return [[float(index)] for index, _ in enumerate(texts)]


class FakeVectorStore:
    def __init__(self) -> None:
        self.chunks: tuple[DocumentChunk, ...] = ()
        self.embeddings: list[list[float]] = []

    async def add_chunks(
        self,
        chunks: tuple[DocumentChunk, ...],
        embeddings: list[list[float]],
    ) -> None:
        self.chunks = chunks
        self.embeddings = embeddings


def test_index_document_embeds_and_stores_chunks() -> None:
    ingest_use_case = IngestDocumentUseCase(
        document_loader=FakeDocumentLoader(),
        document_chunker=FakeDocumentChunker(),
    )
    embedding_model = FakeEmbeddingModel()
    vector_store = FakeVectorStore()
    use_case = IndexDocumentUseCase(
        ingest_document_use_case=ingest_use_case,
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    result = asyncio.run(use_case.execute("original_document.docx"))

    assert embedding_model.texts == [
        "one two three",
        "three four five",
        "five six",
    ]
    assert [chunk.content for chunk in vector_store.chunks] == embedding_model.texts
    assert vector_store.embeddings == [[0.0], [1.0], [2.0]]
    assert result.embeddings_count == 3


def test_index_document_rejects_embedding_count_mismatch() -> None:
    ingest_use_case = IngestDocumentUseCase(
        document_loader=FakeDocumentLoader(),
        document_chunker=FakeDocumentChunker(),
    )
    use_case = IndexDocumentUseCase(
        ingest_document_use_case=ingest_use_case,
        embedding_model=FakeEmbeddingModel(embeddings=[[0.0]]),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(ValueError):
        asyncio.run(use_case.execute("original_document.docx"))
