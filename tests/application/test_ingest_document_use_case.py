import asyncio

import pytest

from app.application.use_cases.ingest_document import IngestDocumentUseCase
from app.domain.entities.document import Document
from app.domain.entities.document_chunk import DocumentChunk


class FakeDocumentLoader:
    def __init__(self, document: Document) -> None:
        self.document = document
        self.source: str | None = None

    async def load(self, source: str) -> Document:
        self.source = source
        return self.document


class FakeDocumentChunker:
    def chunk(self, document: Document) -> tuple[DocumentChunk, ...]:
        return (
            DocumentChunk(id="chunk-1", content="one two three"),
            DocumentChunk(id="chunk-2", content="three four five"),
            DocumentChunk(id="chunk-3", content="five six"),
        )


def test_ingest_document_loads_and_chunks_document() -> None:
    loader = FakeDocumentLoader(
        Document(id="doc", content="one two three four five six")
    )
    use_case = IngestDocumentUseCase(loader, FakeDocumentChunker())

    result = asyncio.run(use_case.execute("original_document.docx"))

    assert loader.source == "original_document.docx"
    assert result.document.id == "doc"
    assert [chunk.content for chunk in result.chunks] == [
        "one two three",
        "three four five",
        "five six",
    ]


def test_ingest_document_rejects_blank_source() -> None:
    loader = FakeDocumentLoader(Document(id="doc", content="content"))
    use_case = IngestDocumentUseCase(loader, FakeDocumentChunker())

    with pytest.raises(ValueError):
        asyncio.run(use_case.execute(" "))
