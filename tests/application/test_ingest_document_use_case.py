import asyncio

import pytest

from app.application.services.text_chunker import ChunkingConfig, TextChunker
from app.application.use_cases.ingest_document import IngestDocumentUseCase
from app.domain.entities.document import Document


class FakeDocumentLoader:
    def __init__(self, document: Document) -> None:
        self.document = document
        self.source: str | None = None

    async def load(self, source: str) -> Document:
        self.source = source
        return self.document


def test_ingest_document_loads_and_chunks_document() -> None:
    loader = FakeDocumentLoader(
        Document(id="doc", content="one two three four five six")
    )
    chunker = TextChunker(ChunkingConfig(max_words=3, overlap_words=1))
    use_case = IngestDocumentUseCase(loader, chunker)

    result = asyncio.run(use_case.execute("documento.docx"))

    assert loader.source == "documento.docx"
    assert result.document.id == "doc"
    assert [chunk.content for chunk in result.chunks] == [
        "one two three",
        "three four five",
        "five six",
    ]


def test_ingest_document_rejects_blank_source() -> None:
    loader = FakeDocumentLoader(Document(id="doc", content="content"))
    use_case = IngestDocumentUseCase(loader, TextChunker())

    with pytest.raises(ValueError):
        asyncio.run(use_case.execute(" "))
