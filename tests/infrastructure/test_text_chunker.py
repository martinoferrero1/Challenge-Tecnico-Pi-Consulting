from types import SimpleNamespace

import pytest

from app.domain.entities.document import Document
from app.infrastructure.text_splitter.text_chunker import (
    DEFAULT_DOCUMENT_SECTIONS_CHUNKING_STRATEGY,
    TextChunker,
    TextSplitterConfig,
)


class FakeSplitter:
    def split_documents(self, documents: list[object]) -> list[SimpleNamespace]:
        assert documents[0].page_content == "one two three four"

        return [
            SimpleNamespace(
                page_content="one two",
                metadata={"source": "original_document.docx", "start_index": 0},
            ),
            SimpleNamespace(
                page_content="three four",
                metadata={"source": "original_document.docx", "start_index": 8},
            ),
        ]


def test_text_chunker_maps_split_documents_to_domain_chunks() -> None:
    document = Document(
        id="doc",
        content="one two three four",
        metadata={"source": "original_document.docx"},
    )
    chunker = TextChunker(splitter=FakeSplitter())

    chunks = chunker.chunk(document)

    assert [chunk.content for chunk in chunks] == ["one two", "three four"]
    assert chunks[0].metadata == {
        "source": "original_document.docx",
        "start_index": "0",
        "document_id": "doc",
        "chunk_index": "0",
    }
    assert chunks[1].metadata["chunk_index"] == "1"


def test_text_chunker_generates_deterministic_ids() -> None:
    document = Document(id="doc", content="one two three four")
    chunker = TextChunker(splitter=FakeSplitter())

    first_run = chunker.chunk(document)
    second_run = chunker.chunk(document)

    assert [chunk.id for chunk in first_run] == [chunk.id for chunk in second_run]


def test_text_chunker_splits_default_document_by_sections() -> None:
    document = Document(
        id="doc",
        content=(
            "Ficción Espacial: Zara descubre una reliquia.\n\n"
            "Ficción Tecnológica: Alex descubre máquinas sintientes.\n\n"
            "Cuento Corto: Emma comparte un día extra."
        ),
        metadata={"source": "original_document.docx"},
    )
    chunker = TextChunker(
        config=TextSplitterConfig(
            chunk_size=2000,
            chunk_overlap=0,
            strategy=DEFAULT_DOCUMENT_SECTIONS_CHUNKING_STRATEGY,
        )
    )

    chunks = chunker.chunk(document)

    assert [chunk.content for chunk in chunks] == [
        "Ficción Espacial: Zara descubre una reliquia.",
        "Ficción Tecnológica: Alex descubre máquinas sintientes.",
        "Cuento Corto: Emma comparte un día extra.",
    ]
    assert chunks[0].metadata["chunk_strategy"] == (
        DEFAULT_DOCUMENT_SECTIONS_CHUNKING_STRATEGY
    )
    assert chunks[0].metadata["section_title"] == "Ficción Espacial"
    assert chunks[1].metadata["section_index"] == "1"


def test_text_splitter_config_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        TextSplitterConfig(chunk_size=100, chunk_overlap=100)
