from types import SimpleNamespace

import pytest

from app.domain.entities.document import Document
from app.infrastructure.text_splitter.text_chunker import (
    TextChunker,
    TextSplitterConfig,
)


class FakeSplitter:
    def split_documents(self, documents: list[object]) -> list[SimpleNamespace]:
        assert documents[0].page_content == "one two three four"

        return [
            SimpleNamespace(
                page_content="one two",
                metadata={"source": "documento.docx", "start_index": 0},
            ),
            SimpleNamespace(
                page_content="three four",
                metadata={"source": "documento.docx", "start_index": 8},
            ),
        ]


def test_text_chunker_maps_split_documents_to_domain_chunks() -> None:
    document = Document(
        id="doc",
        content="one two three four",
        metadata={"source": "documento.docx"},
    )
    chunker = TextChunker(splitter=FakeSplitter())

    chunks = chunker.chunk(document)

    assert [chunk.content for chunk in chunks] == ["one two", "three four"]
    assert chunks[0].metadata == {
        "source": "documento.docx",
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


def test_text_splitter_config_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        TextSplitterConfig(chunk_size=100, chunk_overlap=100)
