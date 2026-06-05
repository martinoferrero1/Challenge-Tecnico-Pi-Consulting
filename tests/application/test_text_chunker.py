import pytest

from app.application.services.text_chunker import ChunkingConfig, TextChunker
from app.domain.entities.document import Document


def test_text_chunker_splits_text_with_overlap() -> None:
    document = Document(
        id="doc",
        content="one two three four five six",
    )
    chunker = TextChunker(ChunkingConfig(max_words=3, overlap_words=1))

    chunks = chunker.chunk(document)

    assert [chunk.content for chunk in chunks] == [
        "one two three",
        "three four five",
        "five six",
    ]
    assert [chunk.metadata["start_word"] for chunk in chunks] == ["0", "2", "4"]
    assert [chunk.metadata["end_word"] for chunk in chunks] == ["3", "5", "6"]


def test_text_chunker_generates_deterministic_chunk_ids() -> None:
    document = Document(
        id="doc",
        content="one two three four five six",
    )
    chunker = TextChunker(ChunkingConfig(max_words=3, overlap_words=1))

    first_run = chunker.chunk(document)
    second_run = chunker.chunk(document)

    assert [chunk.id for chunk in first_run] == [chunk.id for chunk in second_run]


def test_text_chunker_preserves_document_metadata() -> None:
    document = Document(
        id="doc",
        content="one two three",
        metadata={"source": "documento.docx"},
    )

    chunk = TextChunker(ChunkingConfig(max_words=10, overlap_words=0)).chunk(document)[0]

    assert chunk.metadata["source"] == "documento.docx"
    assert chunk.metadata["document_id"] == "doc"
    assert chunk.metadata["chunk_index"] == "0"


def test_chunking_config_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(max_words=10, overlap_words=10)
