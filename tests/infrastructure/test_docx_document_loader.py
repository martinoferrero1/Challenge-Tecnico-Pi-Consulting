import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.infrastructure.document_loaders.docx_document_loader import DocxDocumentLoader


class FakeDocxLoader:
    def __init__(self, source: str) -> None:
        self.source = source

    def load(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(page_content="First paragraph."),
            SimpleNamespace(page_content="Second paragraph."),
        ]


def test_docx_document_loader_extracts_paragraph_text(tmp_path: Path) -> None:
    docx_path = tmp_path / "sample.docx"
    docx_path.write_bytes(b"fake docx content")

    document = asyncio.run(DocxDocumentLoader(loader_cls=FakeDocxLoader).load(str(docx_path)))

    assert document.id == "sample"
    assert document.content == "First paragraph.\nSecond paragraph."
    assert document.metadata["file_name"] == "sample.docx"
    assert document.metadata["source"] == str(docx_path)


def test_docx_document_loader_rejects_unsupported_extension(tmp_path: Path) -> None:
    text_path = tmp_path / "sample.txt"
    text_path.write_text("content")

    with pytest.raises(ValueError):
        asyncio.run(DocxDocumentLoader().load(str(text_path)))


def test_docx_document_loader_rejects_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        asyncio.run(DocxDocumentLoader().load("missing.docx"))
