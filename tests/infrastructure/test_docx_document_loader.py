import asyncio
from pathlib import Path
from zipfile import ZipFile

import pytest

from app.infrastructure.document_loaders.docx_document_loader import DocxDocumentLoader


def test_docx_document_loader_extracts_paragraph_text(tmp_path: Path) -> None:
    docx_path = tmp_path / "sample.docx"
    create_docx_fixture(docx_path, ["First paragraph.", "Second paragraph."])

    document = asyncio.run(DocxDocumentLoader().load(str(docx_path)))

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


def create_docx_fixture(path: Path, paragraphs: list[str]) -> None:
    paragraph_xml = "".join(
        f"<w:p><w:r><w:t>{paragraph}</w:t></w:r></w:p>"
        for paragraph in paragraphs
    )

    document_xml = f"""
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
        <w:body>{paragraph_xml}</w:body>
    </w:document>
    """.strip()

    with ZipFile(path, "w") as docx_file:
        docx_file.writestr("word/document.xml", document_xml)
