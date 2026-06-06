from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

from app.domain.entities.document import Document


class DocxDocumentLoader:
    async def load(self, source: str) -> Document:
        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {source}")
        if path.suffix.lower() != ".docx":
            raise ValueError("Only .docx documents are supported")

        content = self._extract_text(path)

        return Document(
            id=path.stem,
            content=content,
            metadata={
                "source": str(path),
                "file_name": path.name,
            },
        )

    def _extract_text(self, path: Path) -> str:
        with ZipFile(path) as docx_file:
            document_xml = docx_file.read("word/document.xml")

        root = ElementTree.fromstring(document_xml)
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs: list[str] = []

        for paragraph in root.findall(".//w:p", namespace):
            text = "".join(
                node.text or ""
                for node in paragraph.findall(".//w:t", namespace)
            ).strip()

            if text:
                paragraphs.append(text)

        return "\n".join(paragraphs)
