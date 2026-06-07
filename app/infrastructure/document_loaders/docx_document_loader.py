import asyncio
from pathlib import Path
from typing import Any

from app.domain.entities.document import Document


class DocxDocumentLoader:
    def __init__(self, loader_cls: type[Any] | None = None) -> None:
        self.loader_cls = loader_cls

    async def load(self, source: str) -> Document:
        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {source}")
        if path.suffix.lower() != ".docx":
            raise ValueError("Only .docx documents are supported")

        langchain_documents = await asyncio.to_thread(self._load_documents, path)
        content = "\n".join(
            str(document.page_content).strip()
            for document in langchain_documents
            if str(document.page_content).strip()
        )

        return Document(
            id=path.stem,
            content=content,
            metadata={
                "source": str(path),
                "file_name": path.name,
            },
        )

    def _load_documents(self, path: Path) -> list[Any]:
        if self.loader_cls:
            loader_cls = self.loader_cls
        else:
            from langchain_community.document_loaders import Docx2txtLoader

            loader_cls = Docx2txtLoader

        return list(loader_cls(str(path)).load())
