import asyncio
from pathlib import Path
from typing import Any

from app.domain.entities.document import Document


class DocxDocumentLoader:
    """Carga documentos DOCX y los convierte en entidades del dominio."""

    def __init__(self, loader_cls: type[Any] | None = None) -> None:
        """Permite inyectar un loader alternativo para tests.

        Args:
            loader_cls: Clase compatible con el loader de LangChain. Si es
                ``None``, se usa ``Docx2txtLoader``.
        """
        self.loader_cls = loader_cls

    async def load(self, source: str) -> Document:
        """Carga un archivo DOCX desde disco.

        Args:
            source: Ruta local del archivo ``.docx``.

        Returns:
            Documento de dominio con contenido extraído y metadata básica.

        Raises:
            FileNotFoundError: Si ``source`` no existe.
            ValueError: Si ``source`` no tiene extensión ``.docx``.
        """
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
        """Ejecuta el loader concreto de LangChain.

        Args:
            path: Ruta local validada del documento DOCX.

        Returns:
            Lista de documentos devuelta por el loader concreto.
        """
        if self.loader_cls:
            loader_cls = self.loader_cls
        else:
            from langchain_community.document_loaders import Docx2txtLoader

            loader_cls = Docx2txtLoader

        return list(loader_cls(str(path)).load())
