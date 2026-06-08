from typing import Protocol

from app.domain.entities.document import Document
from app.domain.entities.document_chunk import DocumentChunk


class DocumentChunkerPort(Protocol):
    """Contrato para dividir documentos en chunks indexables."""

    def chunk(self, document: Document) -> tuple[DocumentChunk, ...]:
        """Parte un documento en fragmentos listos para embeddings.

        Args:
            document: Documento de dominio ya cargado.

        Returns:
            Tupla de chunks indexables generados desde ``document``.
        """
        raise NotImplementedError
