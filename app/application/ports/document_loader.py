from typing import Protocol

from app.domain.entities.document import Document


class DocumentLoaderPort(Protocol):
    """Contrato para cargar documentos desde una fuente externa."""

    async def load(self, source: str) -> Document:
        """Carga y normaliza un documento a partir de una ruta o fuente.

        Args:
            source: Ruta o identificador del documento fuente.

        Returns:
            Documento de dominio con contenido textual y metadata.
        """
        raise NotImplementedError
