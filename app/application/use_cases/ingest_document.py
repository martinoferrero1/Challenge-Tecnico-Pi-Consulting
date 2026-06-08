from dataclasses import dataclass

from app.application.ports.document_chunker import DocumentChunkerPort
from app.application.ports.document_loader import DocumentLoaderPort
from app.domain.entities.document import Document
from app.domain.entities.document_chunk import DocumentChunk


@dataclass(frozen=True)
class IngestedDocument:
    """Resultado de cargar y chunquear un documento.

    Atributos:
        document: Documento fuente normalizado.
        chunks: Fragmentos generados desde ``document``.
    """

    document: Document
    chunks: tuple[DocumentChunk, ...]


class IngestDocumentUseCase:
    """Caso de uso que carga un documento y genera sus chunks."""

    def __init__(
        self,
        document_loader: DocumentLoaderPort,
        document_chunker: DocumentChunkerPort,
    ) -> None:
        """Recibe los puertos necesarios para ingestar documentos.

        Args:
            document_loader: Puerto encargado de cargar ``Document`` desde una
                fuente.
            document_chunker: Puerto encargado de partir ``Document`` en
                ``DocumentChunk``.
        """
        self.document_loader = document_loader
        self.document_chunker = document_chunker

    async def execute(self, source: str) -> IngestedDocument:
        """Carga un documento desde la fuente indicada y lo divide en chunks.

        Args:
            source: Ruta o identificador del documento a ingestar.

        Returns:
            Resultado con ``document`` y ``chunks``.

        Raises:
            ValueError: Si ``source`` queda vacío.
        """
        print(f"Loading document from source: {source}")
        if not source.strip():
            raise ValueError("Document source cannot be empty")

        document = await self.document_loader.load(source)
        chunks = self.document_chunker.chunk(document)

        return IngestedDocument(document=document, chunks=chunks)
