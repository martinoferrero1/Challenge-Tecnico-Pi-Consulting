from dataclasses import dataclass

from app.application.ports.embedding_model import EmbeddingModelPort
from app.application.ports.vector_store import VectorStorePort
from app.application.use_cases.ingest_document import (
    IngestDocumentUseCase,
    IngestedDocument,
)


@dataclass(frozen=True)
class IndexedDocument:
    """Resultado de indexar chunks con embeddings.

    Atributos:
        ingested_document: Documento ya cargado y dividido en chunks.
        embeddings_count: Cantidad de embeddings generados e indexados.
    """

    ingested_document: IngestedDocument
    embeddings_count: int


class IndexDocumentUseCase:
    """Caso de uso que ingesta, embebbe e indexa un documento."""

    def __init__(
        self,
        ingest_document_use_case: IngestDocumentUseCase,
        embedding_model: EmbeddingModelPort,
        vector_store: VectorStorePort,
    ) -> None:
        """Recibe el pipeline de ingesta, embeddings y vector store.

        Args:
            ingest_document_use_case: Caso de uso que carga y chunkea el
                documento.
            embedding_model: Puerto para generar embeddings de cada chunk.
            vector_store: Puerto para persistir chunks y embeddings.
        """
        self.ingest_document_use_case = ingest_document_use_case
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    async def execute(self, source: str) -> IndexedDocument:
        """Indexa un documento en la base vectorial configurada.

        Args:
            source: Ruta o identificador del documento a indexar.

        Returns:
            Resultado con el documento ingestado y la cantidad de embeddings.

        Raises:
            ValueError: Si el proveedor devuelve una cantidad de embeddings
                distinta de la cantidad de chunks.
        """
        ingested_document = await self.ingest_document_use_case.execute(source)
        chunk_contents = [chunk.content for chunk in ingested_document.chunks]
        embeddings = await self.embedding_model.embed_batch(chunk_contents)

        if len(embeddings) != len(ingested_document.chunks):
            raise ValueError("The number of embeddings must match the number of chunks")

        await self.vector_store.add_chunks(ingested_document.chunks, embeddings)

        return IndexedDocument(
            ingested_document=ingested_document,
            embeddings_count=len(embeddings),
        )
