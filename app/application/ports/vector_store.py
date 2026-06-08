from typing import Protocol, Sequence

from app.domain.entities.document_chunk import DocumentChunk
from app.domain.entities.retrieval import RetrievedChunk


class VectorStorePort(Protocol):
    """Contrato de almacenamiento y búsqueda vectorial de chunks."""

    async def add_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Persiste chunks con sus embeddings asociados.

        Args:
            chunks: Chunks de documento que se van a indexar.
            embeddings: Embeddings correspondientes a ``chunks`` en el mismo
                orden.
        """
        raise NotImplementedError

    async def search(
        self,
        query_embedding: Sequence[float],
        limit: int,
    ) -> list[RetrievedChunk]:
        """Busca los chunks más similares a un embedding de consulta.

        Args:
            query_embedding: Vector de la pregunta o query de retrieval.
            limit: Cantidad máxima de resultados a devolver.

        Returns:
            Lista ordenada de chunks recuperados con score de similitud.
        """
        raise NotImplementedError
