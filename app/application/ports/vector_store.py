from typing import Protocol, Sequence

from app.domain.entities.document_chunk import DocumentChunk
from app.domain.entities.retrieval import RetrievedChunk


class VectorStorePort(Protocol):
    async def add_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        raise NotImplementedError

    async def search(
        self,
        query_embedding: Sequence[float],
        limit: int,
    ) -> list[RetrievedChunk]:
        raise NotImplementedError
