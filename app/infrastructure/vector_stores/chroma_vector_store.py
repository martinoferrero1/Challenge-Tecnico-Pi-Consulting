from pathlib import Path
from typing import Any, Sequence

from app.domain.entities.document_chunk import DocumentChunk
from app.domain.entities.retrieval import RetrievedChunk


class ChromaVectorStore:
    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        vector_store: Any | None = None,
    ) -> None:
        self.vector_store = vector_store or self._build_vector_store(
            persist_dir=persist_dir,
            collection_name=collection_name,
        )

    async def add_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have the same length")

        self.vector_store._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            metadatas=[self._build_metadata(chunk) for chunk in chunks],
            embeddings=[list(embedding) for embedding in embeddings],
        )

    async def search(
        self,
        query_embedding: Sequence[float],
        limit: int,
    ) -> list[RetrievedChunk]:
        if limit <= 0:
            raise ValueError("Search limit must be greater than zero")

        docs_and_scores = self.vector_store.similarity_search_by_vector_with_relevance_scores(
            embedding=list(query_embedding),
            k=limit,
        )

        return [
            RetrievedChunk(
                chunk=DocumentChunk(
                    id=str(document.metadata.get("chunk_id", "")),
                    content=str(document.page_content),
                    metadata=self._string_metadata(document.metadata),
                ),
                similarity_score=float(score),
            )
            for document, score in docs_and_scores
        ]

    def _build_vector_store(self, persist_dir: str, collection_name: str) -> Any:
        from langchain_chroma import Chroma

        path = Path(persist_dir)
        path.mkdir(parents=True, exist_ok=True)

        return Chroma(
            collection_name=collection_name,
            persist_directory=str(path),
        )

    def _build_metadata(self, chunk: DocumentChunk) -> dict[str, str]:
        return {
            **chunk.metadata,
            "chunk_id": chunk.id,
        }

    def _string_metadata(self, metadata: dict[str, Any]) -> dict[str, str]:
        return {
            str(key): str(value)
            for key, value in metadata.items()
            if value is not None
        }
