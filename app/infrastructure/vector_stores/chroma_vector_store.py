from pathlib import Path
from typing import Any, Sequence

from app.domain.entities.document_chunk import DocumentChunk
from app.domain.entities.retrieval import RetrievedChunk


class ChromaVectorStore:
    def __init__(self, persist_dir: str, collection_name: str) -> None:
        import chromadb

        path = Path(persist_dir)
        path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    async def add_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings): # nuevamente chequeo por seguridad, pero no deberia ocurrir
            raise ValueError("Chunks and embeddings must have the same length")

        self.collection.upsert(
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

        response = self.collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=limit,
            include=["documents", "metadatas", "distances"],
        )

        ids = self._first_result_list(response, "ids")
        documents = self._first_result_list(response, "documents")
        metadatas = self._first_result_list(response, "metadatas")
        distances = self._first_result_list(response, "distances")

        results: list[RetrievedChunk] = []
        for chunk_id, content, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            chunk_metadata = self._string_metadata(metadata or {})
            chunk = DocumentChunk(
                id=str(chunk_id),
                content=str(content),
                metadata=chunk_metadata,
            )
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    similarity_score=self._distance_to_similarity(distance),
                )
            )

        return results

    def _build_metadata(self, chunk: DocumentChunk) -> dict[str, str]:
        return {
            **chunk.metadata,
            "chunk_id": chunk.id,
        }

    def _distance_to_similarity(self, distance: float | int) -> float:
        return 1.0 / (1.0 + float(distance) )

    def _first_result_list(
        self,
        response: dict[str, Any],
        key: str,
    ) -> list[Any]:
        values = response.get(key) or [[]]
        if not values:
            return []

        first_value = values[0]
        return first_value if isinstance(first_value, list) else []

    def _string_metadata(self, metadata: dict[str, Any]) -> dict[str, str]:
        return {
            str(key): str(value)
            for key, value in metadata.items()
            if value is not None
        }
