import asyncio

import pytest

from app.domain.entities.document_chunk import DocumentChunk
from app.infrastructure.vector_stores.chroma_vector_store import ChromaVectorStore


class FakeCollection:
    def __init__(self) -> None:
        self.upsert_payload: dict[str, object] | None = None
        self.query_payload: dict[str, object] | None = None

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, str]],
        embeddings: list[list[float]],
    ) -> None:
        self.upsert_payload = {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "embeddings": embeddings,
        }

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        include: list[str],
    ) -> dict[str, object]:
        self.query_payload = {
            "query_embeddings": query_embeddings,
            "n_results": n_results,
            "include": include,
        }

        return {
            "ids": [["chunk-1"]],
            "documents": [["chunk content"]],
            "metadatas": [[{"document_id": "doc", "chunk_index": "0"}]],
            "distances": [[0.25]],
        }


def build_store(collection: FakeCollection) -> ChromaVectorStore:
    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store.client = object()
    store.collection = collection
    return store


def test_chroma_vector_store_adds_chunks_to_collection() -> None:
    collection = FakeCollection()
    store = build_store(collection)
    chunks = [
        DocumentChunk(
            id="chunk-1",
            content="chunk content",
            metadata={"document_id": "doc"},
        )
    ]

    asyncio.run(store.add_chunks(chunks, [[0.1, 0.2]]))

    assert collection.upsert_payload == {
        "ids": ["chunk-1"],
        "documents": ["chunk content"],
        "metadatas": [{"document_id": "doc", "chunk_id": "chunk-1"}],
        "embeddings": [[0.1, 0.2]],
    }


def test_chroma_vector_store_rejects_mismatched_chunks_and_embeddings() -> None:
    store = build_store(FakeCollection())

    with pytest.raises(ValueError):
        asyncio.run(
            store.add_chunks(
                [DocumentChunk(id="chunk-1", content="chunk content")],
                [],
            )
        )


def test_chroma_vector_store_search_maps_results_to_retrieved_chunks() -> None:
    collection = FakeCollection()
    store = build_store(collection)

    results = asyncio.run(store.search(query_embedding=[0.1, 0.2], limit=1))

    assert collection.query_payload == {
        "query_embeddings": [[0.1, 0.2]],
        "n_results": 1,
        "include": ["documents", "metadatas", "distances"],
    }
    assert len(results) == 1
    assert results[0].chunk.id == "chunk-1"
    assert results[0].chunk.content == "chunk content"
    assert results[0].chunk.metadata == {"document_id": "doc", "chunk_index": "0"}
    assert results[0].similarity_score == 0.8


def test_chroma_vector_store_rejects_invalid_search_limit() -> None:
    store = build_store(FakeCollection())

    with pytest.raises(ValueError):
        asyncio.run(store.search(query_embedding=[0.1, 0.2], limit=0))
