import asyncio
from types import SimpleNamespace

import pytest

from app.domain.entities.document_chunk import DocumentChunk
from app.infrastructure.vector_stores.chroma_vector_store import ChromaVectorStore


class FakeCollection:
    def __init__(self) -> None:
        self.upsert_payload: dict[str, object] | None = None

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

class FakeChroma:
    def __init__(self) -> None:
        self._collection = FakeCollection()
        self.search_payload: dict[str, object] | None = None

    def similarity_search_by_vector_with_relevance_scores(
        self,
        embedding: list[float],
        k: int,
    ) -> list[tuple[SimpleNamespace, float]]:
        self.search_payload = {"embedding": embedding, "k": k}
        return [
            (
                SimpleNamespace(
                    page_content="chunk content",
                    metadata={
                        "chunk_id": "chunk-1",
                        "document_id": "doc",
                        "chunk_index": "0",
                    },
                ),
                0.8,
            )
        ]


def test_chroma_vector_store_adds_chunks_to_collection() -> None:
    vector_store = FakeChroma()
    store = ChromaVectorStore(
        persist_dir="unused",
        collection_name="unused",
        vector_store=vector_store,
    )
    chunks = [
        DocumentChunk(
            id="chunk-1",
            content="chunk content",
            metadata={"document_id": "doc"},
        )
    ]

    asyncio.run(store.add_chunks(chunks, [[0.1, 0.2]]))

    assert vector_store._collection.upsert_payload == {
        "ids": ["chunk-1"],
        "documents": ["chunk content"],
        "metadatas": [{"document_id": "doc", "chunk_id": "chunk-1"}],
        "embeddings": [[0.1, 0.2]],
    }


def test_chroma_vector_store_rejects_mismatched_chunks_and_embeddings() -> None:
    store = ChromaVectorStore(
        persist_dir="unused",
        collection_name="unused",
        vector_store=FakeChroma(),
    )

    with pytest.raises(ValueError):
        asyncio.run(
            store.add_chunks(
                [DocumentChunk(id="chunk-1", content="chunk content")],
                [],
            )
        )


def test_chroma_vector_store_search_maps_results_to_retrieved_chunks() -> None:
    vector_store = FakeChroma()
    store = ChromaVectorStore(
        persist_dir="unused",
        collection_name="unused",
        vector_store=vector_store,
    )

    results = asyncio.run(store.search(query_embedding=[0.1, 0.2], limit=1))

    assert vector_store.search_payload == {"embedding": [0.1, 0.2], "k": 1}
    assert len(results) == 1
    assert results[0].chunk.id == "chunk-1"
    assert results[0].chunk.content == "chunk content"
    assert results[0].chunk.metadata == {
        "chunk_id": "chunk-1",
        "document_id": "doc",
        "chunk_index": "0",
    }
    assert results[0].similarity_score == 0.8


def test_chroma_vector_store_rejects_invalid_search_limit() -> None:
    store = ChromaVectorStore(
        persist_dir="unused",
        collection_name="unused",
        vector_store=FakeChroma(),
    )

    with pytest.raises(ValueError):
        asyncio.run(store.search(query_embedding=[0.1, 0.2], limit=0))
