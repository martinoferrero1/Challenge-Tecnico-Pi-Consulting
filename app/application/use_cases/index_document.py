from dataclasses import dataclass

from app.application.ports.embedding_model import EmbeddingModelPort
from app.application.ports.vector_store import VectorStorePort
from app.application.use_cases.ingest_document import (
    IngestDocumentUseCase,
    IngestedDocument,
)


@dataclass(frozen=True)
class IndexedDocument:
    ingested_document: IngestedDocument
    embeddings_count: int


class IndexDocumentUseCase:
    def __init__(
        self,
        ingest_document_use_case: IngestDocumentUseCase,
        embedding_model: EmbeddingModelPort,
        vector_store: VectorStorePort,
    ) -> None:
        self.ingest_document_use_case = ingest_document_use_case
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    async def execute(self, source: str) -> IndexedDocument:
        ingested_document = await self.ingest_document_use_case.execute(source)
        chunk_contents = [chunk.content for chunk in ingested_document.chunks]
        embeddings = await self.embedding_model.embed_batch(chunk_contents)

        if len(embeddings) != len(ingested_document.chunks): # a priori deberia cumplir siempre, pero hago la validacion por seguridad
            raise ValueError("The number of embeddings must match the number of chunks")

        await self.vector_store.add_chunks(ingested_document.chunks, embeddings)

        return IndexedDocument(
            ingested_document=ingested_document,
            embeddings_count=len(embeddings),
        )
