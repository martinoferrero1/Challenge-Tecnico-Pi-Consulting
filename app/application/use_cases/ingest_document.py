from dataclasses import dataclass

from app.application.ports.document_chunker import DocumentChunkerPort
from app.application.ports.document_loader import DocumentLoaderPort
from app.domain.entities.document import Document
from app.domain.entities.document_chunk import DocumentChunk


@dataclass(frozen=True)
class IngestedDocument:
    document: Document
    chunks: tuple[DocumentChunk, ...]


class IngestDocumentUseCase:
    def __init__(
        self,
        document_loader: DocumentLoaderPort,
        document_chunker: DocumentChunkerPort,
    ) -> None:
        self.document_loader = document_loader
        self.document_chunker = document_chunker

    async def execute(self, source: str) -> IngestedDocument:
        print(f"Loading document from source: {source}")
        if not source.strip():
            raise ValueError("Document source cannot be empty")

        document = await self.document_loader.load(source)
        chunks = self.document_chunker.chunk(document)

        return IngestedDocument(document=document, chunks=chunks)
