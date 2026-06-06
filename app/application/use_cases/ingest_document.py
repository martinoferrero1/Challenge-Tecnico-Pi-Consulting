from dataclasses import dataclass

from app.application.ports.document_loader import DocumentLoaderPort
from app.application.services.text_chunker import TextChunker
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
        text_chunker: TextChunker,
    ) -> None:
        self.document_loader = document_loader
        self.text_chunker = text_chunker

    async def execute(self, source: str) -> IngestedDocument:
        if not source.strip():
            raise ValueError("Document source cannot be empty")

        document = await self.document_loader.load(source)
        chunks = self.text_chunker.chunk(document)

        return IngestedDocument(document=document, chunks=chunks)
