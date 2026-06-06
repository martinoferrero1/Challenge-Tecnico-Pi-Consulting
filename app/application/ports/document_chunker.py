from typing import Protocol

from app.domain.entities.document import Document
from app.domain.entities.document_chunk import DocumentChunk


class DocumentChunkerPort(Protocol):
    def chunk(self, document: Document) -> tuple[DocumentChunk, ...]:
        raise NotImplementedError
