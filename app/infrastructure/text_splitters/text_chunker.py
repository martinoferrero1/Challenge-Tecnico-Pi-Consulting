from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from app.domain.entities.document import Document
from app.domain.entities.document_chunk import DocumentChunk


@dataclass(frozen=True)
class TextSplitterConfig:
    chunk_size: int = 800
    chunk_overlap: int = 120

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("The chunk size must be greater than zero")
        if self.chunk_overlap < 0:
            raise ValueError("The chunk overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("The chunk overlap must be lower than the chunk size")


class TextChunker:
    def __init__(
        self,
        config: TextSplitterConfig | None = None,
        splitter: Any | None = None,
    ) -> None:
        self.config = config or TextSplitterConfig()
        self.splitter = splitter or self._build_splitter()

    def chunk(self, document: Document) -> tuple[DocumentChunk, ...]:
        source_document = self._to_source_document(document)
        split_documents = self.splitter.split_documents([source_document])
        chunks: list[DocumentChunk] = []

        for chunk_index, split_document in enumerate(split_documents):
            content = str(split_document.page_content).strip()
            if not content:
                continue

            metadata = self._string_metadata(
                {
                    **document.metadata,
                    **dict(getattr(split_document, "metadata", {}) or {}),
                    "document_id": document.id,
                    "chunk_index": str(chunk_index),
                }
            )

            chunks.append(
                DocumentChunk(
                    id=self._build_chunk_id(document.id, chunk_index, content),
                    content=content,
                    metadata=metadata,
                )
            )

        return tuple(chunks)

    def _build_splitter(self) -> Any:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        return RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            add_start_index=True,
        )

    def _to_source_document(self, document: Document) -> Any:
        from langchain_core.documents import Document as SourceDocument

        return SourceDocument(
            page_content=document.content,
            metadata={
                **document.metadata,
                "document_id": document.id,
            },
        )

    def _build_chunk_id(
        self,
        document_id: str,
        chunk_index: int,
        content: str,
    ) -> str:
        content_hash = sha256(content.encode("utf-8")).hexdigest()[:12]
        return f"{document_id}:chunk:{chunk_index}:{content_hash}"

    def _string_metadata(self, metadata: dict[str, Any]) -> dict[str, str]:
        return {
            str(key): str(value)
            for key, value in metadata.items()
            if value is not None
        }
