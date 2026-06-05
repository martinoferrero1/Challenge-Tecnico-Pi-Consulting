from dataclasses import dataclass
from hashlib import sha256
import re

from app.domain.entities.document import Document
from app.domain.entities.document_chunk import DocumentChunk


@dataclass(frozen=True)
class ChunkingConfig:
    max_words: int = 160
    overlap_words: int = 30

    def __post_init__(self) -> None:
        if self.max_words <= 0:
            raise ValueError("The maximum number of words must be greater than zero")
        if self.overlap_words < 0:
            raise ValueError("The overlap number of words cannot be negative")
        if self.overlap_words >= self.max_words:
            raise ValueError("The overlap must be lower than the chunk size")


class TextChunker:
    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk(self, document: Document) -> tuple[DocumentChunk, ...]:
        words = self._split_words(document.content)
        step = self.config.max_words - self.config.overlap_words
        chunks: list[DocumentChunk] = []

        for chunk_index, start in enumerate(range(0, len(words), step)):
            end = min(start + self.config.max_words, len(words))
            chunk_content = " ".join(words[start:end])

            chunks.append(
                DocumentChunk(
                    id=self._build_chunk_id(document.id, chunk_index, chunk_content),
                    content=chunk_content,
                    metadata={
                        **document.metadata,
                        "document_id": document.id,
                        "chunk_index": str(chunk_index),
                        "start_word": str(start),
                        "end_word": str(end),
                    },
                )
            )

            if end == len(words):
                break

        return tuple(chunks)

    def _build_chunk_id(
        self,
        document_id: str,
        chunk_index: int,
        content: str,
    ) -> str:
        content_hash = sha256(content.encode("utf-8")).hexdigest()[:12]
        return f"{document_id}:chunk:{chunk_index}:{content_hash}"

    def _split_words(self, text: str) -> list[str]:
        return re.findall(r"\S+", text)
