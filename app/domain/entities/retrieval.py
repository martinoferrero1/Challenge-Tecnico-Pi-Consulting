from dataclasses import dataclass

from app.domain.entities.document_chunk import DocumentChunk


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: DocumentChunk
    similarity_score: float

    def __post_init__(self) -> None:
        if self.similarity_score < 0:
            raise ValueError("The similarity score cannot be negative")
