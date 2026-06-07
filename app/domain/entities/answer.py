from dataclasses import dataclass

from app.domain.entities.question import UserQuestion
from app.domain.entities.retrieval import RetrievedChunk


@dataclass(frozen=True)
class Answer:
    question: UserQuestion
    content: str
    context: tuple[RetrievedChunk, ...]
    resolved_query: str | None = None

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("The answer content cannot be empty")
        if self.resolved_query is not None and not self.resolved_query.strip():
            raise ValueError("The resolved query cannot be empty")
