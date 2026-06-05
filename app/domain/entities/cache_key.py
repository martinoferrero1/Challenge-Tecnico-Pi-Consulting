from dataclasses import dataclass


@dataclass(frozen=True)
class AnswerCacheKey:
    question: str
    context_hash: str

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("The question of the cache key cannot be empty")
        if not self.context_hash.strip():
            raise ValueError("The context hash of the cache key cannot be empty")