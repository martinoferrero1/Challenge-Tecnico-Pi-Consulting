from app.domain.entities.answer import Answer
from app.domain.entities.cache_key import AnswerCacheKey


class InMemoryAnswerCache:
    def __init__(self) -> None:
        self._answers: dict[AnswerCacheKey, Answer] = {}

    async def get(self, key: AnswerCacheKey) -> Answer | None:
        return self._answers.get(key)

    async def set(self, key: AnswerCacheKey, answer: Answer) -> None:
        self._answers[key] = answer

    def clear(self) -> None:
        self._answers.clear()
