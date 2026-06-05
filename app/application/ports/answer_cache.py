from typing import Protocol

from app.domain.entities.answer import Answer
from app.domain.entities.cache_key import AnswerCacheKey


class AnswerCachePort(Protocol):
    async def get(self, key: AnswerCacheKey) -> Answer | None:
        raise NotImplementedError

    async def set(self, key: AnswerCacheKey, answer: Answer) -> None:
        raise NotImplementedError
