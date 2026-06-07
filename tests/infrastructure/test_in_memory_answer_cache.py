import asyncio

from app.domain.entities.answer import Answer
from app.domain.entities.cache_key import AnswerCacheKey
from app.domain.entities.question import UserQuestion
from app.infrastructure.answer_caches.in_memory_answer_cache import (
    InMemoryAnswerCache,
)


def test_in_memory_answer_cache_stores_and_returns_answers() -> None:
    cache = InMemoryAnswerCache()
    key = AnswerCacheKey(question="que es zara?", context_hash="context")
    answer = Answer(
        question=UserQuestion(user_name="Ana", content="Que es Zara?"),
        content="Zara es una empresa de moda.",
        context=(),
    )

    asyncio.run(cache.set(key, answer))

    assert asyncio.run(cache.get(key)) == answer


def test_in_memory_answer_cache_returns_none_for_missing_key() -> None:
    cache = InMemoryAnswerCache()
    key = AnswerCacheKey(question="que es zara?", context_hash="context")

    assert asyncio.run(cache.get(key)) is None
