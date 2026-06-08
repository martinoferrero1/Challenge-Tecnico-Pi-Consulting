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


def test_in_memory_answer_cache_lists_answers_by_question() -> None:
    cache = InMemoryAnswerCache()
    zara_answer = Answer(
        question=UserQuestion(user_name="Ana", content="Que es Zara?"),
        content="Zara es una empresa de moda.",
        context=(),
    )
    mango_answer = Answer(
        question=UserQuestion(user_name="Ana", content="Que es Mango?"),
        content="Mango es una empresa de moda.",
        context=(),
    )

    asyncio.run(
        cache.set(
            AnswerCacheKey(question="que es zara?", context_hash="context-a"),
            zara_answer,
        )
    )
    asyncio.run(
        cache.set(
            AnswerCacheKey(question="que es mango?", context_hash="context-b"),
            mango_answer,
        )
    )

    assert asyncio.run(cache.list_by_question(" Que es Zara? ")) == [zara_answer]
