import asyncio
from typing import Sequence

from app.application.use_cases.answer_question import AnswerQuestionUseCase
from app.domain.entities.answer import Answer
from app.domain.entities.cache_key import AnswerCacheKey
from app.domain.entities.document_chunk import DocumentChunk
from app.domain.entities.question import UserQuestion
from app.domain.entities.retrieval import RetrievedChunk


class FakeEmbeddingModel:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def embed_text(self, text: str) -> list[float]:
        self.texts.append(text)
        return [1.0, 2.0, 3.0]

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]


class FakeVectorStore:
    def __init__(self, retrieved_chunks: list[RetrievedChunk]) -> None:
        self.retrieved_chunks = retrieved_chunks
        self.query_embeddings: list[list[float]] = []
        self.limits: list[int] = []

    async def add_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        return None

    async def search(
        self,
        query_embedding: Sequence[float],
        limit: int,
    ) -> list[RetrievedChunk]:
        self.query_embeddings.append(list(query_embedding))
        self.limits.append(limit)
        return self.retrieved_chunks


class FakeLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "Zara es una empresa de moda."


class FakeAnswerCache:
    def __init__(self) -> None:
        self.answers: dict[AnswerCacheKey, Answer] = {}

    async def get(self, key: AnswerCacheKey) -> Answer | None:
        return self.answers.get(key)

    async def set(self, key: AnswerCacheKey, answer: Answer) -> None:
        self.answers[key] = answer


def test_answer_question_generates_and_caches_answer() -> None:
    retrieved_chunk = RetrievedChunk(
        chunk=DocumentChunk(
            id="chunk-1",
            content="Zara es una empresa internacional de moda.",
        ),
        similarity_score=0.9,
    )
    embedding_model = FakeEmbeddingModel()
    vector_store = FakeVectorStore([retrieved_chunk])
    llm = FakeLLM()
    cache = FakeAnswerCache()
    use_case = AnswerQuestionUseCase(
        embedding_model=embedding_model,
        vector_store=vector_store,
        llm=llm,
        answer_cache=cache,
    )

    first_answer = asyncio.run(
        use_case.execute(UserQuestion(user_name="Ana", content="Que es Zara?"))
    )
    cached_answer = asyncio.run(
        use_case.execute(UserQuestion(user_name="Luis", content=" Que es Zara? "))
    )

    assert first_answer.content == "Zara es una empresa de moda."
    assert cached_answer.content == "Zara es una empresa de moda."
    assert cached_answer.question.user_name == "Luis"
    assert len(llm.prompts) == 1
    assert "Zara es una empresa internacional de moda." in llm.prompts[0]
    assert vector_store.limits == [4, 4]


def test_answer_question_returns_fallback_when_no_context_is_found() -> None:
    use_case = AnswerQuestionUseCase(
        embedding_model=FakeEmbeddingModel(),
        vector_store=FakeVectorStore([]),
        llm=FakeLLM(),
        answer_cache=FakeAnswerCache(),
    )

    answer = asyncio.run(
        use_case.execute(UserQuestion(user_name="Ana", content="Pregunta sin datos"))
    )

    assert answer.content == (
        "No encontre informacion suficiente en el documento para responder esa pregunta."
    )
    assert answer.context == ()
