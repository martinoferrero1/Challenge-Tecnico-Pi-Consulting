import asyncio
from typing import Sequence, TypeVar

import pytest
from pydantic import BaseModel

from app.application.errors import ExternalServiceError
from app.application.use_cases.answer_question import (
    AnswerQuestionConfig,
    AnswerQuestionUseCase,
)
from app.domain.entities.answer import Answer
from app.domain.entities.cache_key import AnswerCacheKey
from app.domain.entities.question import ConversationMessage
from app.domain.entities.document_chunk import DocumentChunk
from app.domain.entities.language import DetectedLanguage
from app.domain.entities.question import UserQuestion
from app.domain.entities.retrieval import RetrievedChunk
from app.infrastructure.conversation_stores.in_memory_conversation_store import (
    InMemoryConversationStore,
)


StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)


class FakeEmbeddingModel:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def embed_text(self, text: str) -> list[float]:
        self.texts.append(text)
        return [1.0, 2.0, 3.0]

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]


class FailingEmbeddingModel(FakeEmbeddingModel):
    async def embed_text(self, text: str) -> list[float]:
        raise RuntimeError("embedding provider is down")


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
    def __init__(
        self,
        response: str = "Zara es una empresa de moda 👗.",
        responses: list[str] | None = None,
    ) -> None:
        self.responses = responses or [response]
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        response_index = min(len(self.prompts), len(self.responses) - 1)
        self.prompts.append(prompt)
        return self.responses[response_index]

    async def generate_structured(
        self,
        prompt: str,
        output_schema: type[StructuredOutput],
    ) -> StructuredOutput:
        response_index = min(len(self.prompts), len(self.responses) - 1)
        self.prompts.append(prompt)
        return output_schema.model_validate(
            {
                "decision": self.responses[response_index].strip().casefold(),
            }
        )


class FakeAnswerCache:
    def __init__(self) -> None:
        self.answers: dict[AnswerCacheKey, Answer] = {}

    async def get(self, key: AnswerCacheKey) -> Answer | None:
        return self.answers.get(key)

    async def set(self, key: AnswerCacheKey, answer: Answer) -> None:
        self.answers[key] = answer

    async def list_by_question(self, question: str) -> list[Answer]:
        normalized_question = " ".join(question.strip().split()).casefold()

        return [
            answer
            for key, answer in self.answers.items()
            if key.question == normalized_question
        ]


class FakeLanguageDetector:
    def __init__(
        self,
        language: DetectedLanguage | None = None,
        languages: list[DetectedLanguage | None] | None = None,
    ) -> None:
        self.languages = languages or [language]
        self.texts: list[str] = []

    def detect(self, text: str) -> DetectedLanguage | None:
        language_index = min(len(self.texts), len(self.languages) - 1)
        self.texts.append(text)
        return self.languages[language_index]


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
    language_detector = FakeLanguageDetector(
        DetectedLanguage(name="Spanish", confidence=0.99)
    )
    use_case = AnswerQuestionUseCase(
        embedding_model=embedding_model,
        vector_store=vector_store,
        llm=llm,
        answer_cache=cache,
        language_detector=language_detector,
    )

    first_answer = asyncio.run(
        use_case.execute(UserQuestion(user_name="Ana", content="Que es Zara?"))
    )
    cached_answer = asyncio.run(
        use_case.execute(UserQuestion(user_name="Luis", content=" Que es Zara? "))
    )

    assert first_answer.content == "Zara es una empresa de moda 👗."
    assert cached_answer.content == "Zara es una empresa de moda 👗."
    assert cached_answer.question.user_name == "Luis"
    assert len(llm.prompts) == 1
    assert "Zara es una empresa internacional de moda." in llm.prompts[0]
    assert "You must answer in Spanish." in llm.prompts[0]
    assert "answer in another language" in llm.prompts[0]
    assert "Answer in exactly one sentence." in llm.prompts[0]
    assert "Include one or more relevant emojis" in llm.prompts[0]
    assert "Answer in third person" in llm.prompts[0]
    assert vector_store.limits == [4, 4]


def test_answer_question_returns_fallback_when_no_context_is_found() -> None:
    llm = FakeLLM(
        response="The document does not provide enough information to answer ❓."
    )
    use_case = AnswerQuestionUseCase(
        embedding_model=FakeEmbeddingModel(),
        vector_store=FakeVectorStore([]),
        llm=llm,
        answer_cache=FakeAnswerCache(),
        language_detector=FakeLanguageDetector(
            DetectedLanguage(name="English", confidence=0.98)
        ),
    )

    answer = asyncio.run(
        use_case.execute(UserQuestion(user_name="Ana", content="Pregunta sin datos"))
    )

    assert answer.content == "The document does not provide enough information to answer ❓."
    assert answer.context == ()
    assert "Return only a fallback answer." in llm.prompts[0]
    assert "You must answer in English." in llm.prompts[0]
    assert "exactly one sentence" in llm.prompts[0]
    assert "relevant emojis" in llm.prompts[0]


def test_answer_question_wraps_embedding_failures() -> None:
    use_case = AnswerQuestionUseCase(
        embedding_model=FailingEmbeddingModel(),
        vector_store=FakeVectorStore([]),
        llm=FakeLLM(),
        answer_cache=FakeAnswerCache(),
        language_detector=FakeLanguageDetector(
            DetectedLanguage(name="Spanish", confidence=0.99)
        ),
    )

    with pytest.raises(ExternalServiceError) as error:
        asyncio.run(
            use_case.execute(UserQuestion(user_name="Ana", content="Que es Zara?"))
        )

    assert error.value.cause == "embedding provider is down"


def test_answer_question_does_not_pin_language_when_confidence_is_low() -> None:
    retrieved_chunk = RetrievedChunk(
        chunk=DocumentChunk(
            id="chunk-1",
            content="Zara es una empresa internacional de moda.",
        ),
        similarity_score=0.9,
    )
    llm = FakeLLM()
    use_case = AnswerQuestionUseCase(
        embedding_model=FakeEmbeddingModel(),
        vector_store=FakeVectorStore([retrieved_chunk]),
        llm=llm,
        answer_cache=FakeAnswerCache(),
        language_detector=FakeLanguageDetector(
            DetectedLanguage(name="Spanish", confidence=0.2)
        ),
        config=AnswerQuestionConfig(language_confidence_threshold=0.75),
    )

    asyncio.run(
        use_case.execute(UserQuestion(user_name="Ana", content="Que es Zara?"))
    )

    assert "You must answer in Spanish." not in llm.prompts[0]
    assert "Do not use a default language." in llm.prompts[0]


def test_answer_question_rewrites_retrieval_query_with_conversation_history() -> None:
    retrieved_chunk = RetrievedChunk(
        chunk=DocumentChunk(
            id="chunk-1",
            content="Zara fue fundada en 1975.",
        ),
        similarity_score=0.9,
    )
    embedding_model = FakeEmbeddingModel()
    vector_store = FakeVectorStore([retrieved_chunk])
    llm = FakeLLM(
        responses=[
            "En que anio fue fundada Zara?",
            "Zara fue fundada en 1975.",
        ]
    )
    use_case = AnswerQuestionUseCase(
        embedding_model=embedding_model,
        vector_store=vector_store,
        llm=llm,
        answer_cache=FakeAnswerCache(),
        language_detector=FakeLanguageDetector(
            DetectedLanguage(name="Spanish", confidence=0.99)
        ),
        config=AnswerQuestionConfig(
            conversation_context_mode="rewrite",
            answer_cache_mode="context_aware",
            answer_validation_retries=0,
        ),
    )

    answer = asyncio.run(
        use_case.execute(
            UserQuestion(
                user_name="Ana",
                content="Y en que anio?",
                conversation_history=(
                    ConversationMessage(
                        role="user",
                        content="Quien fundo Zara?",
                    ),
                    ConversationMessage(
                        role="assistant",
                        content="Zara fue fundada por Amancio Ortega.",
                    ),
                ),
            )
        )
    )

    assert answer.content == "Zara fue fundada en 1975."
    assert answer.resolved_query == "En que anio fue fundada Zara?"
    assert embedding_model.texts == ["En que anio fue fundada Zara?"]
    assert len(llm.prompts) == 2
    assert "Rewrite the current user question" in llm.prompts[0]
    assert "Write the output in Spanish." in llm.prompts[0]
    assert "Conversation history:" in llm.prompts[1]
    assert "Resolved question for retrieval:" in llm.prompts[1]


def test_answer_question_uses_stored_conversation_without_request_history() -> None:
    retrieved_chunk = RetrievedChunk(
        chunk=DocumentChunk(
            id="chunk-1",
            content="Zara fue fundada en 1975.",
        ),
        similarity_score=0.9,
    )
    embedding_model = FakeEmbeddingModel()
    vector_store = FakeVectorStore([retrieved_chunk])
    llm = FakeLLM(
        responses=[
            "Zara fue fundada por Amancio Ortega.",
            "En que anio fue fundada Zara?",
            "Zara fue fundada en 1975.",
        ]
    )
    conversation_store = InMemoryConversationStore()
    use_case = AnswerQuestionUseCase(
        embedding_model=embedding_model,
        vector_store=vector_store,
        llm=llm,
        answer_cache=FakeAnswerCache(),
        language_detector=FakeLanguageDetector(
            DetectedLanguage(name="Spanish", confidence=0.99)
        ),
        conversation_store=conversation_store,
        config=AnswerQuestionConfig(
            conversation_context_mode="rewrite",
            answer_validation_retries=0,
        ),
    )

    first_answer = asyncio.run(
        use_case.execute(
            UserQuestion(
                user_name="Ana",
                content="Quien fundo Zara?",
            )
        )
    )
    second_answer = asyncio.run(
        use_case.execute(
            UserQuestion(
                user_name="Ana",
                content="Y en que anio?",
            )
        )
    )

    stored_history = asyncio.run(
        conversation_store.get_recent(
            conversation_key="ana",
            limit=10,
        )
    )

    assert first_answer.content == "Zara fue fundada por Amancio Ortega."
    assert second_answer.content == "Zara fue fundada en 1975."
    assert second_answer.resolved_query == "En que anio fue fundada Zara?"
    assert embedding_model.texts == [
        "Quien fundo Zara?",
        "En que anio fue fundada Zara?",
    ]
    assert "User: Quien fundo Zara?" in llm.prompts[1]
    assert "Assistant: Zara fue fundada por Amancio Ortega." in llm.prompts[1]
    assert [message.content for message in stored_history] == [
        "Quien fundo Zara?",
        "Zara fue fundada por Amancio Ortega.",
        "Y en que anio?",
        "Zara fue fundada en 1975.",
    ]


def test_context_aware_cache_judges_same_question_with_different_context() -> None:
    retrieved_chunk = RetrievedChunk(
        chunk=DocumentChunk(
            id="chunk-1",
            content="El documento contiene fechas de apertura.",
        ),
        similarity_score=0.9,
    )
    embedding_model = FakeEmbeddingModel()
    vector_store = FakeVectorStore([retrieved_chunk])
    llm = FakeLLM(
        responses=[
            "Cuando fue fundada Zara?",
            "Zara fue fundada en 1975.",
            "Cuando abrio Mango?",
            "different",
            "Mango abrio en 1984.",
        ]
    )
    cache = FakeAnswerCache()
    use_case = AnswerQuestionUseCase(
        embedding_model=embedding_model,
        vector_store=vector_store,
        llm=llm,
        answer_cache=cache,
        language_detector=FakeLanguageDetector(
            DetectedLanguage(name="Spanish", confidence=0.99)
        ),
        config=AnswerQuestionConfig(
            conversation_context_mode="rewrite",
            answer_cache_mode="context_aware",
            answer_validation_retries=0,
        ),
    )

    first_answer = asyncio.run(
        use_case.execute(
            UserQuestion(
                user_name="Ana",
                content="Y cuando?",
                conversation_history=(
                    ConversationMessage(
                        role="user",
                        content="Quien fundo Zara?",
                    ),
                ),
            )
        )
    )
    second_answer = asyncio.run(
        use_case.execute(
            UserQuestion(
                user_name="Ana",
                content="Y cuando?",
                conversation_history=(
                    ConversationMessage(
                        role="user",
                        content="Cuando se creo Mango?",
                    ),
                ),
            )
        )
    )

    assert first_answer.content == "Zara fue fundada en 1975."
    assert second_answer.content == "Mango abrio en 1984."
    assert embedding_model.texts == [
        "Cuando fue fundada Zara?",
        "Cuando abrio Mango?",
    ]
    assert len(cache.answers) == 2
    assert len(llm.prompts) == 5
    assert 'Set decision to "same"' in llm.prompts[3]
    assert 'Set decision to "different"' in llm.prompts[3]


def test_context_aware_cache_does_not_shortcut_by_question_without_resolved_query() -> None:
    retrieved_chunk = RetrievedChunk(
        chunk=DocumentChunk(
            id="chunk-1",
            content="El documento contiene fechas de apertura.",
        ),
        similarity_score=0.9,
    )
    embedding_model = FakeEmbeddingModel()
    vector_store = FakeVectorStore([retrieved_chunk])
    llm = FakeLLM(
        responses=[
            "Zara fue fundada en 1975.",
            "different",
            "Mango abrio en 1984.",
        ]
    )
    cache = FakeAnswerCache()
    use_case = AnswerQuestionUseCase(
        embedding_model=embedding_model,
        vector_store=vector_store,
        llm=llm,
        answer_cache=cache,
        language_detector=FakeLanguageDetector(
            DetectedLanguage(name="Spanish", confidence=0.99)
        ),
        config=AnswerQuestionConfig(
            conversation_context_mode="prompt",
            answer_cache_mode="context_aware",
            answer_validation_retries=0,
        ),
    )

    first_answer = asyncio.run(
        use_case.execute(
            UserQuestion(
                user_name="Ana",
                content="Y cuando?",
                conversation_history=(
                    ConversationMessage(
                        role="user",
                        content="Quien fundo Zara?",
                    ),
                ),
            )
        )
    )
    second_answer = asyncio.run(
        use_case.execute(
            UserQuestion(
                user_name="Ana",
                content="Y cuando?",
                conversation_history=(
                    ConversationMessage(
                        role="user",
                        content="Cuando se creo Mango?",
                    ),
                ),
            )
        )
    )

    assert first_answer.content == "Zara fue fundada en 1975."
    assert second_answer.content == "Mango abrio en 1984."
    assert first_answer.resolved_query is None
    assert second_answer.resolved_query is None
    assert embedding_model.texts == ["Y cuando?", "Y cuando?"]
    assert len(cache.answers) == 2
    assert len(llm.prompts) == 3
    assert 'Set decision to "same"' in llm.prompts[1]


def test_context_aware_cache_reuses_answer_when_structured_judge_returns_same() -> None:
    retrieved_chunk = RetrievedChunk(
        chunk=DocumentChunk(
            id="chunk-1",
            content="Zara fue fundada en 1975.",
        ),
        similarity_score=0.9,
    )
    embedding_model = FakeEmbeddingModel()
    vector_store = FakeVectorStore([retrieved_chunk])
    llm = FakeLLM(
        responses=[
            "Cuando fue fundada Zara?",
            "Zara fue fundada en 1975.",
            "En que anio se fundo Zara?",
            "same",
        ]
    )
    use_case = AnswerQuestionUseCase(
        embedding_model=embedding_model,
        vector_store=vector_store,
        llm=llm,
        answer_cache=FakeAnswerCache(),
        language_detector=FakeLanguageDetector(
            DetectedLanguage(name="Spanish", confidence=0.99)
        ),
        config=AnswerQuestionConfig(
            conversation_context_mode="rewrite",
            answer_cache_mode="context_aware",
            answer_validation_retries=0,
        ),
    )

    first_answer = asyncio.run(
        use_case.execute(
            UserQuestion(
                user_name="Ana",
                content="Y cuando?",
                conversation_history=(
                    ConversationMessage(
                        role="user",
                        content="Quien fundo Zara?",
                    ),
                ),
            )
        )
    )
    second_answer = asyncio.run(
        use_case.execute(
            UserQuestion(
                user_name="Ana",
                content="Y cuando?",
                conversation_history=(
                    ConversationMessage(
                        role="user",
                        content="En que anio se fundo Zara?",
                    ),
                ),
            )
        )
    )

    assert first_answer.content == "Zara fue fundada en 1975."
    assert second_answer.content == "Zara fue fundada en 1975."
    assert embedding_model.texts == ["Cuando fue fundada Zara?"]
    assert len(llm.prompts) == 4


def test_answer_question_rewrites_once_with_all_failed_validation_rules() -> None:
    retrieved_chunk = RetrievedChunk(
        chunk=DocumentChunk(
            id="chunk-1",
            content="Zara es una empresa internacional de moda.",
        ),
        similarity_score=0.9,
    )
    llm = FakeLLM(
        responses=[
            "I think Zara is a fashion company. It is international.",
            "Zara is an international fashion company 👗.",
        ]
    )
    use_case = AnswerQuestionUseCase(
        embedding_model=FakeEmbeddingModel(),
        vector_store=FakeVectorStore([retrieved_chunk]),
        llm=llm,
        answer_cache=FakeAnswerCache(),
        language_detector=FakeLanguageDetector(
            languages=[
                DetectedLanguage(name="English", confidence=0.99),
                DetectedLanguage(name="Spanish", confidence=0.99),
            ]
        ),
        config=AnswerQuestionConfig(answer_validation_retries=1),
    )

    answer = asyncio.run(
        use_case.execute(UserQuestion(user_name="Ana", content="What is Zara?"))
    )

    assert answer.content == "Zara is an international fashion company 👗."
    assert len(llm.prompts) == 2
    assert "heuristic assumptions" in llm.prompts[1]
    assert "The answer may not be exactly one sentence." in llm.prompts[1]
    assert "does not include emoji" in llm.prompts[1]
    assert "may not be written in third person" in llm.prompts[1]
    assert "may not be in English" in llm.prompts[1]
    assert "Previous answer:" in llm.prompts[1]
