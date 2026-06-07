from functools import lru_cache
from typing import Protocol

from app.application.ports.answer_cache import AnswerCachePort
from app.application.use_cases.answer_question import (
    AnswerQuestionConfig,
    AnswerQuestionUseCase,
)
from app.infrastructure.answer_caches.in_memory_answer_cache import (
    InMemoryAnswerCache,
)
from app.infrastructure.embedding_models.embedding_model_factory import (
    create_embedding_model,
)
from app.infrastructure.llms.llm_factory import create_llm
from app.infrastructure.vector_stores.chroma_vector_store import ChromaVectorStore


class AnswerQuestionSettings(Protocol):
    rag_retrieval_limit: int
    llm_provider: str
    embedding_provider: str
    openai_api_key: str | None
    openai_llm_model: str
    openai_embedding_model: str
    cohere_api_key: str | None
    cohere_llm_model: str
    cohere_embedding_model: str
    cohere_embedding_input_type: str
    gemini_api_key: str | None
    gemini_llm_model: str
    gemini_embedding_model: str
    chroma_persist_dir: str
    chroma_collection_name: str


def create_answer_question_use_case(
    settings: AnswerQuestionSettings,
) -> AnswerQuestionUseCase:
    return AnswerQuestionUseCase(
        embedding_model=create_embedding_model(settings),
        vector_store=ChromaVectorStore(
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
        ),
        llm=create_llm(settings),
        answer_cache=get_answer_cache(),
        config=AnswerQuestionConfig(
            retrieval_limit=settings.rag_retrieval_limit,
        ),
    )


@lru_cache(maxsize=1)
def get_answer_cache() -> AnswerCachePort:
    return InMemoryAnswerCache()
