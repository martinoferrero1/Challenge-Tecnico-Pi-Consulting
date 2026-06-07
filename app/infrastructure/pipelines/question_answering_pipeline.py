from functools import lru_cache
from typing import Protocol

from app.application.ports.answer_cache import AnswerCachePort
from app.application.ports.conversation_store import ConversationStorePort
from app.application.use_cases.answer_question import (
    AnswerQuestionConfig,
    AnswerQuestionUseCase,
)
from app.infrastructure.answer_caches.in_memory_answer_cache import (
    InMemoryAnswerCache,
)
from app.infrastructure.conversation_stores.in_memory_conversation_store import (
    InMemoryConversationStore,
)
from app.infrastructure.embedding_models.embedding_model_factory import (
    create_embedding_model,
)
from app.infrastructure.language_detectors.lingua_language_detector import (
    LinguaLanguageDetector,
)
from app.infrastructure.llms.llm_factory import create_llm
from app.infrastructure.vector_store.chroma_vector_store import ChromaVectorStore


class AnswerQuestionSettings(Protocol):
    rag_retrieval_limit: int
    conversation_context_mode: str
    answer_cache_mode: str
    conversation_history_limit: int
    language_confidence_threshold: float
    answer_validation_retries: int
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
        language_detector=LinguaLanguageDetector(),
        conversation_store=get_conversation_store(),
        config=AnswerQuestionConfig(
            retrieval_limit=settings.rag_retrieval_limit,
            conversation_context_mode=settings.conversation_context_mode,
            answer_cache_mode=settings.answer_cache_mode,
            conversation_history_limit=settings.conversation_history_limit,
            language_confidence_threshold=settings.language_confidence_threshold,
            answer_validation_retries=settings.answer_validation_retries,
        ),
    )


@lru_cache(maxsize=1)
def get_answer_cache() -> AnswerCachePort:
    return InMemoryAnswerCache()


@lru_cache(maxsize=1)
def get_conversation_store() -> ConversationStorePort:
    return InMemoryConversationStore()
