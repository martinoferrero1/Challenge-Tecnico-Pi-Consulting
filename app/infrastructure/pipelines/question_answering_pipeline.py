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
    """Subset de settings requerido para responder preguntas.

    Atributos:
        rag_retrieval_limit: Cantidad máxima de chunks a recuperar.
        conversation_context_mode: Modo de uso del historial conversacional.
        answer_cache_mode: Modo de cache de respuestas.
        conversation_history_limit: Cantidad máxima de mensajes del historial.
        language_confidence_threshold: Umbral de confianza para idioma.
        answer_validation_retries: Cantidad de reintentos de formato.
        llm_provider: Provider LLM seleccionado.
        embedding_provider: Provider de embeddings seleccionado.
        openai_api_key: API key de OpenAI.
        openai_llm_model: Modelo LLM de OpenAI.
        openai_embedding_model: Modelo de embeddings de OpenAI.
        cohere_api_key: API key de Cohere.
        cohere_llm_model: Modelo LLM de Cohere.
        cohere_embedding_model: Modelo de embeddings de Cohere.
        cohere_embedding_input_type: Tipo de input de embeddings de Cohere.
        gemini_api_key: API key de Gemini.
        gemini_llm_model: Modelo LLM de Gemini.
        gemini_embedding_model: Modelo de embeddings de Gemini.
        chroma_persist_dir: Carpeta de persistencia de Chroma.
        chroma_collection_name: Colección Chroma usada para retrieval.
    """

    rag_retrieval_limit: int
    conversation_context_mode: str
    answer_cache_mode: str
    conversation_history_limit: int
    language_confidence_threshold: float
    answer_validation_retries: int
    llm_provider: str
    llm_temperature: float
    judge_llm_provider: str | None
    judge_llm_model: str | None
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
    """Arma el caso de uso de preguntas con infraestructura concreta.

    Args:
        settings: Configuración requerida para LLM, embeddings, Chroma, cache y
            conversación.

    Returns:
        Caso de uso listo para responder preguntas.
    """
    llm = create_llm(settings)
    judge_llm_provider = getattr(settings, "judge_llm_provider", None)
    judge_llm_model = getattr(settings, "judge_llm_model", None)
    has_judge_provider = bool(judge_llm_provider and judge_llm_provider.strip())
    has_judge_model = bool(judge_llm_model and judge_llm_model.strip())
    cache_judge_llm = (
        create_llm(
            settings,
            provider_override=judge_llm_provider,
            model_override=judge_llm_model,
        )
        if has_judge_provider or has_judge_model
        else llm
    )

    return AnswerQuestionUseCase(
        embedding_model=create_embedding_model(settings),
        vector_store=ChromaVectorStore(
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
        ),
        llm=llm,
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
        cache_judge_llm=cache_judge_llm,
    )


@lru_cache(maxsize=1)
def get_answer_cache() -> AnswerCachePort:
    """Devuelve el cache singleton en memoria.

    Returns:
        Instancia única de ``InMemoryAnswerCache`` para el proceso actual.
    """
    return InMemoryAnswerCache()


@lru_cache(maxsize=1)
def get_conversation_store() -> ConversationStorePort:
    """Devuelve el store singleton de conversación en memoria.

    Returns:
        Instancia única de ``InMemoryConversationStore`` para el proceso actual.
    """
    return InMemoryConversationStore()
