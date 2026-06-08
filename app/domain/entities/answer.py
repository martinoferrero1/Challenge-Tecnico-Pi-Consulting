from dataclasses import dataclass, field

from app.domain.entities.question import UserQuestion
from app.domain.entities.retrieval import RetrievedChunk


@dataclass(frozen=True)
class AnswerDiagnostics:
    """Diagnóstico interno asociado a una respuesta RAG.

    Atributos:
        conversation_context_mode: Modo de contexto conversacional usado en la
            request.
        answer_cache_mode: Modo de cache de respuestas usado en la request.
        cache_hit: Indica si la respuesta salió desde cache.
        cache_hit_source: Nombre del modo o etapa que produjo el cache hit.
        stage_latencies_ms: Latencias acumuladas por etapa en milisegundos.
    """

    conversation_context_mode: str
    answer_cache_mode: str
    cache_hit: bool = False
    cache_hit_source: str | None = None
    stage_latencies_ms: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Answer:
    """Respuesta generada para una pregunta y el contexto usado.

    Atributos:
        question: Pregunta del usuario que originó la respuesta.
        content: Texto final devuelto por el asistente.
        context: Chunks recuperados o reutilizados para fundamentar la
            respuesta.
        resolved_query: Pregunta reescrita para retrieval cuando difiere de la
            pregunta original.
        diagnostics: Diagnóstico interno opcional para evaluación y métricas.
    """

    question: UserQuestion
    content: str
    context: tuple[RetrievedChunk, ...]
    resolved_query: str | None = None
    diagnostics: AnswerDiagnostics | None = None

    def __post_init__(self) -> None:
        """Valida que ``content`` y ``resolved_query`` no sean vacíos.

        Raises:
            ValueError: Si ``content`` queda vacío o si ``resolved_query`` fue
                informado pero no contiene texto útil.
        """
        if not self.content.strip():
            raise ValueError("The answer content cannot be empty")
        if self.resolved_query is not None and not self.resolved_query.strip():
            raise ValueError("The resolved query cannot be empty")
