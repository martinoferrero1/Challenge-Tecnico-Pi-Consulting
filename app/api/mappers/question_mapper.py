from app.api.schemas.question_schema import QuestionRequest
from app.api.schemas.question_schema import AnswerEvaluationResponse
from app.api.schemas.question_schema import AnswerResponse
from app.api.schemas.question_schema import AnswerDiagnosticsResponse
from app.api.schemas.question_schema import RetrievedChunkResponse
from app.domain.entities.answer import Answer
from app.domain.entities.question import UserQuestion


def map_question_request_to_entity(payload: QuestionRequest) -> UserQuestion: # mas alla de que en este caso el payload y la entidad son muy similares, hago esta capa de mapeo para desacoplar la API de la logica de negocio, y da flexibilidad para modificar cualquiera de las dos partes sin afectar a la otra
    """Convierte el payload HTTP en la entidad de pregunta del dominio.

    Args:
        payload: Request validado por FastAPI/Pydantic.

    Returns:
        Entidad ``UserQuestion`` usada por la capa de aplicación.
    """
    return UserQuestion(
        user_name=payload.user_name,
        content=payload.question,
    )


def map_answer_to_response(answer: Answer) -> AnswerResponse:
    """Convierte una respuesta del dominio en la respuesta pública de la API.

    Args:
        answer: Respuesta de dominio generada por el caso de uso.

    Returns:
        Schema público sin diagnósticos internos.
    """
    return AnswerResponse(
        user_name=answer.question.user_name,
        question=answer.question.content,
        answer=answer.content,
    )


def map_answer_to_evaluation_response(answer: Answer) -> AnswerEvaluationResponse:
    """Convierte una respuesta del dominio en salida de evaluación.

    Args:
        answer: Respuesta de dominio con ``diagnostics`` cargado.

    Returns:
        Schema extendido con diagnóstico para métricas offline.

    Raises:
        ValueError: Si ``answer`` no contiene ``diagnostics``.
    """
    return AnswerEvaluationResponse(
        user_name=answer.question.user_name,
        question=answer.question.content,
        answer=answer.content,
        diagnostics=_map_answer_diagnostics(answer),
    )


def _map_answer_diagnostics(answer: Answer) -> AnswerDiagnosticsResponse:
    """Mapea chunks, tiempos y cache hit a un diagnóstico serializable.

    Args:
        answer: Respuesta de dominio cuyo diagnóstico se quiere exponer.

    Returns:
        Diagnóstico listo para serializar como JSON.

    Raises:
        ValueError: Si ``answer.diagnostics`` es ``None``.
    """
    if answer.diagnostics is None:
        raise ValueError("The answer does not include diagnostics")

    return AnswerDiagnosticsResponse(
        conversation_context_mode=answer.diagnostics.conversation_context_mode,
        answer_cache_mode=answer.diagnostics.answer_cache_mode,
        cache_hit=answer.diagnostics.cache_hit,
        cache_hit_source=answer.diagnostics.cache_hit_source,
        resolved_query=answer.resolved_query,
        stage_latencies_ms=answer.diagnostics.stage_latencies_ms,
        retrieved_chunks=[
            RetrievedChunkResponse(
                id=retrieved_chunk.chunk.id,
                content=retrieved_chunk.chunk.content,
                similarity_score=retrieved_chunk.similarity_score,
                metadata=retrieved_chunk.chunk.metadata,
            )
            for retrieved_chunk in answer.context
        ],
    )
