from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_answer_question_use_case
from app.api.mappers.question_mapper import (
    map_answer_to_evaluation_response,
    map_answer_to_response,
    map_question_request_to_entity,
)
from app.api.schemas.question_schema import (
    AnswerEvaluationResponse,
    AnswerResponse,
    QuestionRequest,
)
from app.application.errors import ExternalServiceError
from app.application.use_cases.answer_question import AnswerQuestionUseCase
from app.domain.entities.answer import Answer

router = APIRouter(prefix="/questions", tags=["questions"])


@router.post(
    "",
    response_model=AnswerResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "Invalid question request",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "External service unavailable",
        },
    },
)
async def answer_question(
    payload: QuestionRequest,
    use_case: Annotated[
        AnswerQuestionUseCase,
        Depends(get_answer_question_use_case),
    ],
) -> AnswerResponse:
    """Responde una pregunta con la salida pública del asistente.

    Args:
        payload: Body HTTP validado con ``user_name`` y ``question``.
        use_case: Caso de uso inyectado por FastAPI.

    Returns:
        Respuesta pública sin diagnósticos internos.
    """
    answer = await _execute_question(payload, use_case)
    return map_answer_to_response(answer)


@router.post(
    "/evaluation",
    response_model=AnswerEvaluationResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "Invalid question request",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "External service unavailable",
        },
    },
)
async def answer_question_for_evaluation(
    payload: QuestionRequest,
    use_case: Annotated[
        AnswerQuestionUseCase,
        Depends(get_answer_question_use_case),
    ],
) -> AnswerEvaluationResponse:
    """Responde una pregunta incluyendo diagnóstico para métricas offline.

    Args:
        payload: Body HTTP validado con ``user_name`` y ``question``.
        use_case: Caso de uso inyectado por FastAPI.

    Returns:
        Respuesta extendida con ``diagnostics`` para evaluación.
    """
    answer = await _execute_question(payload, use_case)
    return map_answer_to_evaluation_response(answer)


async def _execute_question(
    payload: QuestionRequest,
    use_case: AnswerQuestionUseCase,
) -> Answer:
    """Ejecuta el caso de uso y traduce errores de aplicación a HTTP.

    Args:
        payload: Body HTTP validado por el schema de entrada.
        use_case: Caso de uso encargado de resolver la pregunta.

    Returns:
        Respuesta de dominio generada por ``use_case``.

    Raises:
        HTTPException: Si el payload produce un error de dominio o si falla un
            servicio externo controlado.
    """
    user_question = map_question_request_to_entity(payload)

    try:
        return await use_case.execute(user_question)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except ExternalServiceError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=error.to_detail(),
        ) from error
