from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_answer_question_use_case
from app.api.mappers.question_mapper import (
    map_answer_to_response,
    map_question_request_to_entity,
)
from app.api.schemas.question_schema import AnswerResponse, QuestionRequest
from app.application.use_cases.answer_question import AnswerQuestionUseCase

router = APIRouter(prefix="/questions", tags=["questions"])


@router.post(
    "",
    response_model=AnswerResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "Invalid question request",
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
    user_question = map_question_request_to_entity(payload)

    try:
        answer = await use_case.execute(user_question)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    return map_answer_to_response(answer)
