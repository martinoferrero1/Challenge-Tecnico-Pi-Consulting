from fastapi import APIRouter, HTTPException, status

from app.api.mappers.question_mapper import map_question_request_to_entity
from app.api.schemas.question_schema import AnswerResponse, QuestionRequest

router = APIRouter(prefix="/questions", tags=["questions"])


@router.post(
    "",
    response_model=AnswerResponse,
    responses={
        status.HTTP_501_NOT_IMPLEMENTED: {
            "description": "---",
        },
    },
)
async def answer_question(payload: QuestionRequest) -> AnswerResponse:
    map_question_request_to_entity(payload)

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Not implemented yet",
    )
