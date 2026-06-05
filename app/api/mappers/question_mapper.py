from app.api.schemas.question_schema import QuestionRequest
from app.domain.entities.question import UserQuestion


def map_question_request_to_entity(payload: QuestionRequest) -> UserQuestion:
    return UserQuestion(
        user_name=payload.user_name,
        content=payload.question,
    )
