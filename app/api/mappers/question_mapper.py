from app.api.schemas.question_schema import QuestionRequest
from app.api.schemas.question_schema import AnswerResponse
from app.domain.entities.answer import Answer
from app.domain.entities.question import UserQuestion


def map_question_request_to_entity(payload: QuestionRequest) -> UserQuestion: # mas alla de que en este caso el payload y la entidad son muy similares, hago esta capa de mapeo para desacoplar la API de la logica de negocio, y da flexibilidad para modificar cualquiera de las dos partes sin afectar a la otra
    return UserQuestion(
        user_name=payload.user_name,
        content=payload.question,
    )


def map_answer_to_response(answer: Answer) -> AnswerResponse:
    return AnswerResponse(
        user_name=answer.question.user_name,
        question=answer.question.content,
        answer=answer.content,
    )
