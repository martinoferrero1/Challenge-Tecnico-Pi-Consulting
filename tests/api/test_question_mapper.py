from app.api.mappers.question_mapper import map_question_request_to_entity
from app.api.schemas.question_schema import QuestionRequest


def test_question_mapper_builds_user_question_from_request() -> None:
    payload = QuestionRequest(
        user_name=" John Doe ",
        question=" Quien es Zara? ",
    )

    question = map_question_request_to_entity(payload)

    assert question.user_name == "John Doe"
    assert question.content == "Quien es Zara?"
