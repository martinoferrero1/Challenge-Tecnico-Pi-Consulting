from fastapi.testclient import TestClient

from app.api.dependencies import get_answer_question_use_case
from app.application.errors import ExternalServiceError
from app.domain.entities.answer import Answer
from app.domain.entities.answer import AnswerDiagnostics
from app.domain.entities.question import UserQuestion
from app.main import app


class FakeAnswerQuestionUseCase:
    def __init__(self) -> None:
        self.questions: list[UserQuestion] = []

    async def execute(self, question: UserQuestion) -> Answer:
        self.questions.append(question)
        return Answer(
            question=question,
            content="Zara es una empresa de moda.",
            context=(),
            diagnostics=AnswerDiagnostics(
                conversation_context_mode="disabled",
                answer_cache_mode="document_context",
                cache_hit=False,
                stage_latencies_ms={"total": 10.0},
            ),
        )


class FailingAnswerQuestionUseCase:
    async def execute(self, question: UserQuestion) -> Answer:
        raise ExternalServiceError(
            cause="LLM provider is unavailable",
        )


def test_answer_question_endpoint_uses_injected_use_case() -> None:
    use_case = FakeAnswerQuestionUseCase()
    previous_overrides = dict(app.dependency_overrides)
    app.dependency_overrides[get_answer_question_use_case] = lambda: use_case

    try:
        response = TestClient(app).post(
            "/api/questions",
            json={
                "user_name": " Ana ",
                "question": " Que es Zara? ",
            },
        )
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous_overrides)

    assert response.status_code == 200
    assert response.json() == {
        "user_name": "Ana",
        "question": "Que es Zara?",
        "answer": "Zara es una empresa de moda.",
    }
    assert use_case.questions == [
        UserQuestion(user_name="Ana", content="Que es Zara?"),
    ]


def test_answer_question_evaluation_endpoint_returns_diagnostics() -> None:
    use_case = FakeAnswerQuestionUseCase()
    previous_overrides = dict(app.dependency_overrides)
    app.dependency_overrides[get_answer_question_use_case] = lambda: use_case

    try:
        response = TestClient(app).post(
            "/api/questions/evaluation",
            json={
                "user_name": "Ana",
                "question": "Que es Zara?",
            },
        )
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous_overrides)

    assert response.status_code == 200
    assert response.json()["diagnostics"] == {
        "conversation_context_mode": "disabled",
        "answer_cache_mode": "document_context",
        "cache_hit": False,
        "cache_hit_source": None,
        "resolved_query": None,
        "stage_latencies_ms": {"total": 10.0},
        "retrieved_chunks": [],
    }


def test_answer_question_endpoint_returns_controlled_external_service_error() -> None:
    previous_overrides = dict(app.dependency_overrides)
    app.dependency_overrides[get_answer_question_use_case] = (
        lambda: FailingAnswerQuestionUseCase()
    )

    try:
        response = TestClient(app).post(
            "/api/questions",
            json={
                "user_name": "Ana",
                "question": "Que es Zara?",
            },
        )
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous_overrides)

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "code": "question_processing_error",
            "message": "The question could not be processed.",
            "cause": "LLM provider is unavailable",
        }
    }
