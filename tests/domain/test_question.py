import pytest

from app.domain.entities.question import UserQuestion


def test_user_question_strips_user_name_and_content() -> None:
    question = UserQuestion(
        user_name=" John Doe ",
        content=" Quien es Zara? ",
    )

    assert question.user_name == "John Doe"
    assert question.content == "Quien es Zara?"


def test_user_question_normalizes_content() -> None:
    question = UserQuestion(
        user_name="John Doe",
        content="  Quien   es   Zara?  ",
    )

    assert question.normalized_content == "quien es zara?"


def test_user_question_rejects_blank_user_name() -> None:
    with pytest.raises(ValueError):
        UserQuestion(user_name=" ", content="Quien es Zara?")


def test_user_question_rejects_blank_content() -> None:
    with pytest.raises(ValueError):
        UserQuestion(user_name="John Doe", content=" ")
