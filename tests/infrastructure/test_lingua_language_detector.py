from types import SimpleNamespace

from app.infrastructure.language_detectors.lingua_language_detector import (
    LinguaLanguageDetector,
)


class FakeDetector:
    def __init__(self, confidence_values) -> None:
        self.confidence_values = confidence_values
        self.texts: list[str] = []

    def compute_language_confidence_values(self, text: str):
        self.texts.append(text)
        return self.confidence_values


def test_lingua_language_detector_returns_best_language() -> None:
    detector = FakeDetector(
        [
            SimpleNamespace(
                language=SimpleNamespace(name="ENGLISH"),
                value=0.83,
            ),
            SimpleNamespace(
                language=SimpleNamespace(name="SPANISH"),
                value=0.97,
            ),
        ]
    )
    language_detector = LinguaLanguageDetector(detector=detector)

    detected_language = language_detector.detect("Quien es Zara?")

    assert detected_language is not None
    assert detected_language.name == "Spanish"
    assert detected_language.confidence == 0.97


def test_lingua_language_detector_ignores_blank_text() -> None:
    detector = FakeDetector([])
    language_detector = LinguaLanguageDetector(detector=detector)

    assert language_detector.detect("  ") is None
    assert detector.texts == []
