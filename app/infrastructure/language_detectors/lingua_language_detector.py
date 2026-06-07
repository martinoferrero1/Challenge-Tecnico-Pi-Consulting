from typing import Any

from app.domain.entities.language import DetectedLanguage


class LinguaLanguageDetector:
    def __init__(self, detector: Any | None = None) -> None:
        self.detector = detector

    def detect(self, text: str) -> DetectedLanguage | None:
        if not text.strip():
            return None

        confidence_values = self._get_detector().compute_language_confidence_values(
            text
        )
        #print(f"detected languages and confidences: {confidence_values}")
        if not confidence_values:
            return None

        best_value = max(confidence_values, key=lambda value: value.value)

        return DetectedLanguage(
            name=self._language_name(best_value.language),
            confidence=float(best_value.value),
        )

    def _get_detector(self) -> Any:
        if self.detector is None:
            from lingua import LanguageDetectorBuilder

            self.detector = LanguageDetectorBuilder.from_all_languages().build()

        return self.detector

    def _language_name(self, language: Any) -> str:
        raw_name = getattr(language, "name", str(language))
        return raw_name.replace("_", " ").title()
