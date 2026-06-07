from typing import Protocol

from app.domain.entities.language import DetectedLanguage


class LanguageDetectorPort(Protocol):
    def detect(self, text: str) -> DetectedLanguage | None:
        raise NotImplementedError
