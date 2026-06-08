from typing import Protocol

from app.domain.entities.language import DetectedLanguage


class LanguageDetectorPort(Protocol):
    """Contrato para detectar idioma de textos de usuario o respuestas."""

    def detect(self, text: str) -> DetectedLanguage | None:
        """Detecta el idioma o devuelve None si no hay confianza suficiente.

        Args:
            text: Texto sobre el cual se intenta detectar idioma.

        Returns:
            Idioma detectado con confianza o ``None`` si no se pudo inferir.
        """
        raise NotImplementedError
