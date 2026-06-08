import logging
from typing import Any

from app.domain.entities.language import DetectedLanguage


logger = logging.getLogger(__name__)


class LinguaLanguageDetector:
    """Detector de idioma basado en Lingua."""

    def __init__(self, detector: Any | None = None) -> None:
        """Permite inyectar un detector para tests.

        Args:
            detector: Instancia compatible con Lingua. Si es ``None``, se crea
                perezosamente.
        """
        self.detector = detector

    def detect(self, text: str) -> DetectedLanguage | None:
        """Detecta idioma y confidence para texto no vacío.

        Args:
            text: Texto sobre el que se calcula confianza de idioma.

        Returns:
            Idioma con mayor confianza o ``None`` si ``text`` está vacío o el
            detector no devuelve valores.
        """
        if not text.strip():
            return None

        confidence_values = self._get_detector().compute_language_confidence_values(
            text
        )

        if not confidence_values:
            return None

        best_value = max(confidence_values, key=lambda value: value.value)

        logger.debug(
            "Best detected language: %s with confidence %.4f",
            best_value.language,
            best_value.value,
        )

        return DetectedLanguage(
            name=self._language_name(best_value.language),
            confidence=float(best_value.value),
        )

    def _get_detector(self) -> Any:
        """Construye perezosamente el detector Lingua.

        Returns:
            Instancia cacheada del detector Lingua.
        """
        if self.detector is None:
            from lingua import LanguageDetectorBuilder

            self.detector = LanguageDetectorBuilder.from_all_languages().build()

        return self.detector

    def _language_name(self, language: Any) -> str:
        """Normaliza el nombre devuelto por Lingua.

        Args:
            language: Objeto o valor de idioma devuelto por Lingua.

        Returns:
            Nombre legible, con guiones bajos reemplazados por espacios.
        """
        raw_name = getattr(language, "name", str(language))
        return raw_name.replace("_", " ").title()
