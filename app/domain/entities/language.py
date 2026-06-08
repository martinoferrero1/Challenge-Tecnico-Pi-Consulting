from dataclasses import dataclass


@dataclass(frozen=True)
class DetectedLanguage:
    """Idioma detectado junto con su confianza.

    Atributos:
        name: Nombre del idioma detectado.
        confidence: Confianza normalizada entre 0 y 1.
    """

    name: str
    confidence: float

    def __post_init__(self) -> None:
        """Valida ``name`` y ``confidence``.

        Raises:
            ValueError: Si ``name`` queda vacío o si ``confidence`` queda fuera
                del rango 0 a 1.
        """
        name = self.name.strip()

        if not name:
            raise ValueError("The language name cannot be empty")
        if not 0 <= self.confidence <= 1:
            raise ValueError("The language confidence must be between 0 and 1")

        object.__setattr__(self, "name", name)
