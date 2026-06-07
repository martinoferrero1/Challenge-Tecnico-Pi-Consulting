from dataclasses import dataclass


@dataclass(frozen=True)
class DetectedLanguage:
    name: str
    confidence: float

    def __post_init__(self) -> None:
        name = self.name.strip()

        if not name:
            raise ValueError("The language name cannot be empty")
        if not 0 <= self.confidence <= 1:
            raise ValueError("The language confidence must be between 0 and 1")

        object.__setattr__(self, "name", name)
