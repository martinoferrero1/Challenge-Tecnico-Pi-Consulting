from typing import Protocol, TypeVar

from pydantic import BaseModel


StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)


class LLMPort(Protocol):
    """Contrato de generación de texto y salidas estructuradas."""

    async def generate(self, prompt: str) -> str:
        """Genera texto libre a partir de un prompt.

        Args:
            prompt: Instrucciones completas enviadas al modelo.

        Returns:
            Texto generado por el LLM.
        """
        raise NotImplementedError

    async def generate_structured(
        self,
        prompt: str,
        output_schema: type[StructuredOutput],
    ) -> StructuredOutput:
        """Genera una respuesta validada contra un schema Pydantic.

        Args:
            prompt: Instrucciones completas enviadas al modelo.
            output_schema: Modelo Pydantic que debe cumplir la salida.

        Returns:
            Instancia de ``output_schema`` validada.
        """
        raise NotImplementedError
