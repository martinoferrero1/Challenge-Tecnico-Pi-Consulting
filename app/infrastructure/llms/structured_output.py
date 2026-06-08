import asyncio
from typing import Any, TypeVar

from pydantic import BaseModel


StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)


async def invoke_structured(
    client: Any,
    prompt: str,
    output_schema: type[StructuredOutput],
) -> StructuredOutput:
    """Invoca un cliente con salida estructurada y valida el resultado.

    Args:
        client: Cliente LLM compatible con ``with_structured_output``.
        prompt: Prompt enviado al cliente estructurado.
        output_schema: Schema Pydantic esperado para validar la respuesta.

    Returns:
        Instancia validada de ``output_schema``.

    Raises:
        NotImplementedError: Si ``client`` no soporta structured output.
    """
    if not hasattr(client, "with_structured_output"):
        raise NotImplementedError("This LLM client does not support structured output")

    structured_client = client.with_structured_output(output_schema)

    if hasattr(structured_client, "ainvoke"):
        response = await structured_client.ainvoke(prompt)
    else:
        response = await asyncio.to_thread(structured_client.invoke, prompt)

    if isinstance(response, output_schema):
        return response
    if isinstance(response, dict):
        return output_schema.model_validate(response)

    return output_schema.model_validate(response)
