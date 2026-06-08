from dataclasses import dataclass


@dataclass(frozen=True)
class ExternalServiceError(Exception):
    """Error controlado para fallas de proveedores externos.

    Atributos:
        cause: Causa concreta reportada por el proveedor o adapter.
        message: Mensaje estable y seguro para exponer desde la API.
    """

    cause: str
    message: str = "The question could not be processed."

    @property
    def code(self) -> str:
        """Devuelve el código estable usado en respuestas HTTP.

        Returns:
            Código de error legible por clientes de la API.
        """
        return "question_processing_error"

    def to_detail(self) -> dict[str, str]:
        """Serializa el error para la respuesta de FastAPI.

        Returns:
            Diccionario con ``code``, ``message`` y ``cause``.
        """
        return {
            "code": self.code,
            "message": self.message,
            "cause": self.cause,
        }
