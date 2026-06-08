from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def health_check() -> dict[str, str]:
    """Devuelve el estado básico de disponibilidad de la API.

    Returns:
        Diccionario con ``status`` en ``ok`` cuando el proceso responde.
    """
    return {"status": "ok"}
