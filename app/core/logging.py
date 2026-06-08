import logging


def configure_logging() -> None:
    """Configura logging base para la API y servicios de aplicación."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
