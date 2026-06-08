from fastapi import FastAPI

from app.api.routes.health_routes import router as health_router
from app.api.routes.question_routes import router as question_router
from app.core.config import settings
from app.core.logging import configure_logging


configure_logging()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)

app.include_router(health_router, prefix=settings.api_prefix)
app.include_router(question_router, prefix=settings.api_prefix)
