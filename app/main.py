from fastapi import FastAPI

from app.api.routes.health_routes import router as health_router
from app.core.config import settings


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)

app.include_router(health_router, prefix=settings.api_prefix)
