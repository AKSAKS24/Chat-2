# app/dependencies.py
from typing import Generator
from .config import get_settings, Settings


def get_app_settings() -> Settings:
    """
    FastAPI dependency for injecting Settings into routes/services.
    Usage example in a router:

        from fastapi import Depends
        from ..dependencies import get_app_settings

        @router.get("/something")
        async def my_route(settings: Settings = Depends(get_app_settings)):
            return {"openai_key_present": settings.OPENAI_API_KEY is not None}
    """
    return get_settings()


# Example pattern for future shared resources (DB, vectorstore, etc.)
# def get_db() -> Generator[Session, None, None]:
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()
