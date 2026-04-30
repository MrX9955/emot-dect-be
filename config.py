"""
Application configuration using pydantic-settings.
Reads from environment variables only (no .env file in production).
"""
import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    mongodb_url: str = "mongodb://localhost:27017"
    database_name: str = "emotion_app"
    secret_key: str = "change-this-secret-key-in-production-32chars"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
        "populate_by_name": True,
    }

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]


def get_settings() -> Settings:
    # Read directly from env vars — bypass lru_cache so Render env vars always win
    return Settings(
        mongodb_url=os.environ.get("MONGODB_URL", "mongodb://localhost:27017"),
        database_name=os.environ.get("DATABASE_NAME", "emotion_app"),
        secret_key=os.environ.get("SECRET_KEY", "change-this-secret-key-in-production-32chars"),
        algorithm=os.environ.get("ALGORITHM", "HS256"),
        access_token_expire_minutes=int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "60")),
        cors_origins=os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"),
    )
