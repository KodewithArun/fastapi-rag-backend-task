"""
Core configuration settings for the application.
Uses Pydantic BaseSettings to read from environment variables.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # App Settings
    PROJECT_NAME: str
    VERSION: str
    API_V1_STR: str
    
    # Qdrant Vector DB Settings
    QDRANT_HOST: str
    QDRANT_PORT: int
    QDRANT_API_KEY: str | None = None

    # Redis Settings
    REDIS_HOST: str
    REDIS_PORT: int

    # Metadata PostgreSQL DB Settings
    DATABASE_URL: str

    # Model Settings (LLM & Embeddings)
    HUGGINGFACE_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=True)

settings = Settings()
