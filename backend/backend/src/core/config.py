from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # API Configuration
    api_key: str = os.getenv("API_KEY", "default_api_key")
    admin_api_key: str = os.getenv("ADMIN_API_KEY", "default_admin_api_key")

    # Cohere Configuration
    cohere_api_key: str = os.getenv("COHERE_API_KEY", "")

    # Qdrant Configuration
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY")
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))

    # Database Configuration
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./test.db")
    neon_database_url: Optional[str] = os.getenv("NEON_DATABASE_URL")

    # Application Configuration
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    port: int = int(os.getenv("PORT", "8000"))
    host: str = os.getenv("HOST", "0.0.0.0")

    # Qdrant Collection Configuration
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "book_content")

    class Config:
        env_file = ".env"


# Global settings instance
settings = Settings()