from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import Generator
import asyncpg
from .config import settings
import logging

logger = logging.getLogger(__name__)

# Synchronous engine for general use
SQLALCHEMY_DATABASE_URL = settings.database_url
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


# Async database connection for Postgres
async def get_async_db_connection():
    """
    Get an async database connection to Postgres/Neon.
    """
    if settings.neon_database_url:
        connection = await asyncpg.connect(dsn=settings.neon_database_url)
        return connection
    else:
        logger.warning("No Neon database URL configured")
        return None


# Dependency for FastAPI
def get_db() -> Generator:
    """
    Dependency function for getting database sessions in FastAPI.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Initialize the database tables
def init_db():
    """
    Initialize the database tables based on the defined models.
    """
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully")