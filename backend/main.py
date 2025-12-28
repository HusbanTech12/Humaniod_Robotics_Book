from src.api.main import app
from src.core.config import settings
from src.core.database import init_db
from src.core.qdrant_client import qdrant_manager
from src.core.logging import get_logger
import uvicorn
import asyncio

logger = get_logger(__name__)


def main():
    """
    Main entry point for the RAG Chatbot API.
    """
    logger.info("Starting RAG Chatbot API...")

    # Initialize database
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise

    # Initialize Qdrant collection
    try:
        qdrant_manager.create_collection()
        logger.info("Qdrant collection created/verified successfully")
    except Exception as e:
        logger.error(f"Failed to create Qdrant collection: {str(e)}")
        raise

    # Run the application
    logger.info(f"Starting server on {settings.host}:{settings.port}")
    uvicorn.run(
        "main:app",  # Point to this file's app instance
        host=settings.host,
        port=settings.port,
        reload=True,  # Set to False in production
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    main()