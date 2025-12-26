from fastapi import APIRouter
from typing import Dict, Any
from ...core.config import settings
from ...core.qdrant_client import qdrant_manager
from ...core.cohere_client import cohere_manager
from ...core.logging import get_logger
import asyncio
import time

router = APIRouter()
logger = get_logger(__name__)


@router.get("/health", summary="Health check endpoint")
async def health_check() -> Dict[str, Any]:
    """
    Check the health status of the service and its dependencies.
    """
    start_time = time.time()

    try:
        # Check internal health
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "1.0.0",
            "response_time_ms": int((time.time() - start_time) * 1000),
            "services": {
                "app": "available"
            }
        }

        # Check if we can access Cohere
        try:
            # Just verify we can access the client (don't make an actual API call to save resources)
            if cohere_manager.client:
                health_status["services"]["cohere"] = "available"
            else:
                health_status["services"]["cohere"] = "unavailable"
        except Exception:
            health_status["services"]["cohere"] = "unavailable"

        # Check if we can access Qdrant
        try:
            # Try to get collection info to verify connection
            info = qdrant_manager.get_collection_info()
            health_status["services"]["qdrant"] = "available"
        except Exception:
            health_status["services"]["qdrant"] = "unavailable"

        logger.info("Health check completed successfully")
        return health_status

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e),
            "response_time_ms": int((time.time() - start_time) * 1000)
        }


@router.get("/ready", summary="Readiness check endpoint")
async def readiness_check() -> Dict[str, Any]:
    """
    Check if the service is ready to handle requests.
    """
    try:
        # For readiness, we might check more specific things
        # For now, just return that we're ready
        return {
            "status": "ready",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        return {
            "status": "not ready",
            "timestamp": time.time(),
            "error": str(e)
        }