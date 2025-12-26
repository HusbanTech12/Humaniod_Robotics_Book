from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Callable
import logging
import time
from ..core.config import settings
from ..core.logging import get_logger
from ..core.database import init_db

# Create FastAPI app instance
app = FastAPI(
    title="RAG Chatbot API",
    description="API for the Retrieval-Augmented Generation Chatbot for Published Book",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Get logger for this module
logger = get_logger(__name__)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # This will be configured more restrictively in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Add custom headers for API key authentication
    # expose_headers=["Access-Control-Allow-Origin"]
)


# Add custom middleware for logging and timing
@app.middleware("http")
async def log_requests(request: dict, call_next: Callable):
    """
    Middleware to log incoming requests and their response times.
    """
    start_time = time.time()

    # Log the request
    logger.info(f"Request: {request.method} {request.url}")

    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(f"Request failed: {request.method} {request.url} - {str(e)}")
        raise

    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)

    logger.info(f"Response: {response.status_code} - Process time: {process_time:.3f}s")

    return response


# Add custom exception handler
@app.exception_handler(Exception)
async def custom_exception_handler(request, exc):
    """
    Custom exception handler for the API.
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# Include API routes
def include_routers():
    """
    Include all API routers after initialization to avoid circular imports.
    """
    # Import routers here to avoid circular imports
    from .endpoints.chat import router as chat_router
    from .endpoints.health import router as health_router
    from .endpoints.ingest import router as ingest_router
    from .endpoints.stream import router as stream_router

    # Include the routers
    app.include_router(chat_router, prefix="/v1", tags=["chat"])
    app.include_router(health_router, prefix="/v1", tags=["health"])
    app.include_router(ingest_router, prefix="/v1", tags=["ingestion"])
    app.include_router(stream_router, prefix="/v1", tags=["streaming"])


# Event handlers
@app.on_event("startup")
async def startup_event():
    """
    Actions to perform on application startup.
    """
    logger.info("Starting up RAG Chatbot API...")

    # Initialize database
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise

    # Initialize other services
    # (This is where you might initialize Qdrant, Cohere connections, etc.)

    logger.info("RAG Chatbot API startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Actions to perform on application shutdown.
    """
    logger.info("Shutting down RAG Chatbot API...")
    # Close any connections or clean up resources here
    logger.info("RAG Chatbot API shutdown complete")


# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint for the API.
    """
    return {
        "message": "RAG Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/v1/health"
    }


# Initialize routers after app is created
include_routers()