from fastapi import APIRouter, HTTPException, Depends, status
from typing import Dict, Any
from ...models.book_chunk import IngestionRequest, IngestionResponse
from ...services.ingestion_service import IngestionService
from ...core.auth import verify_admin_api_key
from ...core.logging import get_logger
from ...core.config import settings
from ...core.qdrant_client import qdrant_manager
import time

router = APIRouter()
logger = get_logger(__name__)


@router.post("/ingest", response_model=IngestionResponse, summary="Ingest book content for RAG")
async def ingest_endpoint(
    ingestion_request: IngestionRequest,
    admin_api_key: str = Depends(verify_admin_api_key)
) -> IngestionResponse:
    """
    Process and store book content for retrieval.
    """
    start_time = time.time()

    try:
        logger.info(f"Starting ingestion for book: {ingestion_request.book_id}")

        # Initialize the Qdrant collection if it doesn't exist
        qdrant_manager.create_collection()

        # Process the ingestion using the IngestionService
        result = await IngestionService.ingest_book_content(
            book_id=ingestion_request.book_id,
            title=ingestion_request.title,
            content=ingestion_request.content,
            metadata=ingestion_request.metadata
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        # Update the processing time in the result
        result["processing_time_ms"] = processing_time_ms

        response = IngestionResponse(
            status=result["status"],
            book_id=result["book_id"],
            chunks_processed=result["chunks_processed"],
            processing_time_ms=processing_time_ms
        )

        logger.info(f"Successfully ingested book {ingestion_request.book_id} with {result['chunks_processed']} chunks in {processing_time_ms}ms")
        return response

    except ValueError as ve:
        logger.warning(f"Validation error during ingestion: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid ingestion data: {str(ve)}"
        )
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during book ingestion"
        )


@router.get("/ingest/status/{book_id}", summary="Get ingestion status for a book")
async def get_ingestion_status(
    book_id: str,
    admin_api_key: str = Depends(verify_admin_api_key)
) -> Dict[str, Any]:
    """
    Get the ingestion status for a specific book.
    """
    try:
        # In a real implementation, this would check the database
        # for ingestion progress/ completion status
        logger.info(f"Checking ingestion status for book: {book_id}")

        # Placeholder implementation
        # In real implementation, check actual status from database
        return {
            "book_id": book_id,
            "status": "completed",  # or "in_progress", "failed", "not_found"
            "chunks_processed": 0,  # Actual number from database
            "total_chunks": 0,      # Total expected chunks
            "progress": 100         # Percentage complete
        }
    except Exception as e:
        logger.error(f"Error getting ingestion status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving ingestion status"
        )


@router.delete("/ingest/{book_id}", summary="Remove a book's content from the system")
async def remove_book(
    book_id: str,
    admin_api_key: str = Depends(verify_admin_api_key)
) -> Dict[str, Any]:
    """
    Remove a book's content from the vector store and metadata.
    """
    try:
        logger.info(f"Removing book content for: {book_id}")

        # In a real implementation:
        # 1. Remove from Qdrant collection
        # 2. Remove metadata from Postgres
        # 3. Update any related records

        # Placeholder implementation
        return {
            "book_id": book_id,
            "status": "removed",
            "message": f"Book {book_id} has been removed from the system"
        }
    except Exception as e:
        logger.error(f"Error removing book: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while removing the book"
        )