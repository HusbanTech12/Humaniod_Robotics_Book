from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator, Dict, Any
import time
from ...models.query import QueryRequest
from ...services.query_service import QueryService
from ...services.streaming_service import StreamingService
from ...core.auth import verify_api_key
from ...core.logging import get_logger
from ...services.rag_service import RAGService

router = APIRouter()
logger = get_logger(__name__)


@router.post("/chat/stream", summary="Submit a query and receive a streaming response with citations")
async def stream_endpoint(
    request: Request,
    query_request: QueryRequest,
    api_key: str = Depends(verify_api_key)
) -> StreamingResponse:
    """
    Process user query using RAG and return streaming response with proper citations.
    Returns Server-Sent Events for streaming the response.
    """
    start_time = time.time()

    async def generate_stream() -> AsyncGenerator[str, None]:
        try:
            logger.info(f"Received streaming query for book: {query_request.book_id}, mode: {query_request.mode}")

            # Create internal query object from request
            query = QueryService.create_query_from_request(query_request)

            # Process the query using RAG service with streaming
            async for event in RAGService.process_query_streaming(query):
                # Format the event as Server-Sent Event
                yield f"data: {event}\n\n"

            # Calculate and send completion time
            total_time_ms = int((time.time() - start_time) * 1000)
            completion_event = {
                "event": "done",
                "data": str(total_time_ms)
            }
            yield f"data: {completion_event}\n\n"

        except Exception as e:
            logger.error(f"Error in streaming endpoint: {str(e)}", exc_info=True)
            error_event = {
                "event": "error",
                "data": {"error": str(e)}
            }
            yield f"data: {error_event}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )


@router.get("/stream/health", summary="Health check for streaming service")
async def streaming_health() -> Dict[str, Any]:
    """
    Health check endpoint specifically for the streaming service.
    """
    try:
        return {
            "status": "healthy",
            "service": "streaming",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Streaming health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "streaming",
            "error": str(e),
            "timestamp": time.time()
        }