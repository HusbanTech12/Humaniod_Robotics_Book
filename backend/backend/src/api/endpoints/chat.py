from fastapi import APIRouter, HTTPException, Depends, status
from typing import Dict, Any
import time
from ...models.query import QueryRequest, QueryResponse
from ...services.query_service import QueryService
from ...services.rag_service import RAGService
from ...core.auth import verify_api_key
from ...core.logging import get_logger
from ...core.config import settings

router = APIRouter()
logger = get_logger(__name__)


@router.post("/chat", response_model=QueryResponse, summary="Submit a query and receive a response with citations")
async def chat_endpoint(
    query_request: QueryRequest,
    api_key: str = Depends(verify_api_key)
) -> QueryResponse:
    """
    Process user query using RAG and return response with proper citations.
    Supports both full-book and selected-text modes with strict isolation.
    """
    start_time = time.time()

    try:
        logger.info(f"Received query for book: {query_request.book_id}, mode: {query_request.mode}")

        # Create internal query object from request
        query = QueryService.create_query_from_request(query_request)

        # Add mode validation and logging for isolation
        if query.mode == QueryMode.selected_text:
            logger.info(f"Processing query in selected-text mode for book: {query.book_id}")
            # Validate selected-text mode requirements
            QueryService.validate_selected_text_mode(query)
        else:
            logger.info(f"Processing query in full-book mode for book: {query.book_id}")
            # Validate full-book mode requirements
            QueryService.validate_full_book_mode(query)

        # Process the query using RAG service with strict mode enforcement
        response = await RAGService.process_query_with_strict_mode_enforcement(query)

        # Calculate total processing time
        total_time_ms = int((time.time() - start_time) * 1000)

        # Format the response
        formatted_response = QueryResponse(
            response=response.content,
            citations=response.citations,  # This will be formatted by the CitationService
            query_id=response.query_id,
            response_id=response.id,
            latency_ms=response.latency_ms
        )

        logger.info(f"Successfully processed {query.mode} mode query in {total_time_ms}ms")
        return formatted_response

    except ValueError as ve:
        logger.warning(f"Validation error in chat endpoint: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(ve)}"
        )
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request"
        )


@router.post("/chat/stream", summary="Submit a query and receive a streaming response with citations")
async def chat_stream_endpoint(
    query_request: QueryRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Process user query using RAG and return streaming response with proper citations.
    This endpoint returns Server-Sent Events for streaming the response.
    """
    from fastapi import Response
    import asyncio
    from ...models.query import Query

    async def generate_stream():
        start_time = time.time()

        try:
            logger.info(f"Received streaming query for book: {query_request.book_id}, mode: {query_request.mode}")

            # Create internal query object from request
            query = QueryService.create_query_from_request(query_request)

            # For now, we'll simulate a streaming response since the actual implementation
            # would require more complex streaming logic with Cohere
            # In a real implementation, we would connect directly to Cohere's streaming API

            # Process the query using RAG service (non-streaming for now)
            response = await RAGService.process_query(query)

            # Simulate streaming by sending the response in chunks
            content = response.content
            chunk_size = 10  # characters per chunk for simulation

            for i in range(0, len(content), chunk_size):
                chunk_text = content[i:i+chunk_size]

                # Send chunk event
                yield f"data: {{" + f'"event": "chunk", "data": {{"content": "{chunk_text}", "is_final": {i+chunk_size >= len(content)}}}' + "}\n\n"
                await asyncio.sleep(0.01)  # Small delay to simulate streaming

            # Send citations event
            formatted_citations = []
            for citation in response.citations:
                formatted_citations.append({
                    'source_text': citation.source_text,
                    'location': citation.location.dict(),
                    'relevance_score': citation.relevance_score
                })

            yield f"data: {{" + f'"event": "citations", "data": {formatted_citations}' + "}\n\n"

            # Send completion event
            total_time_ms = int((time.time() - start_time) * 1000)
            yield f"data: {{" + f'"event": "done", "data": "{total_time_ms}"' + "}\n\n"

        except ValueError as ve:
            logger.warning(f"Validation error in streaming chat endpoint: {str(ve)}")
            yield f"data: {{" + f'"event": "error", "data": "Validation error: {str(ve)}"' + "}\n\n"
        except Exception as e:
            logger.error(f"Error in streaming chat endpoint: {str(e)}", exc_info=True)
            yield f"data: {{" + f'"event": "error", "data": "Internal server error"' + "}\n\n"

    return Response(
        content=generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )


# Add validation and error handling middleware specific to this endpoint
@router.post("/chat/validate", summary="Validate a query without processing it")
async def validate_query_endpoint(
    query_request: QueryRequest,
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Validate a query request without processing it.
    """
    try:
        # Validate the query
        is_valid = QueryService.validate_query(query_request)

        return {
            "valid": is_valid,
            "message": "Query is valid" if is_valid else "Query validation failed",
            "book_id": query_request.book_id
        }
    except ValueError as ve:
        logger.warning(f"Query validation failed: {str(ve)}")
        return {
            "valid": False,
            "message": str(ve),
            "book_id": query_request.book_id
        }
    except Exception as e:
        logger.error(f"Error during query validation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during validation"
        )