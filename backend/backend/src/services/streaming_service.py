from typing import AsyncGenerator, Dict, Any, List
from ..models.query import Query
from ..models.citation import Citation
from ..core.cohere_client import cohere_manager
from ..core.logging import get_logger
import asyncio
import json

logger = get_logger(__name__)


class StreamingService:
    """
    Service for handling streaming responses with Server-Sent Events.
    """

    @staticmethod
    async def generate_streaming_response(
        query: Query,
        context_documents: List[Dict[str, Any]]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate a streaming response for the given query and context.

        Args:
            query: The query to process
            context_documents: List of context documents to use

        Yields:
            Dictionary containing stream events (chunk, citations, done)
        """
        try:
            # Extract document texts for context
            document_texts = [doc['content'] for doc in context_documents if doc.get('content')]

            # For true streaming, we would connect directly to Cohere's streaming API
            # For now, we'll simulate streaming by processing the response in chunks

            # First, generate the full response
            result = cohere_manager.generate_response(
                prompt=query.text,
                documents=document_texts if document_texts else None,
                model="command-r-plus"
            )

            full_response = result.get('text', '')

            # Stream the response in chunks
            chunk_size = 20  # characters per chunk
            for i in range(0, len(full_response), chunk_size):
                chunk_text = full_response[i:i + chunk_size]

                yield {
                    "event": "chunk",
                    "data": {
                        "content": chunk_text,
                        "is_final": (i + chunk_size) >= len(full_response)
                    }
                }

                # Small delay to simulate real streaming
                await asyncio.sleep(0.01)

            # Send citations if available
            raw_citations = result.get('citations', [])
            if raw_citations:
                formatted_citations = []
                for citation in raw_citations:
                    formatted_citations.append({
                        'text': citation.get('text', ''),
                        'start': citation.get('start', 0),
                        'end': citation.get('end', 0),
                        'document_ids': citation.get('document_ids', [])
                    })

                yield {
                    "event": "citations",
                    "data": formatted_citations
                }

            # Send completion event
            yield {
                "event": "done",
                "data": ""
            }

        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}", exc_info=True)
            yield {
                "event": "error",
                "data": {"error": str(e)}
            }

    @staticmethod
    async def format_streaming_chunk(content: str, is_final: bool = False) -> str:
        """
        Format a chunk for Server-Sent Events.

        Args:
            content: The chunk content
            is_final: Whether this is the final chunk

        Returns:
            Formatted string for SSE
        """
        chunk_data = {
            "content": content,
            "is_final": is_final
        }

        return f"data: {json.dumps({'event': 'chunk', 'data': chunk_data})}\n\n"

    @staticmethod
    async def format_streaming_citations(citations: List[Citation]) -> str:
        """
        Format citations for Server-Sent Events.

        Args:
            citations: List of citations

        Returns:
            Formatted string for SSE
        """
        citations_data = []
        for citation in citations:
            citations_data.append({
                "source_text": citation.source_text,
                "location": citation.location.dict(),
                "relevance_score": citation.relevance_score
            })

        return f"data: {json.dumps({'event': 'citations', 'data': citations_data})}\n\n"

    @staticmethod
    async def format_streaming_done(total_time_ms: int) -> str:
        """
        Format the completion event for Server-Sent Events.

        Args:
            total_time_ms: Total processing time in milliseconds

        Returns:
            Formatted string for SSE
        """
        return f"data: {json.dumps({'event': 'done', 'data': str(total_time_ms)})}\n\n"

    @staticmethod
    async def format_streaming_error(error_message: str) -> str:
        """
        Format an error event for Server-Sent Events.

        Args:
            error_message: The error message

        Returns:
            Formatted string for SSE
        """
        error_data = {
            "error": error_message
        }

        return f"data: {json.dumps({'event': 'error', 'data': error_data})}\n\n"