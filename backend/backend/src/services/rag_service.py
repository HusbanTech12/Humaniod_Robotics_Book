from typing import List, Dict, Any, Optional, AsyncGenerator
from ..models.query import Query, QueryMode
from ..models.response import Response as ResponseModel
from ..models.citation import Citation
from .query_service import QueryService
from .citation_service import CitationService
from ..core.cohere_client import cohere_manager
from ..core.qdrant_client import qdrant_manager
from ..core.postgres_client import postgres_manager
from ..core.logging import get_logger
from uuid import uuid4
import time
import asyncio

logger = get_logger(__name__)


class RAGService:
    """
    Service for handling Retrieval-Augmented Generation operations.
    """

    @staticmethod
    async def process_query(query: Query) -> ResponseModel:
        """
        Process a query using RAG methodology with strict dual-mode enforcement.

        Args:
            query: The query to process

        Returns:
            A ResponseModel with the generated response and citations
        """
        start_time = time.time()

        try:
            # Validate dual-mode enforcement
            await RAGService.validate_dual_mode_enforcement(query)

            # Determine the context based on the query mode with strict separation
            if query.mode == QueryMode.selected_text:
                # In selected-text mode, use ONLY the provided text directly, no vector retrieval
                if not query.selected_text or not query.selected_text.strip():
                    raise ValueError("Selected text mode requires selected_text to be provided and non-empty")

                context_documents = [{
                    "content": query.selected_text,
                    "metadata": {"source": "selected_text", "mode": "selected_text"}
                }]
                logger.info("Using selected-text mode context with strict isolation")
            else:
                # In full-book mode, retrieve relevant documents from vector store, no direct text use
                if query.selected_text:
                    logger.warning("Selected text provided in full-book mode, ignoring it for strict mode enforcement")

                context_documents = await RAGService.retrieve_relevant_documents(query)
                logger.info(f"Retrieved {len(context_documents)} relevant documents for full-book mode")

            # Generate the response using the appropriate context
            response_text, raw_citations = await RAGService.generate_response_with_context(
                query.text,
                context_documents
            )

            # Create citations from the raw citations
            response_id = str(uuid4())
            citations = []
            for raw_citation in raw_citations:
                citation = CitationService.create_citation(
                    response_id=response_id,
                    source_text=raw_citation.get('text', ''),
                    location=raw_citation.get('location', {}),
                    relevance_score=raw_citation.get('relevance_score', 0.5)
                )
                citations.append(citation)

            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)

            # Create the response model
            response = ResponseModel(
                id=response_id,
                content=response_text,
                query_id=query.id,
                citations=citations,
                timestamp=time.time(),
                latency_ms=latency_ms,
                model_used="command-r-plus"  # Using the model that was actually used
            )

            logger.info(f"Generated response with {len(citations)} citations in {latency_ms}ms")
            return response

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            raise

    @staticmethod
    async def process_query_with_strict_mode_enforcement(query: Query) -> ResponseModel:
        """
        Process a query with additional strict mode enforcement checks.

        Args:
            query: The query to process

        Returns:
            A ResponseModel with the generated response and citations
        """
        start_time = time.time()

        try:
            # Additional validation for mode enforcement
            if query.mode == QueryMode.selected_text:
                if not query.selected_text or not query.selected_text.strip():
                    raise ValueError("Selected text mode requires selected_text to be provided and non-empty")

                # Ensure no vector retrieval happens in selected-text mode
                context_documents = [{
                    "content": query.selected_text,
                    "metadata": {"source": "selected_text", "mode": "selected_text", "strict_mode": True}
                }]
                logger.info("Processing in selected-text mode with strict enforcement")
            else:
                # Ensure selected text is ignored in full-book mode
                if query.selected_text:
                    logger.info("Ignoring selected_text in full-book mode as per strict mode enforcement")

                context_documents = await RAGService.retrieve_relevant_documents(query)
                logger.info(f"Processing in full-book mode, retrieved {len(context_documents)} documents")

            # Generate the response using the appropriately isolated context
            response_text, raw_citations = await RAGService.generate_response_with_context(
                query.text,
                context_documents
            )

            # Create citations from the raw citations
            response_id = str(uuid4())
            citations = []
            for raw_citation in raw_citations:
                citation = CitationService.create_citation(
                    response_id=response_id,
                    source_text=raw_citation.get('text', ''),
                    location=raw_citation.get('location', {}),
                    relevance_score=raw_citation.get('relevance_score', 0.5)
                )
                citations.append(citation)

            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)

            # Create the response model
            response = ResponseModel(
                id=response_id,
                content=response_text,
                query_id=query.id,
                citations=citations,
                timestamp=time.time(),
                latency_ms=latency_ms,
                model_used="command-r-plus"
            )

            logger.info(f"Generated response with strict mode enforcement in {latency_ms}ms")
            return response

        except Exception as e:
            logger.error(f"Error processing query with strict mode enforcement: {str(e)}", exc_info=True)
            raise

    @staticmethod
    async def retrieve_relevant_documents(query: Query, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from the vector store based on the query.

        Args:
            query: The query to use for retrieval
            top_k: Number of documents to retrieve

        Returns:
            List of relevant documents with metadata
        """
        try:
            # Generate embedding for the query
            embeddings = cohere_manager.embed_text([query.text])
            query_embedding = embeddings[0]  # Get the first (and only) embedding

            # Search in Qdrant
            search_results = qdrant_manager.search_vectors(
                vector=query_embedding,
                limit=top_k
            )

            # Format results
            documents = []
            for result in search_results:
                payload = result.get('payload', {})
                documents.append({
                    'id': result.get('id'),
                    'content': payload.get('content', ''),
                    'metadata': payload.get('metadata', {}),
                    'relevance_score': result.get('score', 0.0)
                })

            logger.info(f"Retrieved {len(documents)} relevant documents from vector store")
            return documents

        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}", exc_info=True)
            raise

    @staticmethod
    async def retrieve_relevant_documents_for_streaming(query: Query, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from the vector store for streaming response.

        Args:
            query: The query to use for retrieval
            top_k: Number of documents to retrieve

        Returns:
            List of relevant documents with metadata
        """
        try:
            # Generate embedding for the query
            embeddings = cohere_manager.embed_text([query.text])
            query_embedding = embeddings[0]  # Get the first (and only) embedding

            # Search in Qdrant
            search_results = qdrant_manager.search_vectors(
                vector=query_embedding,
                limit=top_k
            )

            # Format results
            documents = []
            for result in search_results:
                payload = result.get('payload', {})
                documents.append({
                    'id': result.get('id'),
                    'content': payload.get('content', ''),
                    'metadata': payload.get('metadata', {}),
                    'relevance_score': result.get('score', 0.0)
                })

            logger.info(f"Retrieved {len(documents)} relevant documents for streaming from vector store")
            return documents

        except Exception as e:
            logger.error(f"Error retrieving documents for streaming: {str(e)}", exc_info=True)
            raise

    @staticmethod
    async def generate_response_with_context(
        query_text: str,
        context_documents: List[Dict[str, Any]],
        model: str = "command-r-plus"
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Generate a response using the provided context documents.

        Args:
            query_text: The original query text
            context_documents: List of context documents to use
            model: The model to use for generation

        Returns:
            Tuple of (generated response text, list of raw citations)
        """
        try:
            # Extract document texts for context
            document_texts = [doc['content'] for doc in context_documents if doc.get('content')]

            # Generate the response using Cohere
            result = cohere_manager.generate_response(
                prompt=query_text,
                documents=document_texts if document_texts else None,
                model=model
            )

            response_text = result.get('text', '')
            raw_citations = []

            # Process citations if they exist
            if result.get('citations'):
                for citation in result.get('citations', []):
                    # Find the source document for this citation
                    doc_idx = citation.get('document_ids', [0])[0] if citation.get('document_ids') else 0
                    source_doc = context_documents[doc_idx] if doc_idx < len(context_documents) else {}

                    raw_citations.append({
                        'text': citation.get('text', ''),
                        'location': source_doc.get('metadata', {}),
                        'relevance_score': source_doc.get('relevance_score', 0.5)
                    })

            logger.info(f"Generated response with {len(raw_citations)} citations using model {model}")
            return response_text, raw_citations

        except Exception as e:
            logger.error(f"Error generating response with context: {str(e)}", exc_info=True)
            raise

    @staticmethod
    async def validate_dual_mode_enforcement(query: Query) -> bool:
        """
        Validate that dual-mode enforcement is working correctly with comprehensive logging.

        Args:
            query: The query to validate

        Returns:
            True if mode enforcement is correct, raises ValueError otherwise
        """
        if query.mode == QueryMode.selected_text:
            if not query.selected_text or not query.selected_text.strip():
                logger.error(f"Mode validation failed: Selected text mode requires selected_text to be provided. Query ID: {query.id}")
                raise ValueError("Selected text mode requires selected_text to be provided")

            logger.info(f"Mode isolation validation passed: Processing in selected-text mode. Query ID: {query.id}")
            logger.debug(f"Selected text length: {len(query.selected_text) if query.selected_text else 0} characters. Query ID: {query.id}")
            return True
        elif query.mode == QueryMode.full_book:
            if query.selected_text:
                logger.warning(f"In full-book mode but selected_text was provided. It will be ignored for strict isolation. Query ID: {query.id}")
            logger.info(f"Mode isolation validation passed: Processing in full-book mode. Query ID: {query.id}")
            return True
        else:
            logger.error(f"Unknown query mode received: {query.mode}. Query ID: {query.id}")
            raise ValueError(f"Unknown query mode: {query.mode}")

    @staticmethod
    async def process_selected_text_mode(query: Query) -> ResponseModel:
        """
        Process a query specifically in selected-text mode.

        Args:
            query: The query to process (must be in selected-text mode)

        Returns:
            A ResponseModel with the generated response
        """
        if query.mode != QueryMode.selected_text:
            raise ValueError("This method should only be called for selected-text mode queries")

        # Validate the query
        QueryService.validate_selected_text_mode(query)

        start_time = time.time()

        # Use the selected text as the context
        context_documents = [{
            "content": query.selected_text,
            "metadata": {"source": "selected_text", "mode": "selected_text"}
        }]

        # Generate the response
        response_text, raw_citations = await RAGService.generate_response_with_context(
            query.text,
            context_documents
        )

        # Create citations
        response_id = str(uuid4())
        citations = []
        for raw_citation in raw_citations:
            citation = CitationService.create_citation(
                response_id=response_id,
                source_text=raw_citation.get('text', ''),
                location=raw_citation.get('location', {}),
                relevance_score=raw_citation.get('relevance_score', 0.5)
            )
            citations.append(citation)

        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)

        # Create the response model
        response = ResponseModel(
            id=response_id,
            content=response_text,
            query_id=query.id,
            citations=citations,
            timestamp=time.time(),
            latency_ms=latency_ms,
            model_used="command-r-plus"
        )

        logger.info(f"Processed selected-text query in {latency_ms}ms")
        return response

    @staticmethod
    async def process_query_streaming(query: Query) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a query and yield streaming results.

        Args:
            query: The query to process

        Yields:
            Dictionary containing stream events (chunk, citations, done)
        """
        start_time = time.time()

        try:
            # Determine the context based on the query mode
            if query.mode == QueryMode.selected_text:
                # In selected-text mode, use the provided text directly
                context_documents = [{"content": query.selected_text, "metadata": {"source": "selected_text"}}]
                logger.info("Using selected-text mode context for streaming")
            else:
                # In full-book mode, retrieve relevant documents from vector store
                context_documents = await RAGService.retrieve_relevant_documents(query, top_k=5)
                logger.info(f"Retrieved {len(context_documents)} relevant documents for streaming")

            # Extract document texts for context
            document_texts = [doc['content'] for doc in context_documents if doc.get('content')]

            # Generate streaming response using Cohere
            # For true streaming, we would use Cohere's streaming API directly
            # For now, we'll simulate streaming by processing the response in chunks
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

            # Process and send citations if available
            raw_citations = result.get('citations', [])
            if raw_citations:
                formatted_citations = []
                for citation in raw_citations:
                    # Find the source document for this citation
                    doc_idx = citation.get('document_ids', [0])[0] if citation.get('document_ids') else 0
                    source_doc = context_documents[doc_idx] if doc_idx < len(context_documents) else {}

                    formatted_citations.append({
                        'text': citation.get('text', ''),
                        'location': source_doc.get('metadata', {}),
                        'relevance_score': source_doc.get('relevance_score', 0.5)
                    })

                yield {
                    "event": "citations",
                    "data": formatted_citations
                }

            # Send completion event
            total_time_ms = int((time.time() - start_time) * 1000)
            yield {
                "event": "done",
                "data": {"processing_time_ms": total_time_ms}
            }

        except Exception as e:
            logger.error(f"Error in streaming query processing: {str(e)}", exc_info=True)
            yield {
                "event": "error",
                "data": {"error": str(e)}
            }