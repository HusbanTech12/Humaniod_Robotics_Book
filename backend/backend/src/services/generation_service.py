from typing import List, Dict, Any, Optional
from ..models.query import Query
from ..core.cohere_client import cohere_manager
from ..core.logging import get_logger
import time

logger = get_logger(__name__)


class GenerationService:
    """
    Service for handling text generation using Cohere.
    """

    @staticmethod
    async def generate_response(
        query_text: str,
        context_documents: Optional[List[str]] = None,
        model: str = "command-r-plus",
        temperature: float = 0.3,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Generate a response to the query using the provided context.

        Args:
            query_text: The user's query
            context_documents: Optional list of context documents to ground the response
            model: The generation model to use
            temperature: Controls randomness of the response
            max_tokens: Maximum number of tokens to generate

        Returns:
            Dictionary containing the generated response and metadata
        """
        try:
            start_time = time.time()

            # Generate response using Cohere
            result = cohere_manager.generate_response(
                prompt=query_text,
                documents=context_documents,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Calculate generation time
            generation_time = time.time() - start_time

            response_data = {
                'text': result.get('text', ''),
                'model': result.get('model', model),
                'finish_reason': result.get('finish_reason', ''),
                'citations': result.get('citations', []),
                'generation_time': generation_time,
                'model_used': model
            }

            logger.info(f"Generated response using {model} in {generation_time:.3f}s")
            return response_data

        except Exception as e:
            logger.error(f"Error in generation service: {str(e)}", exc_info=True)
            raise

    @staticmethod
    async def generate_with_reranking(
        query_text: str,
        documents: List[str],
        model: str = "command-r-plus",
        temperature: float = 0.3,
        max_tokens: int = 1000,
        rerank_top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Generate a response with reranking of documents for better relevance.

        Args:
            query_text: The user's query
            documents: List of documents to consider
            model: The generation model to use
            temperature: Controls randomness of the response
            max_tokens: Maximum number of tokens to generate
            rerank_top_n: Number of top documents to keep after reranking

        Returns:
            Dictionary containing the generated response and metadata
        """
        try:
            start_time = time.time()

            # Rerank documents based on relevance to the query
            reranked_results = cohere_manager.rerank_documents(
                query=query_text,
                documents=documents,
                top_n=rerank_top_n
            )

            # Extract the top reranked documents
            reranked_documents = [
                result['document']['text'] if isinstance(result['document'], dict) else result['document']
                for result in reranked_results
            ]

            # Generate response with reranked documents
            result = cohere_manager.generate_response(
                prompt=query_text,
                documents=reranked_documents,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Calculate generation time
            generation_time = time.time() - start_time

            response_data = {
                'text': result.get('text', ''),
                'model': result.get('model', model),
                'finish_reason': result.get('finish_reason', ''),
                'citations': result.get('citations', []),
                'generation_time': generation_time,
                'reranked_count': len(reranked_documents),
                'model_used': model
            }

            logger.info(f"Generated response with reranking in {generation_time:.3f}s using {len(reranked_documents)} documents")
            return response_data

        except Exception as e:
            logger.error(f"Error in generation service with reranking: {str(e)}", exc_info=True)
            raise

    @staticmethod
    async def validate_generation_input(query_text: str, documents: Optional[List[str]] = None) -> bool:
        """
        Validate the inputs for generation.

        Args:
            query_text: The query text to validate
            documents: Optional documents to validate

        Returns:
            True if inputs are valid, raises ValueError otherwise
        """
        if not query_text or not query_text.strip():
            raise ValueError("Query text is required and cannot be empty")

        if len(query_text.strip()) > 1000:
            raise ValueError("Query text exceeds maximum length of 1000 characters")

        if documents is not None:
            if not isinstance(documents, list):
                raise ValueError("Documents must be a list of strings")

            for i, doc in enumerate(documents):
                if not isinstance(doc, str):
                    raise ValueError(f"Document at index {i} must be a string")
                if not doc.strip():
                    raise ValueError(f"Document at index {i} cannot be empty")

        return True

    @staticmethod
    async def format_generation_prompt(query: Query, context: str = "") -> str:
        """
        Format the prompt for generation with proper context and instructions.

        Args:
            query: The query object
            context: Additional context to include in the prompt

        Returns:
            Formatted prompt string
        """
        # Create a system message that emphasizes citation and grounding
        system_message = (
            "You are a helpful assistant that answers questions based on provided book content. "
            "Always provide specific citations to the source material when answering. "
            "If the information is not available in the provided context, clearly state that the information is not in the source material. "
            "Be accurate, concise, and directly address the user's question."
        )

        # Combine system message with context and query
        if context:
            prompt = f"{system_message}\n\nContext: {context}\n\nQuestion: {query.text}\n\nAnswer:"
        else:
            prompt = f"{system_message}\n\nQuestion: {query.text}\n\nAnswer:"

        return prompt

    @staticmethod
    async def extract_citations_from_response(response_text: str, source_documents: List[str]) -> List[Dict[str, Any]]:
        """
        Extract citations from the generated response by identifying references to source documents.

        Args:
            response_text: The generated response text
            source_documents: The documents that were used to generate the response

        Returns:
            List of citation dictionaries
        """
        citations = []

        # Simple approach: Look for portions of source documents in the response
        # This is a simplified implementation - in practice, you'd use more sophisticated NLP
        for i, doc in enumerate(source_documents):
            if len(doc) > 50:  # Only consider non-trivial documents
                # Look for fragments of the document in the response
                doc_preview = doc[:200]  # First 200 chars as a simple identifier
                if doc_preview[:50] in response_text:  # Check if first 50 chars appear in response
                    citations.append({
                        'text': doc_preview,
                        'document_index': i,
                        'relevance': 0.8  # High relevance since it's directly referenced
                    })

        logger.info(f"Extracted {len(citations)} potential citations from response")
        return citations