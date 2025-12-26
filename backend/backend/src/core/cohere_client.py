import cohere
from typing import List, Dict, Any, Optional
from .config import settings
from .secrets import SecretsManager
import logging

logger = logging.getLogger(__name__)


class CohereManager:
    """
    Manages interactions with Cohere API for embeddings and text generation.
    """

    def __init__(self):
        """
        Initialize the Cohere client with API key from settings.
        """
        api_key = SecretsManager.get_cohere_api_key()
        if not api_key:
            raise ValueError("Cohere API key not configured")

        self.client = cohere.Client(api_key)

    def embed_text(self, texts: List[str], model: str = "embed-multilingual-v3.0") -> List[List[float]]:
        """
        Generate embeddings for the provided texts using Cohere.

        Args:
            texts: List of texts to embed
            model: The embedding model to use (default: embed-multilingual-v3.0)

        Returns:
            List of embedding vectors
        """
        try:
            response = self.client.embed(
                texts=texts,
                model=model,
                input_type="search_document"  # Using search_document for book content
            )
            logger.info(f"Generated embeddings for {len(texts)} texts using model {model}")
            return [embedding for embedding in response.embeddings]
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def generate_response(
        self,
        prompt: str,
        documents: Optional[List[str]] = None,
        model: str = "command-r-plus",
        temperature: float = 0.3,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Generate a response using Cohere's language model with optional document grounding.

        Args:
            prompt: The input prompt for generation
            documents: Optional list of documents to ground the response in
            model: The generation model to use (default: command-r-plus)
            temperature: Controls randomness (0.0-1.0, default: 0.3)
            max_tokens: Maximum number of tokens to generate (default: 1000)

        Returns:
            Dictionary containing the generated response and metadata
        """
        try:
            if documents:
                # Use documents for grounded generation
                response = self.client.chat(
                    message=prompt,
                    documents=documents,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                # Generate without documents
                response = self.client.chat(
                    message=prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

            result = {
                'text': response.text,
                'model': model,
                'finish_reason': response.finish_reason,
                'citations': []
            }

            # Extract citations if available
            if hasattr(response, 'citations') and response.citations:
                result['citations'] = [
                    {
                        'text': citation.text,
                        'start': citation.start,
                        'end': citation.end,
                        'document_ids': citation.document_ids
                    }
                    for citation in response.citations
                ]

            logger.info(f"Generated response using model {model}")
            return result
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def rerank_documents(
        self,
        query: str,
        documents: List[str],
        top_n: int = 10,
        model: str = "rerank-multilingual-v2.0"
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to the query using Cohere rerank.

        Args:
            query: The query to rank documents against
            documents: List of documents to rerank
            top_n: Number of top documents to return
            model: The rerank model to use

        Returns:
            List of reranked documents with relevance scores
        """
        try:
            response = self.client.rerank(
                model=model,
                query=query,
                documents=documents,
                top_n=top_n
            )

            reranked_results = []
            for idx, result in enumerate(response.results):
                reranked_results.append({
                    'index': result.index,
                    'document': result.document,
                    'relevance_score': result.relevance_score
                })

            logger.info(f"Reranked {len(documents)} documents, returning top {top_n}")
            return reranked_results
        except Exception as e:
            logger.error(f"Error reranking documents: {e}")
            raise


# Global instance
cohere_manager = CohereManager()