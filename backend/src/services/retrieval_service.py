from typing import List, Dict, Any, Optional
from ..models.query import Query
from ..core.cohere_client import cohere_manager
from ..core.qdrant_client import qdrant_manager
from ..core.postgres_client import postgres_manager
from ..core.logging import get_logger
import time

logger = get_logger(__name__)


class RetrievalService:
    """
    Service for handling document retrieval from vector store.
    """

    @staticmethod
    async def retrieve_relevant_chunks(
        query_text: str,
        book_id: str,
        top_k: int = 10,
        min_score: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks from the vector store based on the query.

        Args:
            query_text: The query text to search for
            book_id: The ID of the book to search within
            top_k: Number of top results to return
            min_score: Minimum similarity score for results

        Returns:
            List of relevant document chunks with metadata
        """
        try:
            start_time = time.time()

            # Generate embedding for the query
            embeddings = cohere_manager.embed_text([query_text])
            query_embedding = embeddings[0]  # Get the first (and only) embedding

            # Search in Qdrant
            search_results = qdrant_manager.search_vectors(
                vector=query_embedding,
                limit=top_k * 2  # Get more results than needed for filtering
            )

            # Filter results by minimum score and book ID if needed
            filtered_results = []
            for result in search_results:
                score = result.get('score', 0.0)
                payload = result.get('payload', {})

                # Filter by minimum score
                if score >= min_score:
                    # If book_id is provided, filter by it as well
                    if book_id and payload.get('book_id') != book_id:
                        continue

                    filtered_results.append({
                        'id': result.get('id'),
                        'content': payload.get('content', ''),
                        'metadata': payload.get('metadata', {}),
                        'score': score
                    })

            # Limit to top_k after filtering
            relevant_chunks = sorted(filtered_results, key=lambda x: x['score'], reverse=True)[:top_k]

            retrieval_time = time.time() - start_time
            logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks in {retrieval_time:.3f}s")
            return relevant_chunks

        except Exception as e:
            logger.error(f"Error retrieving chunks: {str(e)}", exc_info=True)
            raise

    @staticmethod
    async def retrieve_chunks_by_ids(chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve specific chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs to retrieve

        Returns:
            List of document chunks with metadata
        """
        try:
            # This would require a Qdrant client method to get points by ID
            # For now, we'll return an empty list as a placeholder
            chunks = []

            logger.info(f"Retrieved {len(chunks)} chunks by IDs")
            return chunks

        except Exception as e:
            logger.error(f"Error retrieving chunks by IDs: {str(e)}", exc_info=True)
            raise

    @staticmethod
    async def retrieve_chunks_by_metadata(
        metadata_filters: Dict[str, Any],
        book_id: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks based on metadata filters.

        Args:
            metadata_filters: Dictionary of metadata filters to apply
            book_id: The ID of the book to search within
            top_k: Number of top results to return

        Returns:
            List of document chunks matching the filters
        """
        try:
            # This would require implementing filtered searches in Qdrant
            # For now, we'll return an empty list as a placeholder
            chunks = []

            logger.info(f"Retrieved {len(chunks)} chunks by metadata filters")
            return chunks

        except Exception as e:
            logger.error(f"Error retrieving chunks by metadata: {str(e)}", exc_info=True)
            raise

    @staticmethod
    async def validate_retrieval_query(query: Query) -> bool:
        """
        Validate the query for retrieval operations.

        Args:
            query: The query to validate

        Returns:
            True if valid, raises ValueError otherwise
        """
        if not query.text or not query.text.strip():
            raise ValueError("Query text is required for retrieval")

        if len(query.text.strip()) > 1000:
            raise ValueError("Query text exceeds maximum length of 1000 characters")

        if not query.book_id or not query.book_id.strip():
            raise ValueError("Book ID is required for retrieval")

        return True

    @staticmethod
    async def retrieve_with_reranking(
        query_text: str,
        book_id: str,
        top_k: int = 10,
        rerank_top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks and then rerank them using Cohere.

        Args:
            query_text: The query text to search for
            book_id: The ID of the book to search within
            top_k: Number of results to initially retrieve
            rerank_top_n: Number of top results after reranking

        Returns:
            List of reranked document chunks with metadata
        """
        try:
            start_time = time.time()

            # First, retrieve initial set of chunks
            initial_chunks = await RetrievalService.retrieve_relevant_chunks(
                query_text=query_text,
                book_id=book_id,
                top_k=top_k,
                min_score=0.0  # Don't filter by score initially for reranking
            )

            # Extract content for reranking
            documents = [chunk['content'] for chunk in initial_chunks if chunk.get('content')]

            if not documents:
                logger.info("No documents found for reranking")
                return []

            # Use Cohere to rerank the documents
            reranked_results = cohere_manager.rerank_documents(
                query=query_text,
                documents=documents,
                top_n=rerank_top_n
            )

            # Map reranked results back to original chunks
            reranked_chunks = []
            for rerank_result in reranked_results:
                original_idx = rerank_result['index']
                if original_idx < len(initial_chunks):
                    chunk = initial_chunks[original_idx].copy()
                    chunk['rerank_score'] = rerank_result['relevance_score']
                    chunk['original_score'] = initial_chunks[original_idx].get('score', 0.0)
                    reranked_chunks.append(chunk)

            rerank_time = time.time() - start_time
            logger.info(f"Reranked {len(reranked_chunks)} chunks in {rerank_time:.3f}s")
            return reranked_chunks

        except Exception as e:
            logger.error(f"Error in retrieval with reranking: {str(e)}", exc_info=True)
            raise

    @staticmethod
    async def get_collection_statistics(book_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about the stored documents.

        Args:
            book_id: Optional book ID to get statistics for a specific book

        Returns:
            Dictionary with collection statistics
        """
        try:
            collection_info = qdrant_manager.get_collection_info()

            stats = {
                'total_chunks': collection_info['point_count'],
                'vector_size': collection_info['vector_size'],
                'distance_metric': collection_info['distance']
            }

            # If book_id is specified, get book-specific stats
            if book_id:
                # This would require querying Postgres for book-specific metadata
                try:
                    book_chunks = await postgres_manager.get_book_chunks(book_id, limit=1)
                    stats['book_exists'] = len(book_chunks) > 0
                    stats['book_chunk_count'] = len(book_chunks)
                except Exception:
                    stats['book_exists'] = False
                    stats['book_chunk_count'] = 0

            logger.info(f"Retrieved collection statistics: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Error getting collection statistics: {str(e)}", exc_info=True)
            raise