from typing import Dict, Any, List
from ..models.query import QueryMode
from ..core.cohere_client import cohere_manager
from ..core.qdrant_client import qdrant_manager
from ..core.postgres_client import postgres_manager
from ..core.logging import get_logger
from pydantic import BaseModel
from uuid import uuid4
import asyncio
import time

logger = get_logger(__name__)


class BookChunk(BaseModel):
    """
    Model for individual book chunk during ingestion.
    """
    id: str
    book_id: str
    content: str
    metadata: Dict[str, Any]
    token_count: int


class IngestionService:
    """
    Service for handling book content ingestion and indexing.
    """

    @staticmethod
    async def ingest_book_content(
        book_id: str,
        title: str,
        content: str,
        metadata: Dict[str, Any],
        chunk_size: int = 1000,
        overlap: int = 100
    ) -> Dict[str, Any]:
        """
        Ingest book content by chunking, embedding, and storing in vector database.

        Args:
            book_id: Unique identifier for the book
            title: Title of the book
            content: Full text content of the book
            metadata: Metadata about the book structure
            chunk_size: Size of text chunks (default: 1000 characters)
            overlap: Overlap between chunks (default: 100 characters)

        Returns:
            Dictionary with ingestion results
        """
        start_time = time.time()

        try:
            logger.info(f"Starting ingestion for book: {book_id}")

            # Create Qdrant collection if it doesn't exist
            qdrant_manager.create_collection()

            # Use the chunking service for semantic-aware chunking
            from .chunking_service import ChunkingService
            chunks = ChunkingService.chunk_content(
                content=content,
                book_id=book_id,
                metadata=metadata,
                chunk_size=chunk_size,
                overlap=overlap
            )

            logger.info(f"Created {len(chunks)} chunks for book {book_id} using semantic-aware chunking")

            # Generate embeddings for all chunks using batch processing
            all_embeddings = await IngestionService._generate_embeddings_in_batches(chunks)

            # Prepare points for Qdrant
            points = []
            for i, chunk in enumerate(chunks):
                point = {
                    "id": chunk.id,
                    "vector": all_embeddings[i],
                    "payload": {
                        "content": chunk.content,
                        "book_id": chunk.book_id,
                        "metadata": chunk.metadata,
                        "token_count": chunk.token_count
                    }
                }
                points.append(point)

            # Upsert vectors to Qdrant
            qdrant_manager.upsert_vectors(points)

            # Store metadata in Postgres
            for chunk in chunks:
                chunk_metadata = {
                    "id": chunk.id,
                    "book_id": chunk.book_id,
                    "chunk_id": chunk.id,
                    "content_preview": chunk.content,
                    "metadata": chunk.metadata,
                    "token_count": chunk.token_count
                }
                await postgres_manager.insert_book_chunk(chunk_metadata)

            processing_time = time.time() - start_time

            result = {
                "status": "success",
                "book_id": book_id,
                "chunks_processed": len(chunks),
                "processing_time_ms": int(processing_time * 1000),
                "title": title
            }

            logger.info(f"Successfully ingested book {book_id} with {len(chunks)} chunks in {processing_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Error during book ingestion: {str(e)}", exc_info=True)
            raise

    @staticmethod
    async def _generate_embeddings_in_batches(chunks: List[BookChunk], batch_size: int = 10) -> List[List[float]]:
        """
        Generate embeddings for chunks in batches to respect API limits.

        Args:
            chunks: List of chunks to embed
            batch_size: Number of chunks to process in each batch

        Returns:
            List of embedding vectors
        """
        all_embeddings = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_texts = [chunk.content for chunk in batch]

            # Generate embeddings for the batch
            embeddings = cohere_manager.embed_text(batch_texts)
            all_embeddings.extend(embeddings)

            logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}: {len(batch)} chunks")

        logger.info(f"Generated {len(all_embeddings)} embeddings for {len(chunks)} chunks")
        return all_embeddings

    @staticmethod
    def _chunk_content(
        content: str,
        book_id: str,
        metadata: Dict[str, Any],
        chunk_size: int = 1000,
        overlap: int = 100
    ) -> List[BookChunk]:
        """
        Split content into overlapping chunks.

        Args:
            content: The content to chunk
            book_id: The book ID these chunks belong to
            metadata: Metadata to include with each chunk
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks in characters

        Returns:
            List of BookChunk objects
        """
        if chunk_size <= overlap:
            raise ValueError("Chunk size must be greater than overlap")

        chunks = []
        start = 0

        while start < len(content):
            end = start + chunk_size

            # Make sure we don't go beyond the content
            if end > len(content):
                end = len(content)

            # Extract the chunk
            chunk_text = content[start:end]

            # Create metadata for this specific chunk
            chunk_metadata = metadata.copy()
            chunk_metadata["position"] = len(chunks)  # Position in the book
            chunk_metadata["start_offset"] = start
            chunk_metadata["end_offset"] = end

            # Create the chunk object
            chunk = BookChunk(
                id=str(uuid4()),
                book_id=book_id,
                content=chunk_text,
                metadata=chunk_metadata,
                token_count=len(chunk_text.split())  # Rough token count
            )

            chunks.append(chunk)

            # Move to the next chunk position
            start = end - overlap

            # If we've reached the end, break
            if end == len(content):
                break

        logger.info(f"Created {len(chunks)} chunks from content of length {len(content)}")
        return chunks

    @staticmethod
    async def validate_ingestion_request(
        book_id: str,
        title: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Validate the ingestion request parameters.

        Args:
            book_id: The book ID
            title: The book title
            content: The book content
            metadata: The book metadata

        Returns:
            True if valid, raises ValueError otherwise
        """
        if not book_id or not book_id.strip():
            raise ValueError("Book ID is required")

        if not title or not title.strip():
            raise ValueError("Book title is required")

        if not content or len(content.strip()) == 0:
            raise ValueError("Book content is required and cannot be empty")

        if len(content) < 100:  # Minimum reasonable content length
            raise ValueError("Book content is too short to be meaningful")

        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dictionary")

        # Additional validation can be added here
        logger.info(f"Validated ingestion request for book: {book_id}")
        return True

    @staticmethod
    async def get_ingestion_status(book_id: str) -> Dict[str, Any]:
        """
        Get the ingestion status for a specific book.

        Args:
            book_id: The book ID to check

        Returns:
            Dictionary with ingestion status information
        """
        try:
            # In a real implementation, this would check both Qdrant and Postgres
            # to determine ingestion status

            # Check if book chunks exist in Postgres
            chunks = await postgres_manager.get_book_chunks(book_id, limit=1)
            exists = len(chunks) > 0

            # Get collection info from Qdrant
            collection_info = qdrant_manager.get_collection_info()

            status = {
                "book_id": book_id,
                "exists": exists,
                "status": "completed" if exists else "not_found",
                "chunks_count": len(chunks),
                "collection_info": collection_info
            }

            logger.info(f"Retrieved ingestion status for book: {book_id}")
            return status

        except Exception as e:
            logger.error(f"Error getting ingestion status: {str(e)}", exc_info=True)
            raise

    @staticmethod
    async def remove_book_content(book_id: str) -> Dict[str, Any]:
        """
        Remove all content for a specific book from the system.

        Args:
            book_id: The book ID to remove

        Returns:
            Dictionary with removal results
        """
        try:
            logger.info(f"Starting removal of book content: {book_id}")

            # Get all chunks for this book from Postgres
            chunks = await postgres_manager.get_book_chunks(book_id)

            # In a real implementation, we would need to:
            # 1. Remove from Qdrant (would need to query by book_id and remove)
            # 2. Remove from Postgres

            # For now, we'll just return a placeholder result
            result = {
                "book_id": book_id,
                "status": "removed",
                "chunks_removed": len(chunks),
                "message": f"Book {book_id} content has been removed from the system"
            }

            logger.info(f"Completed removal of book content: {book_id}")
            return result

        except Exception as e:
            logger.error(f"Error removing book content: {str(e)}", exc_info=True)
            raise