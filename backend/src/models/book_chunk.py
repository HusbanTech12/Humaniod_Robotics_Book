from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from uuid import UUID, uuid4
from datetime import datetime


class BookChunk(BaseModel):
    """
    Model for book content chunks stored in the vector database with metadata for retrieval.
    """
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the chunk (Qdrant point ID)")
    book_id: str = Field(..., description="Identifier for the book this chunk belongs to")
    content: str = Field(..., description="The text content of this chunk")
    metadata: Dict[str, Any] = Field(..., description="Metadata for the chunk")
    embedding: Optional[list] = Field(None, description="Vector embedding of the content (optional for input)")

    class Config:
        # Allow arbitrary types for UUID handling
        arbitrary_types_allowed = True


class BookChunkForStorage(BaseModel):
    """
    Model for book chunks specifically for storage in the database.
    """
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the chunk")
    book_id: str = Field(..., description="Identifier for the book this chunk belongs to")
    chunk_id: str = Field(..., description="Reference to Qdrant point ID")
    content_preview: Optional[str] = Field(None, description="First 200 chars for reference only")
    metadata: Dict[str, Any] = Field(..., description="Full metadata including location info")
    token_count: int = Field(..., description="Number of tokens in this chunk")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the chunk was created")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="When the chunk was last updated")


class IngestionRequest(BaseModel):
    """
    Model for book ingestion request.
    """
    book_id: str = Field(..., description="Unique identifier for the book", example="book-978-0123456789")
    title: str = Field(..., description="Title of the book", example="The Art of Programming")
    content: str = Field(..., description="Full text content of the book with structural metadata")
    metadata: Dict[str, Any] = Field(..., description="Metadata about the book structure")


class IngestionResponse(BaseModel):
    """
    Model for book ingestion response.
    """
    status: str = Field(..., description="Status of the ingestion process", example="success")
    book_id: str = Field(..., description="Identifier for the book that was ingested")
    chunks_processed: int = Field(..., description="Number of content chunks successfully processed")
    processing_time_ms: int = Field(..., description="Time taken to process the book in milliseconds")