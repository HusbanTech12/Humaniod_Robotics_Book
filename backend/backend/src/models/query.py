from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from uuid import UUID, uuid4
from datetime import datetime


class QueryMode(str, Enum):
    full_book = "full-book"
    selected_text = "selected-text"


class QueryRequest(BaseModel):
    """
    Model for incoming query requests.
    """
    query: str = Field(..., description="The user's question about the book", max_length=1000)
    book_id: str = Field(..., description="Identifier for the book being queried", example="book-978-0123456789")
    mode: QueryMode = Field(default=QueryMode.full_book, description="The operational mode for this query")
    selected_text: Optional[str] = Field(None, description="Text selected by user in selected-text mode")


class QueryResponse(BaseModel):
    """
    Model for query responses.
    """
    response: str = Field(..., description="The AI-generated response")
    citations: List[dict] = Field(..., description="Citations supporting the response")
    query_id: str = Field(..., description="Unique identifier for the original query")
    response_id: str = Field(..., description="Unique identifier for this response")
    latency_ms: int = Field(..., description="Time taken to generate the response in milliseconds")


class StreamChunk(BaseModel):
    """
    Model for streaming response chunks.
    """
    content: str = Field(..., description="A chunk of the response text")
    is_final: bool = Field(default=False, description="Whether this is the final chunk")


class Query(BaseModel):
    """
    Model for internal query representation.
    """
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the query")
    text: str = Field(..., description="The actual query text from the user", max_length=1000)
    mode: QueryMode = Field(..., description="The operational mode for this query")
    selected_text: Optional[str] = Field(None, description="Text selected by user in selected-text mode")
    book_id: str = Field(..., description="Identifier for the book being queried")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the query was submitted")
    session_id: Optional[str] = Field(None, description="Client-side session identifier (for analytics only)")

    class Config:
        use_enum_values = True  # Store enum values as strings

    def __init__(self, **data):
        # Validate that selected_text is provided when mode is selected-text
        mode = data.get('mode')
        selected_text = data.get('selected_text')

        if isinstance(mode, str):
            mode = QueryMode(mode)

        if mode == QueryMode.selected_text and not selected_text:
            raise ValueError("selected_text is required when mode is selected-text")

        super().__init__(**data)


class QueryUpdate(BaseModel):
    """
    Model for updating query parameters.
    """
    mode: Optional[QueryMode] = None
    selected_text: Optional[str] = None

    class Config:
        use_enum_values = True  # Store enum values as strings