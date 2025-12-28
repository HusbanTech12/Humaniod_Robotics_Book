from pydantic import BaseModel, Field
from typing import List, Optional
from uuid import UUID, uuid4
from datetime import datetime
from .citation import Citation


class Response(BaseModel):
    """
    Model for internal response representation.
    """
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the response")
    content: str = Field(..., description="The generated response text")
    query_id: str = Field(..., description="Reference to the original query")
    citations: List[Citation] = Field(default=[], description="List of citations supporting the response")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the response was generated")
    latency_ms: int = Field(..., description="Time taken to generate the response")
    model_used: str = Field(..., description="The model that generated the response")

    class Config:
        # Allow arbitrary types for UUID handling
        arbitrary_types_allowed = True