from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from uuid import uuid4
from datetime import datetime


class LocationInfo(BaseModel):
    """
    Model for location information within the book.
    """
    chapter: Optional[str] = Field(None, description="Chapter title or identifier")
    section: Optional[str] = Field(None, description="Section title or identifier")
    page: Optional[int] = Field(None, description="Page number")
    paragraph: Optional[int] = Field(None, description="Paragraph number within section")


class Citation(BaseModel):
    """
    Model for citation information.
    """
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the citation")
    response_id: str = Field(..., description="Reference to the parent response")
    source_text: str = Field(..., description="The text from the book that supports this citation")
    location: LocationInfo = Field(..., description="Location information in the book")
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0, description="How relevant this citation is to the query (0.0-1.0)")

    class Config:
        # Allow arbitrary types for UUID handling
        arbitrary_types_allowed = True