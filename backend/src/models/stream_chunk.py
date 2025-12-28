from pydantic import BaseModel, Field
from typing import Optional


class StreamChunk(BaseModel):
    """
    Model for streaming response chunks.
    """
    content: str = Field(..., description="A chunk of the response text")
    is_final: bool = Field(default=False, description="Whether this is the final chunk in the stream")