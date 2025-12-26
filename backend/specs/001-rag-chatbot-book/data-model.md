# Data Model: RAG Chatbot for Published Book

**Feature**: RAG Chatbot for Published Book
**Date**: 2025-12-26
**Branch**: 001-rag-chatbot-book

## Overview

This document defines the data models for the RAG chatbot system, focusing on the entities identified in the feature specification and aligned with the constitutional requirements for privacy and data minimization.

## Core Entities

### Query
**Description**: User's question or request for information from the book, including metadata about the query context (full-book vs. selected-text mode)

**Fields**:
- `id` (UUID): Unique identifier for the query
- `text` (String, required): The actual query text from the user
- `mode` (Enum: "full-book" | "selected-text", required): The operational mode for this query
- `selected_text` (String, optional): Text selected by user in selected-text mode (null in full-book mode)
- `book_id` (String, required): Identifier for the book being queried
- `timestamp` (DateTime, required): When the query was submitted
- `session_id` (String, optional): Client-side session identifier (for analytics only, not stored server-side)

**Validation Rules**:
- Text must be non-empty and less than 1000 characters
- Mode must be one of the allowed values
- If mode is "selected-text", selected_text must be provided
- Timestamp must be current time (server-generated)

### Response
**Description**: AI-generated answer to user query, including content, citations, and metadata

**Fields**:
- `id` (UUID): Unique identifier for the response
- `content` (String, required): The generated response text
- `query_id` (UUID, required): Reference to the original query
- `citations` (Array of Citation objects, required): List of citations supporting the response
- `timestamp` (DateTime, required): When the response was generated
- `latency_ms` (Integer): Time taken to generate the response
- `model_used` (String): The model that generated the response

**Validation Rules**:
- Content must be non-empty
- Citations array must contain at least one citation
- Query_id must reference a valid query
- Latency must be positive

### Citation
**Description**: Reference to specific location in the book (chapter, section, page) that supports information in the response

**Fields**:
- `id` (UUID): Unique identifier for the citation
- `response_id` (UUID, required): Reference to the parent response
- `source_text` (String, required): The text from the book that supports the citation
- `location` (Object, required): Location information in the book
  - `chapter` (String, optional): Chapter title or identifier
  - `section` (String, optional): Section title or identifier
  - `page` (Integer, optional): Page number
  - `paragraph` (Integer, optional): Paragraph number within section
- `relevance_score` (Float): How relevant this citation is to the query (0.0-1.0)

**Validation Rules**:
- Source_text must be non-empty
- At least one location field must be provided
- Relevance_score must be between 0.0 and 1.0

### Session
**Description**: User's interaction context including conversation history, stored client-side only (server does not store personal information)

**Fields**:
- `id` (String): Client-generated session identifier
- `created_at` (DateTime): When the session was started
- `last_interaction` (DateTime): When the last interaction occurred
- `book_id` (String): The book associated with this session

**Validation Rules**:
- Server only stores anonymous session metadata for analytics (no personal data)
- Server does not store query history or personal information

### Book Content Chunk
**Description**: A segment of the book content stored in the vector database with metadata for retrieval

**Fields**:
- `id` (UUID): Unique identifier for the chunk (Qdrant point ID)
- `book_id` (String, required): Identifier for the book this chunk belongs to
- `content` (String, required): The text content of this chunk
- `metadata` (Object, required): Metadata for the chunk
  - `chapter` (String, optional): Chapter title or identifier
  - `section` (String, optional): Section title or identifier
  - `page_range` (Object, optional): Page range of this content
    - `start` (Integer)
    - `end` (Integer)
  - `paragraph_range` (Object, optional): Paragraph range within section
    - `start` (Integer)
    - `end` (Integer)
  - `token_count` (Integer): Number of tokens in this chunk
  - `chunk_index` (Integer): Position of this chunk in the sequence
- `embedding` (Array of Float): Vector embedding of the content

**Validation Rules**:
- Content must be between 512-1024 tokens (per constitutional requirements)
- Book_id must be valid
- Metadata must include at least one location identifier
- Embedding must match the expected dimension (1024 for Cohere embeddings)

## Database Schema

### Neon Postgres Schema

```sql
-- Table for book chunk metadata (not the actual content, which is in Qdrant)
CREATE TABLE book_chunks (
    id UUID PRIMARY KEY,
    book_id VARCHAR(255) NOT NULL,
    chunk_id VARCHAR(255) NOT NULL, -- Reference to Qdrant point ID
    content_preview TEXT, -- First 200 chars for reference only
    metadata JSONB NOT NULL, -- Full metadata including location info
    token_count INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_book_chunks_book_id ON book_chunks(book_id);
CREATE INDEX idx_book_chunks_chunk_id ON book_chunks(chunk_id);
CREATE INDEX idx_book_chunks_metadata ON book_chunks USING GIN (metadata);

-- Table for system analytics (no personal data)
CREATE TABLE analytics (
    id UUID PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL, -- 'query_submitted', 'response_generated', etc.
    session_id VARCHAR(255), -- Anonymous session identifier
    book_id VARCHAR(255) NOT NULL,
    metadata JSONB, -- Event-specific metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_analytics_session_id ON analytics(session_id);
CREATE INDEX idx_analytics_event_type ON analytics(event_type);
CREATE INDEX idx_analytics_created_at ON analytics(created_at);
```

## Relationships

1. **Query → Response**: One-to-one (each query generates one response)
2. **Response → Citations**: One-to-many (each response can have multiple citations)
3. **Book Content Chunk**: Independent entities stored in Qdrant with metadata in Postgres
4. **Session**: Exists only client-side; server only tracks anonymous analytics

## State Transitions

### Query State Flow
1. **Submitted**: Query received by the system
2. **Processing**: Embedding and retrieval in progress
3. **Response Generated**: Response created with citations
4. **Completed**: Response delivered to client

### Book Content Chunk State Flow
1. **Ingestion Queued**: Chunk ready for processing
2. **Embedded**: Embedding generated via Cohere
3. **Indexed**: Stored in Qdrant vector database
4. **Metadata Stored**: Metadata stored in Postgres
5. **Active**: Available for retrieval

## Compliance Notes

- No personal user data is stored server-side per constitutional requirements
- Query and response data are not persisted beyond immediate processing
- Session data remains client-side in localStorage
- Analytics only track anonymous, non-personal metrics