# Research: RAG Chatbot for Published Book

**Feature**: RAG Chatbot for Published Book
**Date**: 2025-12-26
**Branch**: 001-rag-chatbot-book

## Executive Summary

This research document outlines the technical decisions, architecture patterns, and best practices for implementing the RAG chatbot for a published book. The implementation will leverage Cohere for embeddings and generation, Qdrant for vector storage, and FastAPI for the backend API.

## Technology Decisions

### Decision: Use FastAPI for Backend Framework
**Rationale**: FastAPI provides async support, automatic API documentation (OpenAPI/Swagger), built-in validation with Pydantic, and excellent performance for AI service orchestration. It's ideal for handling concurrent requests from the chat interface and integrating with external AI services.

**Alternatives considered**:
- Flask: Less performant for async operations, requires more boilerplate
- Django: Overkill for API-only service, heavier framework
- Starlette: Lower-level, would require more custom implementation

### Decision: Use Cohere embed-multilingual-v3.0 for Embeddings
**Rationale**: Required by constitutional principle of Cohere exclusivity. The multilingual model provides better support for diverse book content and is the latest generation with improved performance.

**Alternatives considered**:
- embed-english-v3.0: Only if book is exclusively in English
- Previous generation models: Would be less performant

### Decision: Use Cohere Command R+ for Generation
**Rationale**: Required by constitutional principle. Command R+ provides the best performance for grounded generation and citation capabilities needed for this project.

**Alternatives considered**:
- Command R: Slightly less capable but still compliant with constitutional requirements

### Decision: Use Qdrant Cloud Free Tier for Vector Storage
**Rationale**: Required by constitutional principle. Qdrant provides efficient similarity search and handles the vector storage needs of the RAG system.

**Alternatives considered**:
- Pinecone: Would require different API integration, violates constitutional requirement
- ChromaDB: Self-hosted option but violates constitutional requirement for Qdrant
- Weaviate: Would violate constitutional requirement

### Decision: Use Neon Serverless Postgres for Metadata Storage
**Rationale**: Required by constitutional principle. Provides reliable storage for chunk metadata, provenance tracking, and session information with serverless scalability.

**Alternatives considered**:
- SQLite: Would be simpler but doesn't meet constitutional requirement
- MongoDB: Would violate constitutional requirement for Postgres

## Architecture Patterns

### Pattern: Dual-Mode Operation with Strict Isolation
**Implementation**: Separate code paths for full-book mode vs. selected-text mode to ensure zero contextual leakage between modes.

**Rationale**: Critical to meet the constitutional requirement of dual-mode contextual precision. The system must never allow full-corpus knowledge to influence selected-text responses.

### Pattern: Streaming Responses with Citation Tracking
**Implementation**: Use FastAPI's streaming responses combined with Cohere's streaming capabilities to provide real-time feedback with proper citation tracking.

**Rationale**: Meets user experience requirements for real-time interaction while maintaining citation discipline.

### Pattern: Client-Side Session Management
**Implementation**: Use browser localStorage for session continuity without server-side storage of personal information.

**Rationale**: Required to meet privacy and security constitutional requirements while providing seamless user experience.

## Best Practices Research

### For FastAPI in AI Applications
- Use async/await for external API calls to maximize concurrency
- Implement proper error handling for external service failures
- Use Pydantic models for request/response validation
- Implement rate limiting to prevent abuse
- Use dependency injection for service management

### For RAG Systems
- Implement semantic chunking with overlap to maintain context
- Use configurable top-k for retrieval to balance performance and accuracy
- Implement proper grounding validation to ensure responses are based on provided context
- Use proper citation extraction to link responses to source material

### For Embeddable Widgets
- Keep JavaScript bundle size minimal
- Implement proper cross-origin communication
- Use CSS isolation to prevent style conflicts
- Implement proper cleanup to avoid memory leaks
- Support both iframe and script tag embedding methods

## Security Considerations

### Input Sanitization
- All user queries must be sanitized before processing
- Prevent injection attacks in the query processing pipeline
- Validate all input parameters to API endpoints

### Privacy Protection
- No user queries stored server-side
- No session data with personal information
- Proper handling of client-side storage

### API Security
- Secure API endpoints with appropriate authentication for ingestion
- Rate limiting to prevent abuse
- Proper CORS configuration for embeddable widget

## Performance Optimization

### For Qdrant
- Use appropriate vector dimensions (1024 as specified in requirements)
- Implement efficient search parameters (top-k, filters)
- Consider payload optimization to reduce network overhead

### For Cohere API
- Use appropriate model parameters for generation quality vs speed trade-offs
- Implement proper batching where possible
- Consider caching for frequently asked questions

### For Database
- Proper indexing on metadata for efficient lookup
- Connection pooling for database operations
- Efficient queries for metadata retrieval

## Compliance Verification

All decisions align with the constitutional requirements:
- ✅ Cohere exclusivity maintained
- ✅ Qdrant for vector storage
- ✅ FastAPI for backend
- ✅ Neon Postgres for metadata
- ✅ Privacy-first architecture
- ✅ Dual-mode isolation
- ✅ Performance targets achievable