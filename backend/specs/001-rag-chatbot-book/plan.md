# Implementation Plan: RAG Chatbot for Published Book

**Branch**: `001-rag-chatbot-book` | **Date**: 2025-12-26 | **Spec**: [Link to spec](spec.md)
**Input**: Feature specification from `/specs/001-rag-chatbot-book/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implementation of an integrated Retrieval-Augmented Generation (RAG) chatbot for a published book, leveraging Cohere for embeddings and generation, Qdrant for vector storage, and FastAPI for the backend. The system will support dual-mode operation (full-book and selected-text) with strict contextual isolation, real-time streaming responses with citations, and mobile-responsive design. The backend will be built with Python/FastAPI and the frontend as a lightweight embeddable widget.

## Technical Context

**Language/Version**: Python 3.11+ (for async support and modern features needed for FastAPI and AI integration)
**Primary Dependencies**: FastAPI, cohere, qdrant-client, asyncpg, pydantic v2, psycopg[binary]
**Storage**: Neon Serverless Postgres for metadata and Qdrant Cloud for vector storage
**Testing**: pytest, pytest-asyncio, pytest-cov for backend; Jest or Vitest for frontend widget
**Target Platform**: Linux server (backend), cross-platform web browser (frontend widget)
**Project Type**: web application - backend API with embeddable frontend widget
**Performance Goals**: ≤ 4 second p95 response latency, handle concurrent requests efficiently
**Constraints**: Must stay within Qdrant Cloud Free Tier and Neon Serverless Postgres limits
**Scale/Scope**: Single book corpus, multiple concurrent users, responsive across devices

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] Cohere API exclusivity: All embedding and generation will use only Cohere APIs (Command R+ and embed-multilingual-v3.0)
- [x] Vector retrieval with Qdrant: Implementation will use Qdrant Cloud Free Tier for semantic search
- [x] Backend orchestration with FastAPI: Will use FastAPI for asynchronous API services
- [x] Metadata persistence with Neon Postgres: Book chunk metadata will be stored in Neon Serverless Postgres
- [x] Dual-mode contextual precision: Code paths will be strictly separated for full-book vs selected-text modes
- [x] Performance excellence: Target ≤ 4s p95 latency will be monitored and optimized
- [x] Privacy and security posture: No user queries or personal data will be stored server-side
- [x] Delivery mechanism: Widget will be distributed via iframe/script tag for book integration
- [x] Universal accessibility: Responsive design will ensure cross-device compatibility

## Project Structure

### Documentation (this feature)
```text
specs/001-rag-chatbot-book/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)
```text
backend/
├── src/
│   ├── models/
│   ├── services/
│   ├── api/
│   └── core/
└── tests/

widget/
├── src/
│   ├── components/
│   ├── utils/
│   └── styles/
├── dist/
└── tests/

scripts/
└── ingestion/
    └── book_ingestion.py
```

**Structure Decision**: Web application with separate backend service and frontend widget. The backend handles all RAG logic (embedding, retrieval, generation) while the frontend provides the embeddable chat interface. The ingestion script handles book content processing.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Multiple services | Required by architectural pattern | Cohere + Qdrant + Postgres are distinct specialized services that cannot be combined |