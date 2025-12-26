---
id: 2
title: rag-chatbot-planning
stage: plan
date: '2025-12-26'
surface: agent
model: claude-sonnet-4-5-20250929
feature: rag-chatbot-book
branch: 001-rag-chatbot-book
user: Claude
command: /sp.plan
labels:
  - rag
  - chatbot
  - book
  - planning
  - architecture
links:
  SPEC: specs/001-rag-chatbot-book/spec.md
  PLAN: specs/001-rag-chatbot-book/plan.md
  TICKET: 'null'
  ADR: 'null'
  PR: 'null'
files:
  - specs/001-rag-chatbot-book/plan.md
  - specs/001-rag-chatbot-book/research.md
  - specs/001-rag-chatbot-book/data-model.md
  - specs/001-rag-chatbot-book/contracts/chat-api.yaml
  - specs/001-rag-chatbot-book/quickstart.md
  - CLAUDE.md
tests: []
---

# RAG Chatbot Planning

## User Input

```
/sp.plan
Project: Integrated Retrieval-Augmented Generation (RAG) Chatbot for Published Book (Cohere-Powered)
Project Overview
Construct and embed a high-fidelity RAG chatbot directly within the published digital book. The system shall leverage Cohere exclusively for embedding and generation, FastAPI as the backend framework, Neon Serverless Postgres for metadata persistence, and Qdrant Cloud Free Tier for vector storage and retrieval.
The chatbot must support two distinct operational modes with absolute contextual integrity:

Full-book mode: Semantic retrieval across the entire book corpus followed by grounded generation
Selected-text mode: Exclusive use of user-highlighted passage(s) as the sole grounding context, with complete bypass of vector retrieval

All development, testing, and deployment activities shall strictly adhere to the governing principles, standards, constraints, and success criteria articulated in /sp.constitution and the precise technical specifications in /sp.specify.
Execution Phases and Milestones
Phase 1: Foundation and Infrastructure Readiness (Estimated: 1–2 days | Start: December 26, 2025)

Task 1.1: Establish development environment
Initialise repository with Git; define project structure (backend, widget, scripts, tests)
Install required dependencies (FastAPI ≥ 0.110, cohere, qdrant-client, asyncpg, pydantic v2, etc.)

Task 1.2: Validate external service connectivity
Qdrant: Confirm access using provided endpoint, cluster ID, and API key; create collection book_content (1024 dimensions, cosine distance)
Cohere: Execute test embedding and chat calls to verify API key functionality
Neon Postgres: Establish connection and create schema (book_chunks, sessions, feedback_logs)

Task 1.3: Implement secure secrets management
Centralise all credentials in environment variables; enforce .gitignore exclusion
Document deployment secret injection procedures

Milestone: All infrastructure services fully accessible and responsive; development environment operational.

Phase 2: Book Ingestion Pipeline (Estimated: 2–3 days)

Task 2.1: Content preparation and chunking strategy
Finalise book input format (structured text with chapter/section/page metadata)
Implement semantic-aware chunking with controlled overlap (target 512–1024 tokens per chunk)

Task 2.2: Develop ingestion core
Batch embedding via Cohere embed-english-v3.0 (or multilingual variant)
Parallel upsert to Qdrant with comprehensive payload metadata
Synchronous metadata insertion into Neon for provenance tracking

Task 2.3: Secure ingestion endpoint (/ingest)
Implement authenticated POST endpoint accepting complete book payload
Include progress reporting and idempotency safeguards

Task 2.4: Execute full-book ingestion
Run one-time administrative script on final book content

Milestone: Entire book corpus successfully indexed in Qdrant and Neon; sample retrievals return expected chunks with accurate metadata.

Phase 3: Retrieval and Generation Core (Estimated: 3–4 days)

Task 3.1: Semantic retrieval implementation
Query embedding → Qdrant search (configurable top-k 12–16)
Optional Cohere rerank integration (conditional activation post-testing)

Task 3.2: Strict dual-mode enforcement
Explicit code path separation: selected_text → direct document grounding only
Full-book → retrieval pipeline only

Task 3.3: Grounded generation with citation discipline
Cohere chat invocation using documents parameter
System preamble mandating explicit citations in defined format
Post-processing for citation extraction and source mapping

Task 3.4: Primary API endpoint (/chat)
Support streaming responses where available
Return structured output including response text, citations, and source metadata

Task 3.5: Supporting infrastructure
Health endpoint, rate limiting, CORS restrictions, structured logging (non-PII)

Milestone: End-to-end functionality validated for both modes; responses consistently grounded with verifiable citations; observed latency within 4-second target.

Phase 4: Embeddable Frontend Widget (Estimated: 2–3 days)

Task 4.1: Widget architecture and UI design
Develop lightweight, framework-minimal JavaScript component
Implement clean query interface and selected-text detection trigger

Task 4.2: Backend integration and user experience
Real-time streaming display with inline citation highlighting
Client-side session continuity via localStorage (privacy-preserving)

Task 4.3: Cross-platform compatibility
Mobile-first responsive design
Support both script-tag and iframe embedding methods

Milestone: Widget successfully embedded in prototype book page; seamless interaction in both query modes across devices.

Phase 5: Comprehensive Testing and Validation (Estimated: 2–3 days)

Task 5.1: Automated test suite development
Unit and integration tests covering chunking, retrieval, generation, and mode isolation

Task 5.2: Accuracy and performance benchmarking
Execute ≥ 100 curated test queries (balanced across modes)
Validate ≥ 95% grounding accuracy and citation presence
Confirm zero contextual leakage in selected-text scenarios
Profile p95 latency under simulated load

Task 5.3: Security and privacy audit
Verify absence of query persistence and secret exposure
Conduct dependency vulnerability scanning

Milestone: All success criteria from /sp.constitution achieved; system ready for production deployment.

Phase 6: Production Deployment and Launch (Estimated: 1–2 days)

Task 6.1: Backend deployment
Select and configure hosting platform with secret management
Deploy application and confirm operational health

Task 6.2: Widget distribution
Publish static assets to global CDN for low-latency delivery

Task 6.3: Final production ingestion and book integration
Execute verified ingestion in production environment
Embed widget within live published book

Task 6.4: Observability establishment
Implement monitoring of latency percentiles, error rates, and resource consumption (anonymised)

Milestone: Chatbot fully operational within the published book; available to readers with demonstrated reliability and performance.

Risk Management

Resource limits: Proactive batching, top-k optimisation, and usage monitoring to remain within free-tier boundaries
Latency variance: Iterative tuning of retrieval parameters and conditional model fallback
Citation inconsistency: Reinforced system prompts combined with response post-processing
Contextual leakage: Dedicated isolation tests with automated failure detection

Overall Timeline
Total estimated duration: 11–17 days from December 26, 2025, subject to book complexity and iteration requirements.
Required Resources

Lead developer proficient in Python/FastAPI, JavaScript, and modern AI integration patterns
Final machine-readable book content with structural metadata
Hosting accounts for backend and static asset delivery

This execution plan ensures rigorous alignment with /sp.constitution and /sp.specify while delivering a professional, robust, and reader-enhancing RAG chatbot experience
```

## Response Summary

Completed the implementation planning for the RAG Chatbot project. Created the following artifacts:

1. **Plan Document** (`specs/001-rag-chatbot-book/plan.md`): Comprehensive implementation plan with technical context, constitution check, and project structure
2. **Research Document** (`specs/001-rag-chatbot-book/research.md`): Technology decisions, architecture patterns, and best practices research
3. **Data Model** (`specs/001-rag-chatbot-book/data-model.md`): Detailed entity definitions and database schema for the system
4. **API Contracts** (`specs/001-rag-chatbot-book/contracts/chat-api.yaml`): OpenAPI specification for the chatbot API endpoints
5. **Quickstart Guide** (`specs/001-rag-chatbot-book/quickstart.md`): Step-by-step setup and usage instructions
6. **Agent Context Update** (CLAUDE.md): Updated Claude Code configuration with new technologies and recent changes

All artifacts align with the constitutional requirements including Cohere exclusivity, dual-mode contextual precision, privacy-first architecture, and performance targets.

## Outcome

The planning phase is complete with all required artifacts created. The implementation is ready to proceed to the task generation phase (`/sp.tasks`).