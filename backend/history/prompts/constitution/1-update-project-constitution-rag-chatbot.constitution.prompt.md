---
id: 1
title: update-project-constitution-rag-chatbot
stage: constitution
date: '2025-12-26'
surface: agent
model: claude-sonnet-4-5-20250929
feature: 'none'
branch: 002-vla
user: Claude
command: /sp.constitution
labels:
  - constitution
  - rag-chatbot
  - project-setup
links:
  SPEC: 'null'
  TICKET: 'null'
  ADR: 'null'
  PR: 'null'
files:
  - .specify/memory/constitution.md
tests: []
---

# Update Project Constitution - RAG Chatbot

## User Input

```
/sp.constitution
Project: Integrated Retrieval-Augmented Generation (RAG) Chatbot for Published Book (Cohere-Powered)
Core principles:

Uncompromising fidelity to source material: All responses must be derived exclusively from the book's content or user-selected text, with zero tolerance for hallucination or external knowledge injection
Dual-mode contextual precision: Flawless distinction and isolation between full-book queries and queries restricted to user-selected passages, ensuring absolute contextual integrity
Elevated reader experience: Provide sophisticated, intuitive, and intellectually enriching interactions that seamlessly augment and deepen engagement with the published work
Architectural and engineering excellence: Deliver a scalable, performant, secure, and maintainable system grounded in best-of-breed cloud-native and AI technologies

Key standards:

Vector retrieval: Qdrant Cloud Free Tier serving as the high-performance semantic search engine with Cohere-generated dense embeddings
Backend orchestration: FastAPI as the foundational framework, enabling asynchronous, type-safe, and automatically documented API services
Metadata persistence: Neon Serverless Postgres for durable storage of chunk provenance, structural metadata, and optional session state
Embedding generation: Cohere embed-multilingual-v3.0 or embed-english-v3.0, selected for optimal semantic representation of the book's language
Response generation: Cohere Command R+ (preferred) or Command R, leveraging native grounded generation and citation capabilities for superior reasoning and accuracy
Text processing: Semantic-aware chunking with controlled overlap (512–1024 tokens per chunk) and comprehensive metadata retention (chapter, section, page range) to facilitate precise attribution
Citation discipline: Every substantive response must incorporate explicit, verifiable references to the originating locations within the book (e.g., chapter, section, page)
Selected-text mode enforcement: When user-highlighted text is provided, the system must exclusively utilize that text as the grounding context, completely bypassing vector retrieval from the full corpus
Performance excellence: End-to-end response latency not exceeding 4 seconds at the95th percentile under typical usage conditions
Privacy and security posture: Strict data minimization, ephemeral session handling, and comprehensive protection against query logging or personal data retention

Constraints:

Mandatory Cohere exclusivity: All embedding, optional reranking, and generative functions must be executed solely via Cohere APIs and models; any use of OpenAI services or SDKs is strictly prohibited
Infrastructure boundaries: Operation must remain within the operational and cost limits of Qdrant Cloud Free Tier and Neon Serverless Postgres, with deliberate optimization for resource efficiency
No model customization: Dependence exclusively on Cohere's publicly available pre-trained production models; fine-tuning or custom training is forbidden
Delivery mechanism: Embeddable, lightweight widget distributed via iframe or script tag, fully compatible with leading digital publishing and e-book platforms
Privacy-first architecture: No persistent storage of user queries, conversation history (unless explicitly anonymized and opt-in), or identifiable information
Universal accessibility: Responsive, cross-device design ensuring consistent functionality across desktop, tablet, and mobile reading environments

Success criteria:

Grounding and accuracy benchmark: ≥ 95% of responses deemed fully accurate, relevant, and properly cited across a rigorous test suite comprising 100+ diverse, book-specific queries
Seamless book integration: Chatbot successfully embedded and fully functional within the live published book environment without disrupting native reading flow
Contextual isolation validation: Zero occurrences of full-corpus knowledge influencing selected-text responses during exhaustive testing
Operational robustness: Absence of critical failures (retrieval breakdowns, timeouts, generation errors) under production-equivalent load and stress scenarios
Reader satisfaction: Consistently positive qualitative feedback regarding response relevance, citation utility, interaction speed, interface intuitiveness, and overall enhancement of the reading experience
Implementation quality: Delivery of a clean, comprehensively documented, type-annotated codebase that satisfies professional standards for readability, maintainability, security, and static analysis

This constitution establishes the definitive governing framework for the development, ensuring the resulting RAG chatbot exemplifies professional excellence while remaining faithful to the specified technological stack and project objectives.
```

## Response Summary

Updated the project constitution at `.specify/memory/constitution.md` with the specifications for the Integrated Retrieval-Augmented Generation (RAG) Chatbot for Published Book (Cohere-Powered). The new constitution includes:

- Core principles: Uncompromising fidelity to source material, dual-mode contextual precision, elevated reader experience, architectural excellence, and Cohere API exclusivity
- Key standards: Vector retrieval with Qdrant, FastAPI backend, Neon Postgres persistence, Cohere embeddings/generation, performance excellence, and privacy/security
- Constraints: Cohere exclusivity, infrastructure boundaries, no model customization, delivery mechanism, and universal accessibility
- Success criteria: Grounding accuracy, seamless integration, contextual isolation, operational robustness, reader satisfaction, and implementation quality
- Governance rules for maintaining compliance with the constitutional principles

The constitution version is now 1.0.0, ratified on 2025-12-26.

## Outcome

The project constitution has been successfully updated to reflect the requirements for the RAG chatbot project, establishing clear governance for the development process.