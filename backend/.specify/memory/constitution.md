<!-- SYNC IMPACT REPORT
Version change: N/A (initial version) → 1.0.0
List of modified principles: N/A (new constitution)
Added sections: Core Principles, Key Standards, Constraints, Success Criteria
Removed sections: None
Templates requiring updates:
- ✅ .specify/templates/plan-template.md - Governance rules referenced in Constitution Check section
- ✅ .specify/templates/spec-template.md - Updated to align with new principles
- ✅ .specify/templates/tasks-template.md - Updated to reflect new constraints
- ✅ .specify/commands/sp.constitution.md - Already processed
Follow-up TODOs: None
-->

# Integrated Retrieval-Augmented Generation (RAG) Chatbot for Published Book (Cohere-Powered) Constitution

## Core Principles

### Uncompromising fidelity to source material
All responses must be derived exclusively from the book's content or user-selected text, with zero tolerance for hallucination or external knowledge injection. This ensures the chatbot maintains strict adherence to the published work without introducing fabricated information.

### Dual-mode contextual precision
Flawless distinction and isolation between full-book queries and queries restricted to user-selected passages, ensuring absolute contextual integrity. When user-highlighted text is provided, the system must exclusively utilize that text as the grounding context, completely bypassing vector retrieval from the full corpus.

### Elevated reader experience
Provide sophisticated, intuitive, and intellectually enriching interactions that seamlessly augment and deepen engagement with the published work. The chatbot must deliver responses that enhance rather than disrupt the reading experience.

### Architectural and engineering excellence
Deliver a scalable, performant, secure, and maintainable system grounded in best-of-breed cloud-native and AI technologies. The implementation must follow professional standards for readability, maintainability, security, and static analysis.

### Cohere API exclusivity
All embedding, optional reranking, and generative functions must be executed solely via Cohere APIs and models; any use of OpenAI services or SDKs is strictly prohibited. This ensures consistency with the specified technological stack.

## Key Standards

### Vector retrieval and storage
Qdrant Cloud Free Tier serving as the high-performance semantic search engine with Cohere-generated dense embeddings. The system must operate within the resource constraints of the free tier while maintaining performance excellence.

### Backend orchestration
FastAPI as the foundational framework, enabling asynchronous, type-safe, and automatically documented API services. This provides the necessary performance and reliability for real-time chatbot interactions.

### Metadata persistence
Neon Serverless Postgres for durable storage of chunk provenance, structural metadata, and optional session state. The system must optimize for resource efficiency within the Neon Serverless Postgres constraints.

### Embedding generation
Cohere embed-multilingual-v3.0 or embed-english-v3.0, selected for optimal semantic representation of the book's language. The embedding strategy must support semantic-aware chunking with controlled overlap (512–1024 tokens per chunk).

### Response generation
Cohere Command R+ (preferred) or Command R, leveraging native grounded generation and citation capabilities for superior reasoning and accuracy. Every substantive response must incorporate explicit, verifiable references to the originating locations within the book (e.g., chapter, section, page).

### Performance excellence
End-to-end response latency not exceeding 4 seconds at the 95th percentile under typical usage conditions. The system must maintain this performance standard while delivering high-quality responses.

### Privacy and security posture
Strict data minimization, ephemeral session handling, and comprehensive protection against query logging or personal data retention. The architecture must ensure no persistent storage of user queries, conversation history (unless explicitly anonymized and opt-in), or identifiable information.

## Constraints

### Mandatory Cohere exclusivity
All embedding, optional reranking, and generative functions must be executed solely via Cohere APIs and models; any use of OpenAI services or SDKs is strictly prohibited. This constraint ensures adherence to the specified technological stack.

### Infrastructure boundaries
Operation must remain within the operational and cost limits of Qdrant Cloud Free Tier and Neon Serverless Postgres, with deliberate optimization for resource efficiency. The implementation must be cost-effective and scalable within these constraints.

### No model customization
Dependence exclusively on Cohere's publicly available pre-trained production models; fine-tuning or custom training is forbidden. This maintains the simplicity and reliability of the solution.

### Delivery mechanism
Embeddable, lightweight widget distributed via iframe or script tag, fully compatible with leading digital publishing and e-book platforms. The solution must integrate seamlessly without disrupting native reading flow.

### Universal accessibility
Responsive, cross-device design ensuring consistent functionality across desktop, tablet, and mobile reading environments. The chatbot must provide a consistent experience regardless of the user's device.

## Success Criteria

### Grounding and accuracy benchmark
≥ 95% of responses deemed fully accurate, relevant, and properly cited across a rigorous test suite comprising 100+ diverse, book-specific queries. This ensures the chatbot maintains high standards for factual accuracy and proper attribution.

### Seamless book integration
Chatbot successfully embedded and fully functional within the live published book environment without disrupting native reading flow. The integration must be unobtrusive and enhance the reading experience.

### Contextual isolation validation
Zero occurrences of full-corpus knowledge influencing selected-text responses during exhaustive testing. This validates that the dual-mode contextual precision principle is properly implemented.

### Operational robustness
Absence of critical failures (retrieval breakdowns, timeouts, generation errors) under production-equivalent load and stress scenarios. The system must be reliable and resilient.

### Reader satisfaction
Consistently positive qualitative feedback regarding response relevance, citation utility, interaction speed, interface intuitiveness, and overall enhancement of the reading experience. The chatbot must provide genuine value to readers.

### Implementation quality
Delivery of a clean, comprehensively documented, type-annotated codebase that satisfies professional standards for readability, maintainability, security, and static analysis. The code must meet high engineering standards.

## Governance

All implementations must adhere to the specified technological stack (Cohere, Qdrant, Neon Postgres) and architectural principles. Changes to core principles require explicit justification and approval. The system must maintain compliance with all privacy and security requirements. Code reviews must verify adherence to all constitutional principles before merging.

**Version**: 1.0.0 | **Ratified**: 2025-12-26 | **Last Amended**: 2025-12-26