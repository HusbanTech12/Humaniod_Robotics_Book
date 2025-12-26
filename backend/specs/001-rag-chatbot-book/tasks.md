---
description: "Task list template for feature implementation"
---

# Tasks: RAG Chatbot for Published Book

**Input**: Design documents from `/specs/001-rag-chatbot-book/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: The examples below include test tasks. Tests are OPTIONAL - only include them if explicitly requested in the feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- **Web app**: `backend/src/`, `frontend/src/`
- **Mobile**: `api/src/`, `ios/src/` or `android/src/`
- Paths shown below assume single project - adjust based on plan.md structure

<!--
  ============================================================================
  IMPORTANT: The tasks below are SAMPLE TASKS for illustration purposes only.

  The /sp.tasks command MUST replace these with actual tasks based on:
  - User stories from spec.md (with their priorities P1, P2, P3...)
  - Feature requirements from plan.md
  - Entities from data-model.md
  - Endpoints from contracts/

  Tasks MUST be organized by user story so each story can be:
  - Implemented independently
  - Tested independently
  - Delivered as an MVP increment

  DO NOT keep these sample tasks in the generated tasks.md file.
  ============================================================================
-->

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Create project structure per implementation plan in backend/, widget/, scripts/
- [x] T002 Initialize Python project with FastAPI, cohere, qdrant-client, asyncpg, pydantic v2 dependencies in backend/requirements.txt
- [x] T003 [P] Configure linting and formatting tools (black, ruff) in backend/
- [x] T004 [P] Set up environment configuration management in backend/.env.example

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

Examples of foundational tasks (adjust based on your project):

- [x] T005 Set up database schema and migrations framework in backend/src/core/database.py
- [x] T006 [P] Implement authentication/authorization framework in backend/src/core/auth.py
- [x] T007 [P] Setup API routing and middleware structure in backend/src/api/main.py
- [x] T008 Create base models/entities that all stories depend on in backend/src/models/query.py, backend/src/models/response.py, backend/src/models/citation.py
- [x] T009 Configure error handling and logging infrastructure in backend/src/core/logging.py
- [x] T010 Setup environment configuration management in backend/src/core/config.py
- [x] T011 [P] Create Qdrant client and connection utilities in backend/src/core/qdrant_client.py
- [x] T012 [P] Create Cohere client and connection utilities in backend/src/core/cohere_client.py
- [x] T013 [P] Create Neon Postgres client and connection utilities in backend/src/core/postgres_client.py
- [x] T014 [P] Implement secure secrets management in backend/src/core/secrets.py

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Book Query via Chat Interface (Priority: P1) 🎯 MVP

**Goal**: Users can ask questions about the published book content through a chat interface and receive accurate, cited responses based on the book's content

**Independent Test**: Can be fully tested by entering various questions about book content and verifying that responses are accurate, relevant, and properly cited from the book

### Tests for User Story 1 (OPTIONAL - only if tests requested) ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T015 [P] [US1] Contract test for /chat endpoint in backend/tests/contract/test_chat.py
- [ ] T016 [P] [US1] Integration test for full-book query user journey in backend/tests/integration/test_chat_full_book.py

### Implementation for User Story 1

- [x] T017 [P] [US1] Create Query model in backend/src/models/query.py (based on data-model.md)
- [x] T018 [P] [US1] Create Response model in backend/src/models/response.py (based on data-model.md)
- [x] T019 [P] [US1] Create Citation model in backend/src/models/citation.py (based on data-model.md)
- [x] T020 [US1] Implement QueryService in backend/src/services/query_service.py (handles query validation and processing)
- [x] T021 [US1] Implement RAGService in backend/src/services/rag_service.py (handles retrieval and generation)
- [x] T022 [US1] Implement CitationService in backend/src/services/citation_service.py (handles citation extraction and formatting)
- [x] T023 [US1] Implement /chat endpoint in backend/src/api/endpoints/chat.py (based on contracts/chat-api.yaml)
- [x] T024 [US1] Add validation and error handling to chat endpoint
- [x] T025 [US1] Add logging for user story 1 operations
- [x] T026 [US1] Implement basic retrieval from Qdrant in backend/src/services/retrieval_service.py
- [x] T027 [US1] Implement Cohere generation in backend/src/services/generation_service.py

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Selected Text Context Activation (Priority: P2)

**Goal**: Users can select text within the book and activate the AI chatbot to ask questions specifically about that selected text, with the system providing responses grounded only in that text segment

**Independent Test**: Can be fully tested by selecting text, activating the contextual "Ask AI" trigger, and verifying responses are restricted to the selected text without incorporating broader book knowledge

### Tests for User Story 2 (OPTIONAL - only if tests requested) ⚠️

- [ ] T028 [P] [US2] Contract test for /chat endpoint with selected-text mode in backend/tests/contract/test_chat_selected_text.py
- [ ] T029 [P] [US2] Integration test for selected-text query user journey in backend/tests/integration/test_chat_selected_text.py

### Implementation for User Story 2

- [x] T030 [P] [US2] Update Query model to handle selected-text mode in backend/src/models/query.py
- [x] T031 [US2] Enhance QueryService to handle selected-text mode in backend/src/services/query_service.py
- [x] T032 [US2] Update RAGService to implement strict dual-mode enforcement in backend/src/services/rag_service.py
- [x] T033 [US2] Implement direct document grounding for selected-text mode in backend/src/services/rag_service.py
- [x] T034 [US2] Add mode validation and routing logic to /chat endpoint in backend/src/api/endpoints/chat.py
- [x] T035 [US2] Add comprehensive logging for mode isolation validation

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Real-time Streaming Response with Citations (Priority: P3)

**Goal**: Users receive real-time streaming responses from the chatbot with clickable and highlighted citations that link to relevant sections of the book

**Independent Test**: Can be fully tested by submitting queries and verifying that responses stream in real-time with properly formatted citations that link to book sections

### Tests for User Story 3 (OPTIONAL - only if tests requested) ⚠️

- [ ] T036 [P] [US3] Contract test for /chat/stream endpoint in backend/tests/contract/test_stream.py
- [ ] T037 [P] [US3] Integration test for streaming response user journey in backend/tests/integration/test_streaming.py

### Implementation for User Story 3

- [x] T038 [P] [US3] Create StreamChunk model in backend/src/models/stream_chunk.py
- [x] T039 [US3] Implement streaming response functionality in backend/src/services/streaming_service.py
- [x] T040 [US3] Implement /chat/stream endpoint in backend/src/api/endpoints/stream.py (based on contracts/chat-api.yaml)
- [x] T041 [US3] Update RAGService to support streaming responses in backend/src/services/rag_service.py
- [x] T042 [US3] Implement citation extraction during streaming in backend/src/services/citation_service.py
- [x] T043 [US3] Add SSE (Server-Sent Events) support for streaming in backend/src/api/endpoints/stream.py

**Checkpoint**: At this point, User Stories 1, 2 AND 3 should all work independently

---

## Phase 6: User Story 4 - Mobile-Responsive Chat Interface (Priority: P2)

**Goal**: Users can access and interact with the RAG chatbot seamlessly across desktop, tablet, and mobile devices with a responsive design that maintains functionality and usability

**Independent Test**: Can be fully tested by accessing the interface on different devices and verifying that all functionality remains accessible and usable

### Tests for User Story 4 (OPTIONAL - only if tests requested) ⚠️

- [ ] T044 [P] [US4] Contract test for widget API integration in widget/tests/contract/test_widget_api.js
- [ ] T045 [P] [US4] Integration test for responsive UI behavior in widget/tests/integration/test_responsive.js

### Implementation for User Story 4

- [x] T046 [P] [US4] Create widget project structure in widget/src/
- [x] T047 [P] [US4] Create core widget components in widget/src/components/
- [x] T048 [P] [US4] Create responsive UI styles in widget/src/styles/
- [x] T049 [US4] Implement chat interface component in widget/src/components/ChatInterface.js
- [x] T050 [US4] Implement responsive design with CSS media queries in widget/src/styles/chat-widget.css
- [x] T051 [US4] Implement widget initialization and embedding logic in widget/src/index.js
- [x] T052 [US4] Add mobile-specific interactions and touch support in widget/src/components/MobileSupport.js
- [x] T053 [US4] Implement iframe and script tag embedding methods in widget/src/embed.js

**Checkpoint**: At this point, User Stories 1, 2, 3 AND 4 should all work independently

---

## Phase 7: User Story 5 - Client-Side Session Management (Priority: P3)

**Goal**: Users can maintain their chat session using client-side storage (localStorage) without exposing personal information to the server, maintaining privacy compliance

**Independent Test**: Can be fully tested by verifying that session data is stored client-side only and no personal information is transmitted to the server

### Tests for User Story 5 (OPTIONAL - only if tests requested) ⚠️

- [ ] T054 [P] [US5] Contract test for session management in widget/tests/contract/test_session.js
- [ ] T055 [P] [US5] Integration test for client-side session persistence in widget/tests/integration/test_session_storage.js

### Implementation for User Story 5

- [x] T056 [P] [US5] Create Session model for client-side use in widget/src/models/session.js
- [x] T057 [US5] Implement localStorage session management in widget/src/services/session_storage.js
- [x] T058 [US5] Add session persistence to chat interface in widget/src/components/ChatInterface.js
- [x] T059 [US5] Implement session restoration on widget initialization in widget/src/index.js
- [x] T060 [US5] Add privacy compliance validation to session management

**Checkpoint**: All user stories should now be independently functional

---

[Add more user story phases as needed, following the same pattern]

---

## Phase 8: Book Ingestion Pipeline

**Goal**: Implement the book ingestion pipeline to prepare book content for RAG operations

**Independent Test**: Can be tested by ingesting sample book content and verifying that chunks are properly stored in Qdrant and Postgres with accurate metadata

### Tests for Book Ingestion (OPTIONAL - only if tests requested) ⚠️

- [ ] T061 [P] Contract test for /ingest endpoint in backend/tests/contract/test_ingest.py
- [ ] T062 [P] Integration test for book ingestion pipeline in backend/tests/integration/test_ingestion.py

### Implementation for Book Ingestion

- [x] T063 Create Book Content Chunk model in backend/src/models/book_chunk.py (based on data-model.md)
- [x] T064 Implement semantic-aware chunking with controlled overlap in backend/src/services/chunking_service.py
- [x] T065 Implement ingestion core with batch embedding in backend/src/services/ingestion_service.py
- [x] T066 Implement /ingest endpoint in backend/src/api/endpoints/ingest.py (based on contracts/chat-api.yaml)
- [x] T067 Implement parallel upsert to Qdrant with comprehensive payload metadata in backend/src/services/qdrant_service.py
- [x] T068 Implement synchronous metadata insertion into Neon for provenance tracking in backend/src/services/postgres_service.py
- [x] T069 Create ingestion script in scripts/ingestion/book_ingestion.py

**Checkpoint**: Book ingestion pipeline should be fully functional

---

## Phase 9: Widget Integration and Selected-Text Detection

**Goal**: Implement the contextual "Ask AI" trigger that activates when users highlight text in the book

**Independent Test**: Can be tested by selecting text in a book page and verifying that the "Ask AI" trigger appears and functions correctly

### Implementation for Widget Integration

- [ ] T070 Implement selected-text detection in widget/src/components/TextSelection.js
- [ ] T071 Implement contextual "Ask AI" trigger in widget/src/components/ContextMenu.js
- [ ] T072 Integrate selected-text mode with chat interface in widget/src/components/ChatInterface.js
- [ ] T073 Add cross-origin communication support in widget/src/services/cross_origin.js

**Checkpoint**: Selected-text functionality should be fully integrated with the widget

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T074 [P] Documentation updates in docs/
- [ ] T075 Code cleanup and refactoring
- [ ] T076 Performance optimization across all stories
- [ ] T077 [P] Additional unit tests (if requested) in backend/tests/unit/
- [ ] T078 Security hardening
- [ ] T079 Run quickstart.md validation
- [ ] T080 Implement health check endpoint in backend/src/api/endpoints/health.py
- [ ] T081 Add rate limiting to API endpoints in backend/src/middleware/rate_limit.py
- [ ] T082 Configure CORS restrictions for embeddable widget in backend/src/middleware/cors.py

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - May integrate with US1 but should be independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - May integrate with US1/US2 but should be independently testable
- **User Story 4 (P2)**: Can start after Foundational (Phase 2) - May integrate with US1/US2/US3 but should be independently testable
- **User Story 5 (P3)**: Can start after Foundational (Phase 2) - May integrate with other stories but should be independently testable

### Within Each User Story

- Tests (if included) MUST be written and FAIL before implementation
- Models before services
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All tests for a user story marked [P] can run in parallel
- Models within a story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together (if tests requested):
Task: "Contract test for /chat endpoint in backend/tests/contract/test_chat.py"
Task: "Integration test for full-book query user journey in backend/tests/integration/test_chat_full_book.py"

# Launch all models for User Story 1 together:
Task: "Create Query model in backend/src/models/query.py"
Task: "Create Response model in backend/src/models/response.py"
Task: "Create Citation model in backend/src/models/citation.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Add User Story 4 → Test independently → Deploy/Demo
6. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence