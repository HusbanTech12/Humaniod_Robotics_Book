# RAG Chatbot for Published Book

An integrated Retrieval-Augmented Generation (RAG) Chatbot for Published Book (Cohere-Powered) that enables readers to ask questions about book content and receive accurate, cited responses.

## Features

- **Dual-mode operation**: Full-book queries and selected-text mode with strict contextual isolation
- **Real-time streaming**: Streaming responses with citations as they're generated
- **Privacy-first**: Client-side session management with no server-side personal data storage
- **Embeddable widget**: Lightweight, responsive widget for easy book integration
- **Cohere-powered**: Uses Cohere's Command R+ for generation and embed-multilingual-v3.0 for embeddings
- **Vector retrieval**: Qdrant Cloud Free Tier for semantic search capabilities
- **Cross-platform**: Responsive design works on desktop, tablet, and mobile devices

## Architecture

The system consists of:

1. **Backend API**: FastAPI-based service handling RAG operations
2. **Vector Database**: Qdrant Cloud for semantic search
3. **Metadata Storage**: Neon Serverless Postgres for chunk provenance
4. **Frontend Widget**: Embeddable JavaScript component
5. **Ingestion Pipeline**: Scripts for processing book content

## Tech Stack

- **Backend**: Python 3.11+, FastAPI, Uvicorn
- **AI/ML**: Cohere (embed-multilingual-v3.0, Command R+)
- **Vector DB**: Qdrant Cloud
- **Metadata DB**: Neon Serverless Postgres
- **Frontend**: Vanilla JavaScript, CSS
- **Embedding**: Script tag or iframe integration

## Installation

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables in `.env`:
   ```env
   COHERE_API_KEY=your_cohere_api_key
   QDRANT_URL=your_qdrant_cluster_url
   QDRANT_API_KEY=your_qdrant_api_key
   NEON_DATABASE_URL=your_neon_database_url
   API_KEY=your_public_api_key
   ADMIN_API_KEY=your_admin_api_key
   ```

4. Start the server:
   ```bash
   uvicorn src.api.main:app --reload --port 8000
   ```

### Widget Integration

The widget can be embedded in two ways:

#### Script Tag Method

```html
<div id="book-chatbot"></div>
<script src="/widget/dist/chat-widget.js"></script>
<script>
  BookChatWidget.init({
    containerId: 'book-chatbot',
    apiUrl: 'http://localhost:8000/v1',
    bookId: 'your-book-id',
    apiKey: 'your-public-api-key'
  });
</script>
```

#### Iframe Method

```html
<iframe
  src="/widget/embedded.html?bookId=your-book-id&apiKey=your-api-key"
  width="100%"
  height="500px"
  frameborder="0">
</iframe>
```

## API Endpoints

- `POST /v1/chat`: Submit a query and receive a response with citations
- `POST /v1/chat/stream`: Submit a query and receive a streaming response
- `POST /v1/ingest`: Ingest book content for RAG (admin only)
- `GET /v1/health`: Health check endpoint

## Book Ingestion

To ingest a book:

1. Prepare your book content in text format with structural metadata
2. Use the ingestion script:
   ```bash
   python scripts/ingestion/book_ingestion.py \
     --book-id "book-978-0123456789" \
     --title "Your Book Title" \
     --content-file "/path/to/book.txt" \
     --metadata-file "/path/to/metadata.json"
   ```

## Performance Targets

- ≤ 4 seconds p95 end-to-end latency under typical concurrent load
- ≥ 95% of responses are accurate, relevant, and properly cited
- Zero occurrences of full-corpus knowledge influencing selected-text responses
- 99% uptime during peak usage periods

## Privacy & Security

- No persistent storage of queries, responses, or user identifiers
- Client-side session management using localStorage only
- Transport security enforced with HTTPS
- Designed for GDPR/CCPA compliance through absence of personal data collection

## Development

The project follows a spec-driven development approach:

1. `/sp.constitution` - Project principles and constraints
2. `/sp.specify` - Feature requirements and user scenarios
3. `/sp.plan` - Technical architecture and implementation plan
4. `/sp.tasks` - Testable tasks with implementation cases
5. `/sp.implement` - Execute the implementation

## Contributing

See the contributing guidelines in the project documentation.

## License

MIT License - see the LICENSE file for details.