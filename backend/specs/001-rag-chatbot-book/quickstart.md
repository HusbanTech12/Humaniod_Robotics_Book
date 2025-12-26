# Quickstart Guide: RAG Chatbot for Published Book

**Feature**: RAG Chatbot for Published Book
**Date**: 2025-12-26
**Branch**: 001-rag-chatbot-book

## Overview

This guide provides a quick setup and usage guide for the RAG Chatbot system. The system enables readers to ask questions about a published book and receive accurate, cited responses based on the book's content.

## Prerequisites

- Python 3.11+
- Access to Cohere API (Command R+ and embed-multilingual-v3.0)
- Access to Qdrant Cloud (Free Tier endpoint and API key)
- Access to Neon Serverless Postgres
- Book content in text format with structural metadata

## Environment Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Install backend dependencies:**
   ```bash
   cd backend
   pip install fastapi uvicorn cohere qdrant-client asyncpg pydantic python-multipart
   ```

3. **Set up environment variables:**
   Create a `.env` file in the backend directory:
   ```env
   COHERE_API_KEY=your_cohere_api_key
   QDRANT_URL=https://93db2543-e764-4df1-8365-f25f65dd9e5e.us-east4-0.gcp.cloud.qdrant.io
   QDRANT_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.UV8eLKPeJsr5WJB1KRKK1V8DS8nub1Td41clNGemrFI
   DATABASE_URL=your_neon_postgres_connection_string
   API_KEY=your_public_api_key
   ADMIN_API_KEY=your_admin_api_key_for_ingestion
   ```

## Backend Setup

1. **Start the backend server:**
   ```bash
   cd backend
   uvicorn src.main:app --reload --port 8000
   ```

2. **Verify the service is running:**
   - Health check: `GET http://localhost:8000/health`
   - API docs: `http://localhost:8000/docs`

## Book Ingestion

1. **Prepare your book content** in JSON format with structural metadata:
   ```json
   {
     "book_id": "book-978-0123456789",
     "title": "Your Book Title",
     "content": "Full book content with structural markers...",
     "metadata": {
       "chapters": [
         {
           "id": "ch1",
           "title": "Chapter 1",
           "start_page": 1,
           "end_page": 25
         }
       ]
     }
   }
   ```

2. **Ingest the book content:**
   ```bash
   curl -X POST http://localhost:8000/ingest \
     -H "Content-Type: application/json" \
     -H "X-Admin-API-Key: your_admin_api_key" \
     -d @book_content.json
   ```

## Using the Chat API

1. **Full-book mode query:**
   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -H "X-API-Key: your_public_api_key" \
     -d '{
       "query": "What are the main themes in the book?",
       "book_id": "book-978-0123456789",
       "mode": "full-book"
     }'
   ```

2. **Selected-text mode query:**
   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -H "X-API-Key: your_public_api_key" \
     -d '{
       "query": "What does this section mean?",
       "book_id": "book-978-0123456789",
       "mode": "selected-text",
       "selected_text": "This is the specific text the user has selected..."
     }'
   ```

## Frontend Widget Integration

1. **Include the widget in your book page:**
   ```html
   <!-- Option 1: Script tag -->
   <script src="/widget/dist/chat-widget.js"></script>
   <div id="book-chatbot"></div>
   <script>
     BookChatWidget.init({
       containerId: 'book-chatbot',
       apiUrl: 'http://localhost:8000',
       bookId: 'book-978-0123456789',
       apiKey: 'your_public_api_key'
     });
   </script>
   ```

2. **Or use iframe embedding:**
   ```html
   <iframe
     src="/widget/embedded.html?bookId=book-978-0123456789&apiKey=your_public_api_key"
     width="100%"
     height="500px"
     frameborder="0">
   </iframe>
   ```

## Testing the System

1. **Run backend tests:**
   ```bash
   cd backend
   pytest tests/
   ```

2. **Test dual-mode functionality:**
   - Verify full-book mode queries return results from entire corpus
   - Verify selected-text mode queries only use provided text
   - Confirm citations are properly formatted and linkable

3. **Performance test:**
   - Check response latency is ≤ 4 seconds for 95th percentile
   - Verify system handles concurrent requests appropriately

## Deployment

1. **Deploy backend to production:**
   - Deploy to Render, Railway, or Fly.io with secret management
   - Ensure environment variables are properly configured

2. **Deploy widget to CDN:**
   - Build the widget: `npm run build` in the widget directory
   - Upload `widget/dist/` to a CDN (Vercel, Netlify, Cloudflare Pages)

3. **Final ingestion in production:**
   - Execute book ingestion in production environment
   - Verify chatbot functionality within the published book

## Troubleshooting

- **API rate limits**: Check Cohere and Qdrant usage metrics
- **Slow responses**: Verify vector database indexing and network latency
- **Citation issues**: Confirm book content was properly chunked with metadata
- **CORS errors**: Check backend CORS configuration for widget domain

## Next Steps

- Review the complete API documentation at `/docs`
- Explore the data model and system architecture
- Set up monitoring and observability for production use
- Configure custom styling for the chat widget to match your book's design