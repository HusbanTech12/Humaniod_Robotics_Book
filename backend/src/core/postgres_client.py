import asyncpg
from typing import List, Dict, Optional, Any
from .config import settings
from .secrets import SecretsManager
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PostgresManager:
    """
    Manages interactions with Postgres database for metadata storage.
    """

    def __init__(self):
        """
        Initialize the Postgres manager with connection parameters.
        """
        self.database_url = SecretsManager.get_neon_database_url()
        if not self.database_url:
            logger.warning("No Neon database URL configured")

    async def get_connection(self):
        """
        Get a database connection.
        """
        if self.database_url:
            return await asyncpg.connect(dsn=self.database_url)
        else:
            raise ValueError("No database URL configured")

    async def initialize_tables(self):
        """
        Initialize required tables in the database.
        """
        conn = await self.get_connection()
        try:
            # Create book_chunks table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS book_chunks (
                    id UUID PRIMARY KEY,
                    book_id VARCHAR(255) NOT NULL,
                    chunk_id VARCHAR(255) NOT NULL,
                    content_preview TEXT,
                    metadata JSONB NOT NULL,
                    token_count INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create indexes
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_book_chunks_book_id ON book_chunks(book_id)
            ''')
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_book_chunks_chunk_id ON book_chunks(chunk_id)
            ''')
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_book_chunks_metadata ON book_chunks USING GIN (metadata)
            ''')

            # Create analytics table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS analytics (
                    id UUID PRIMARY KEY,
                    event_type VARCHAR(50) NOT NULL,
                    session_id VARCHAR(255),
                    book_id VARCHAR(255) NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create indexes for analytics
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_analytics_session_id ON analytics(session_id)
            ''')
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_analytics_event_type ON analytics(event_type)
            ''')
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_analytics_created_at ON analytics(created_at)
            ''')

            logger.info("Database tables initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database tables: {e}")
            raise
        finally:
            await conn.close()

    async def insert_book_chunk(self, chunk_data: Dict[str, Any]):
        """
        Insert a book chunk into the database with synchronous insertion for provenance tracking.

        Args:
            chunk_data: Dictionary containing chunk information
        """
        conn = await self.get_connection()
        try:
            # Use synchronous insertion for provenance tracking
            await conn.execute('''
                INSERT INTO book_chunks (id, book_id, chunk_id, content_preview, metadata, token_count)
                VALUES ($1, $2, $3, $4, $5, $6)
            ''',
                chunk_data['id'],
                chunk_data['book_id'],
                chunk_data['chunk_id'],
                chunk_data['content_preview'][:200] if chunk_data['content_preview'] else None,  # First 200 chars
                json.dumps(chunk_data['metadata']),
                chunk_data['token_count']
            )
            logger.info(f"Synchronously inserted book chunk with ID: {chunk_data['id']} for provenance tracking")
        except Exception as e:
            logger.error(f"Error inserting book chunk synchronously: {e}")
            raise
        finally:
            await conn.close()

    async def get_book_chunks(self, book_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve book chunks for a specific book.

        Args:
            book_id: The ID of the book to retrieve chunks for
            limit: Maximum number of chunks to retrieve

        Returns:
            List of book chunks
        """
        conn = await self.get_connection()
        try:
            rows = await conn.fetch('''
                SELECT id, book_id, chunk_id, content_preview, metadata, token_count, created_at
                FROM book_chunks
                WHERE book_id = $1
                ORDER BY created_at
                LIMIT $2
            ''', book_id, limit)

            chunks = []
            for row in rows:
                chunks.append({
                    'id': str(row['id']),
                    'book_id': row['book_id'],
                    'chunk_id': row['chunk_id'],
                    'content_preview': row['content_preview'],
                    'metadata': row['metadata'],
                    'token_count': row['token_count'],
                    'created_at': row['created_at']
                })

            logger.info(f"Retrieved {len(chunks)} chunks for book {book_id}")
            return chunks
        except Exception as e:
            logger.error(f"Error retrieving book chunks: {e}")
            raise
        finally:
            await conn.close()

    async def log_analytics_event(self, event_type: str, session_id: Optional[str], book_id: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Log an analytics event.

        Args:
            event_type: Type of event (e.g., 'query_submitted', 'response_generated')
            session_id: Session identifier (anonymous)
            book_id: Book identifier
            metadata: Additional event metadata
        """
        conn = await self.get_connection()
        try:
            await conn.execute('''
                INSERT INTO analytics (id, event_type, session_id, book_id, metadata)
                VALUES (gen_random_uuid(), $1, $2, $3, $4)
            ''',
                event_type,
                session_id,
                book_id,
                json.dumps(metadata) if metadata else None
            )
            logger.info(f"Logged analytics event: {event_type}")
        except Exception as e:
            logger.error(f"Error logging analytics event: {e}")
            raise
        finally:
            await conn.close()

    async def get_analytics(self, event_type: Optional[str] = None, book_id: Optional[str] = None, days: int = 30) -> List[Dict[str, Any]]:
        """
        Retrieve analytics data.

        Args:
            event_type: Optional event type filter
            book_id: Optional book ID filter
            days: Number of days to look back

        Returns:
            List of analytics events
        """
        conn = await self.get_connection()
        try:
            query = '''
                SELECT id, event_type, session_id, book_id, metadata, created_at
                FROM analytics
                WHERE created_at >= NOW() - INTERVAL '$3 days'
            '''
            params = [days]

            if event_type:
                query += ' AND event_type = $4'
                params.append(event_type)
            if book_id:
                query += ' AND book_id = $5'
                params.append(book_id)

            query += ' ORDER BY created_at DESC'

            rows = await conn.fetch(query, *params)

            events = []
            for row in rows:
                events.append({
                    'id': str(row['id']),
                    'event_type': row['event_type'],
                    'session_id': row['session_id'],
                    'book_id': row['book_id'],
                    'metadata': row['metadata'],
                    'created_at': row['created_at']
                })

            logger.info(f"Retrieved {len(events)} analytics events")
            return events
        except Exception as e:
            logger.error(f"Error retrieving analytics: {e}")
            raise
        finally:
            await conn.close()


# Global instance
postgres_manager = PostgresManager()