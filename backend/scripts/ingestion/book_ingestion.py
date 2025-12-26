#!/usr/bin/env python3
"""
Book Ingestion Script

This script handles the ingestion of book content for the RAG chatbot.
It processes the book content, chunks it, generates embeddings, and stores it in the vector database.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add the backend src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend/src"))

from services.ingestion_service import IngestionService
from services.chunking_service import ChunkingService
from core.logging import setup_logging
from core.config import settings


async def load_book_content(file_path: str) -> str:
    """
    Load book content from a file.

    Args:
        file_path: Path to the book file

    Returns:
        The book content as a string
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Book file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"Loaded book content from {file_path} ({len(content)} characters)")
    return content


async def load_book_metadata(metadata_file: str) -> Dict[str, Any]:
    """
    Load book metadata from a JSON file.

    Args:
        metadata_file: Path to the metadata file

    Returns:
        The book metadata as a dictionary
    """
    if metadata_file and os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"Loaded metadata from {metadata_file}")
        return metadata
    else:
        print("No metadata file provided, using default metadata")
        return {}


async def main():
    parser = argparse.ArgumentParser(description="Ingest book content for RAG chatbot")
    parser.add_argument("--book-id", required=True, help="Unique identifier for the book")
    parser.add_argument("--title", required=True, help="Title of the book")
    parser.add_argument("--content-file", required=True, help="Path to the book content file")
    parser.add_argument("--metadata-file", help="Path to the book metadata file (JSON)")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Size of text chunks (default: 1000)")
    parser.add_argument("--overlap", type=int, default=100, help="Overlap between chunks (default: 100)")

    args = parser.parse_args()

    # Set up logging
    setup_logging()

    print(f"Starting ingestion for book: {args.book_id}")
    print(f"Title: {args.title}")
    print(f"Content file: {args.content_file}")
    print(f"Chunk size: {args.chunk_size}, Overlap: {args.overlap}")

    try:
        # Load book content
        content = await load_book_content(args.content_file)

        # Load metadata if provided
        metadata = await load_book_metadata(args.metadata_file)

        # Add command-line provided metadata
        metadata.update({
            "title": args.title,
            "source_file": args.content_file,
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "ingested_at": __import__('datetime').datetime.utcnow().isoformat()
        })

        # Perform the ingestion
        result = await IngestionService.ingest_book_content(
            book_id=args.book_id,
            title=args.title,
            content=content,
            metadata=metadata,
            chunk_size=args.chunk_size,
            overlap=args.overlap
        )

        print("\nIngestion completed successfully!")
        print(f"Book ID: {result['book_id']}")
        print(f"Title: {result['title']}")
        print(f"Chunks processed: {result['chunks_processed']}")
        print(f"Processing time: {result['processing_time_ms']}ms")

    except Exception as e:
        print(f"\nError during ingestion: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())