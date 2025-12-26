from typing import List, Dict, Any
from ..models.book_chunk import BookChunk
from ..core.logging import get_logger
import re

logger = get_logger(__name__)


class ChunkingService:
    """
    Service for handling semantic-aware chunking of book content with controlled overlap.
    """

    @staticmethod
    def chunk_content(
        content: str,
        book_id: str,
        metadata: Dict[str, Any],
        chunk_size: int = 1000,  # Target characters per chunk
        overlap: int = 100,      # Overlap characters between chunks
        min_chunk_size: int = 200  # Minimum chunk size before discarding
    ) -> List[BookChunk]:
        """
        Split content into overlapping chunks based on semantic boundaries.

        Args:
            content: The content to chunk
            book_id: The book ID these chunks belong to
            metadata: Metadata to include with each chunk
            chunk_size: Target size of each chunk in characters
            overlap: Number of overlapping characters between chunks
            min_chunk_size: Minimum size for a chunk to be included

        Returns:
            List of BookChunk objects
        """
        if chunk_size <= overlap:
            raise ValueError("Chunk size must be greater than overlap")

        # Try to split on semantic boundaries (paragraphs, sentences) first
        potential_splits = ChunkingService._find_semantic_boundaries(content)

        # If we don't have good semantic boundaries, fall back to character-based splitting
        if len(potential_splits) < 2:
            logger.info("No semantic boundaries found, using character-based splitting")
            potential_splits = list(range(0, len(content), max(1, chunk_size // 2)))

        chunks = []
        start = 0

        while start < len(content):
            # Find the best end point for this chunk
            end = start + chunk_size

            # Don't exceed content length
            if end > len(content):
                end = len(content)

            # Try to find a good semantic boundary within the target range
            best_split = end
            for split_point in reversed(potential_splits):
                if start + chunk_size * 0.7 <= split_point <= start + chunk_size * 1.3:
                    best_split = split_point
                    break

            # If we found a good semantic split within range, use it
            if start + min_chunk_size <= best_split < end:
                end = best_split

            # Extract the chunk
            chunk_text = content[start:end]

            # Skip if chunk is too small (unless it's the last chunk)
            if len(chunk_text) < min_chunk_size and end < len(content):
                start = end
                continue

            # Create metadata for this specific chunk
            chunk_metadata = metadata.copy()
            chunk_metadata["position"] = len(chunks)  # Position in the book
            chunk_metadata["start_offset"] = start
            chunk_metadata["end_offset"] = end
            chunk_metadata["chunk_size"] = len(chunk_text)

            # Create the chunk object
            chunk = BookChunk(
                book_id=book_id,
                content=chunk_text,
                metadata=chunk_metadata,
                token_count=len(chunk_text.split())  # Rough token count
            )

            chunks.append(chunk)

            # Move to the next chunk position with overlap
            next_start = end - overlap
            if next_start <= start:  # Prevent infinite loop
                next_start = start + max(len(chunk_text), 1)
            start = next_start

            # If we've reached the end, break
            if end == len(content):
                break

        logger.info(f"Created {len(chunks)} chunks from content of length {len(content)} with target size {chunk_size}")
        return chunks

    @staticmethod
    def _find_semantic_boundaries(content: str) -> List[int]:
        """
        Find potential semantic boundaries in the content.

        Args:
            content: The content to analyze

        Returns:
            List of character positions that are good splitting points
        """
        boundaries = set()

        # Find paragraph boundaries (double newlines)
        for match in re.finditer(r'\n\s*\n', content):
            boundaries.add(match.end())

        # Find sentence boundaries (periods, question marks, exclamation marks)
        # followed by whitespace and capital letter or end of content
        for match in re.finditer(r'[.!?]\s+(?=[A-Z])|[.!?]$', content):
            boundaries.add(match.end())

        # Find section headers (lines that look like headers)
        lines = content.split('\n')
        current_pos = 0
        for i, line in enumerate(lines):
            if ChunkingService._is_header_line(line):
                # Add boundary after the header
                header_end = current_pos + len(line) + 1  # +1 for the newline
                boundaries.add(header_end)
            current_pos += len(line) + 1  # +1 for the newline

        # Convert to sorted list
        result = sorted(list(boundaries))

        # Remove boundaries that are too close to each other (less than 100 chars)
        filtered_result = []
        for pos in result:
            if not filtered_result or pos - filtered_result[-1] >= 100:
                filtered_result.append(pos)

        return filtered_result

    @staticmethod
    def _is_header_line(line: str) -> bool:
        """
        Determine if a line looks like a header/section title.

        Args:
            line: The line to check

        Returns:
            True if the line appears to be a header
        """
        line = line.strip()
        if not line:
            return False

        # Check for common header patterns
        # - Short lines (less than 50 chars)
        # - Lines ending with punctuation that might be titles
        # - Lines that are mostly uppercase (indicating titles)
        is_short = len(line) < 50
        has_title_punct = line.endswith((':', '.', '—', '-'))
        is_mostly_upper = sum(1 for c in line if c.isupper()) / len(line) > 0.6 if line else False

        # Check if it looks like a numbered section (e.g., "1. Introduction")
        is_numbered_section = bool(re.match(r'^\d+[\.\-)\s]\s*\w', line))

        return is_short and (has_title_punct or is_mostly_upper or is_numbered_section)

    @staticmethod
    def validate_chunking_parameters(
        chunk_size: int,
        overlap: int,
        min_chunk_size: int
    ) -> bool:
        """
        Validate chunking parameters.

        Args:
            chunk_size: The chunk size to validate
            overlap: The overlap to validate
            min_chunk_size: The minimum chunk size to validate

        Returns:
            True if parameters are valid, raises ValueError otherwise
        """
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")

        if overlap < 0:
            raise ValueError("Overlap cannot be negative")

        if overlap >= chunk_size:
            raise ValueError("Overlap must be less than chunk size")

        if min_chunk_size <= 0:
            raise ValueError("Minimum chunk size must be positive")

        if min_chunk_size >= chunk_size:
            raise ValueError("Minimum chunk size must be less than chunk size")

        logger.info(f"Validated chunking parameters: size={chunk_size}, overlap={overlap}, min={min_chunk_size}")
        return True

    @staticmethod
    def get_content_statistics(content: str) -> Dict[str, Any]:
        """
        Get statistics about the content to help with chunking decisions.

        Args:
            content: The content to analyze

        Returns:
            Dictionary with content statistics
        """
        lines = content.split('\n')
        paragraphs = [p for p in content.split('\n\n') if p.strip()]

        stats = {
            'total_chars': len(content),
            'total_words': len(content.split()),
            'total_lines': len(lines),
            'total_paragraphs': len(paragraphs),
            'avg_line_length': sum(len(line) for line in lines) / len(lines) if lines else 0,
            'avg_paragraph_length': sum(len(p) for p in paragraphs) / len(paragraphs) if paragraphs else 0,
        }

        logger.debug(f"Content statistics: {stats}")
        return stats