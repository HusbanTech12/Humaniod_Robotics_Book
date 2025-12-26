from typing import List, Dict, Any, Optional
from ..models.citation import Citation, LocationInfo
from ..core.logging import get_logger
from uuid import uuid4

logger = get_logger(__name__)


class CitationService:
    """
    Service for handling citation extraction and formatting.
    """

    @staticmethod
    def create_citation(
        response_id: str,
        source_text: str,
        location: Dict[str, Any],
        relevance_score: float = 0.0
    ) -> Citation:
        """
        Create a citation object from provided information.

        Args:
            response_id: The ID of the response this citation belongs to
            source_text: The text from the book that supports this citation
            location: Location information in the book
            relevance_score: Relevance score (0.0-1.0)

        Returns:
            A Citation object
        """
        # Validate inputs
        if not response_id:
            raise ValueError("Response ID is required for citation")
        if not source_text or not source_text.strip():
            raise ValueError("Source text is required for citation")

        # Create location info object
        location_info = LocationInfo(
            chapter=location.get('chapter'),
            section=location.get('section'),
            page=location.get('page'),
            paragraph=location.get('paragraph')
        )

        # Create citation
        citation = Citation(
            response_id=response_id,
            source_text=source_text.strip(),
            location=location_info,
            relevance_score=min(max(relevance_score, 0.0), 1.0)  # Clamp to 0.0-1.0
        )

        logger.info(f"Created citation with ID: {citation.id} for response: {response_id}")
        return citation

    @staticmethod
    def extract_citations_from_response(
        response_text: str,
        source_documents: List[Dict[str, Any]]
    ) -> List[Citation]:
        """
        Extract citations from a response based on source documents.

        Args:
            response_text: The generated response text
            source_documents: List of source documents with metadata

        Returns:
            List of Citation objects
        """
        citations = []

        for doc in source_documents:
            # Extract relevant information from the document
            source_text = doc.get('content', '')[:500]  # Limit to first 500 chars for the citation
            metadata = doc.get('metadata', {})
            doc_id = doc.get('id', str(uuid4()))

            # Create location info from metadata
            location = {
                'chapter': metadata.get('chapter'),
                'section': metadata.get('section'),
                'page': metadata.get('page'),
                'paragraph': metadata.get('paragraph')
            }

            # Create citation
            citation = CitationService.create_citation(
                response_id="",  # Will be set when we know the response ID
                source_text=source_text,
                location=location,
                relevance_score=doc.get('relevance_score', 0.5)
            )

            citations.append(citation)

        logger.info(f"Extracted {len(citations)} citations from source documents")
        return citations

    @staticmethod
    def format_citations_for_response(citations: List[Citation]) -> List[Dict[str, Any]]:
        """
        Format citations for inclusion in the API response.

        Args:
            citations: List of Citation objects

        Returns:
            List of citation dictionaries formatted for API response
        """
        formatted_citations = []

        for citation in citations:
            formatted_citations.append({
                'source_text': citation.source_text,
                'location': {
                    'chapter': citation.location.chapter,
                    'section': citation.location.section,
                    'page': citation.location.page,
                    'paragraph': citation.location.paragraph
                },
                'relevance_score': citation.relevance_score
            })

        logger.info(f"Formatted {len(formatted_citations)} citations for API response")
        return formatted_citations

    @staticmethod
    def validate_citation(citation: Citation) -> bool:
        """
        Validate a citation object.

        Args:
            citation: The citation to validate

        Returns:
            True if valid, raises ValueError otherwise
        """
        if not citation.source_text or not citation.source_text.strip():
            raise ValueError("Citation source text is required")

        if citation.relevance_score < 0.0 or citation.relevance_score > 1.0:
            raise ValueError("Citation relevance score must be between 0.0 and 1.0")

        # At least one location field should be provided
        location = citation.location
        if not any([
            location.chapter,
            location.section,
            location.page,
            location.paragraph
        ]):
            logger.warning(f"Citation {citation.id} has no location information")

        return True

    @staticmethod
    def merge_citations(citations: List[Citation]) -> List[Citation]:
        """
        Merge duplicate citations based on source text.

        Args:
            citations: List of citations to merge

        Returns:
            List of unique citations
        """
        seen_texts = set()
        unique_citations = []

        for citation in citations:
            text_key = citation.source_text.strip().lower()
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_citations.append(citation)

        logger.info(f"Merged {len(citations)} citations down to {len(unique_citations)} unique citations")
        return unique_citations

    @staticmethod
    async def extract_citations_during_streaming(
        response_chunk: str,
        source_documents: List[Dict[str, Any]]
    ) -> List[Citation]:
        """
        Extract citations from a response chunk during streaming.

        Args:
            response_chunk: A chunk of the response text
            source_documents: List of source documents with metadata

        Returns:
            List of Citation objects
        """
        citations = []

        # In a real implementation, this would use more sophisticated NLP to identify
        # references to source documents in the response chunk
        # For now, we'll do a simple keyword match

        for doc in source_documents:
            content = doc.get('content', '')
            # Look for content fragments in the response chunk
            if len(content) > 50:  # Only consider non-trivial documents
                # Simple approach: check if document content appears in the chunk
                if content[:100] in response_chunk:  # Check if first 100 chars appear in chunk
                    metadata = doc.get('metadata', {})

                    location_info = LocationInfo(
                        chapter=metadata.get('chapter'),
                        section=metadata.get('section'),
                        page=metadata.get('page'),
                        paragraph=metadata.get('paragraph')
                    )

                    citation = Citation(
                        response_id="",  # Will be set when we know the response ID
                        source_text=content[:200],  # Limit to first 200 chars
                        location=location_info,
                        relevance_score=metadata.get('relevance_score', 0.5)
                    )

                    citations.append(citation)

        logger.info(f"Extracted {len(citations)} citations from response chunk during streaming")
        return citations