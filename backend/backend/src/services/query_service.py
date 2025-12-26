from typing import Optional
from ..models.query import Query, QueryRequest, QueryMode
from ..core.logging import get_logger
from uuid import uuid4

logger = get_logger(__name__)


class QueryService:
    """
    Service for handling query validation and processing.
    """

    @staticmethod
    def validate_query(query_request: QueryRequest) -> bool:
        """
        Validate the incoming query request.

        Args:
            query_request: The query request to validate

        Returns:
            True if the query is valid, raises ValueError otherwise
        """
        # Check if query text is provided and not empty
        if not query_request.query or not query_request.query.strip():
            raise ValueError("Query text is required and cannot be empty")

        # Check query length
        if len(query_request.query) > 1000:
            raise ValueError("Query text exceeds maximum length of 1000 characters")

        # Validate book_id
        if not query_request.book_id or not query_request.book_id.strip():
            raise ValueError("Book ID is required")

        # If mode is selected-text, ensure selected_text is provided
        if query_request.mode == QueryMode.selected_text:
            if not query_request.selected_text or not query_request.selected_text.strip():
                raise ValueError("Selected text is required when mode is selected-text")

        logger.info(f"Query validation passed for book_id: {query_request.book_id}")
        return True

    @staticmethod
    def create_query_from_request(query_request: QueryRequest) -> Query:
        """
        Create an internal Query object from the request.

        Args:
            query_request: The query request

        Returns:
            A Query object ready for processing
        """
        # Validate the request first
        QueryService.validate_query(query_request)

        # Create the internal query object
        query = Query(
            text=query_request.query,
            mode=query_request.mode,
            selected_text=query_request.selected_text,
            book_id=query_request.book_id
        )

        logger.info(f"Created internal query object with ID: {query.id}")
        return query

    @staticmethod
    def validate_selected_text_mode(query: Query) -> bool:
        """
        Validate that the query is properly set up for selected-text mode.

        Args:
            query: The query to validate

        Returns:
            True if valid for selected-text mode, raises ValueError otherwise
        """
        if query.mode == QueryMode.selected_text:
            if not query.selected_text or not query.selected_text.strip():
                raise ValueError("Selected text is required for selected-text mode")
            return True
        return False

    @staticmethod
    def update_query_for_selected_text_mode(query: Query, selected_text: str) -> Query:
        """
        Update a query to be used in selected-text mode.

        Args:
            query: The query to update
            selected_text: The selected text to use for context

        Returns:
            Updated Query object in selected-text mode
        """
        if not selected_text or not selected_text.strip():
            raise ValueError("Selected text is required for selected-text mode")

        # Create a new query object with selected-text mode and the provided text
        updated_query = Query(
            id=query.id,
            text=query.text,
            mode=QueryMode.selected_text,
            selected_text=selected_text,
            book_id=query.book_id,
            timestamp=query.timestamp,
            session_id=query.session_id
        )

        logger.info(f"Updated query {query.id} for selected-text mode")
        return updated_query

    @staticmethod
    def validate_full_book_mode(query: Query) -> bool:
        """
        Validate that the query is properly set up for full-book mode.

        Args:
            query: The query to validate

        Returns:
            True if valid for full-book mode
        """
        if query.mode == QueryMode.full_book:
            # In full-book mode, we don't need selected_text
            return True
        return False

    @staticmethod
    def process_query_text(query_text: str) -> str:
        """
        Process and clean the query text.

        Args:
            query_text: The raw query text

        Returns:
            Cleaned and processed query text
        """
        # Strip leading/trailing whitespace
        processed_text = query_text.strip()

        # Additional processing can be added here (e.g., removing special characters, etc.)
        # For now, just return the stripped text
        return processed_text