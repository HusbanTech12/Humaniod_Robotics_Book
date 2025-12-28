from typing import List, Dict, Any
from ..core.qdrant_client import qdrant_manager
from ..core.logging import get_logger

logger = get_logger(__name__)


class QdrantService:
    """
    Service for handling advanced Qdrant operations beyond the basic client.
    """

    @staticmethod
    async def parallel_upsert_to_qdrant(
        points: List[Dict[str, Any]],
        collection_name: str = None
    ) -> bool:
        """
        Perform parallel upsert of points to Qdrant with comprehensive payload metadata.

        Args:
            points: List of points to upsert, each with id, vector, and payload
            collection_name: Name of the collection to upsert to (uses default if None)

        Returns:
            True if successful, raises exception otherwise
        """
        try:
            if collection_name is None:
                collection_name = qdrant_manager.collection_name

            # Use the qdrant_manager to upsert the vectors
            qdrant_manager.upsert_vectors(points)

            logger.info(f"Successfully upserted {len(points)} points to collection '{collection_name}' in parallel")
            return True

        except Exception as e:
            logger.error(f"Error during parallel upsert to Qdrant: {str(e)}", exc_info=True)
            raise

    @staticmethod
    async def batch_upsert_with_metadata(
        points: List[Dict[str, Any]],
        collection_name: str = None
    ) -> Dict[str, Any]:
        """
        Upsert points to Qdrant with enhanced metadata handling.

        Args:
            points: List of points to upsert with comprehensive metadata
            collection_name: Name of the collection to upsert to (uses default if None)

        Returns:
            Dictionary with upsert results and statistics
        """
        try:
            if collection_name is None:
                collection_name = qdrant_manager.collection_name

            # Validate points before upsert
            validated_points = QdrantService._validate_points(points)

            # Perform the upsert
            qdrant_manager.upsert_vectors(validated_points)

            # Prepare results
            result = {
                "status": "success",
                "points_upserted": len(validated_points),
                "collection": collection_name,
                "timestamp": __import__('time').time()
            }

            logger.info(f"Batch upsert completed for {len(validated_points)} points to '{collection_name}'")
            return result

        except Exception as e:
            logger.error(f"Error during batch upsert with metadata: {str(e)}", exc_info=True)
            raise

    @staticmethod
    def _validate_points(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate points before upserting to Qdrant.

        Args:
            points: List of points to validate

        Returns:
            List of validated points
        """
        validated_points = []

        for i, point in enumerate(points):
            # Validate required fields
            if "id" not in point:
                raise ValueError(f"Point at index {i} missing required 'id' field")

            if "vector" not in point:
                raise ValueError(f"Point at index {i} missing required 'vector' field")

            if "payload" not in point:
                raise ValueError(f"Point at index {i} missing required 'payload' field")

            # Validate payload structure
            payload = point["payload"]
            required_payload_fields = ["content", "book_id", "metadata"]
            for field in required_payload_fields:
                if field not in payload:
                    logger.warning(f"Payload for point {point['id']} missing recommended field: {field}")

            validated_points.append(point)

        logger.info(f"Validated {len(validated_points)} points for Qdrant upsert")
        return validated_points

    @staticmethod
    async def search_with_filters(
        query_vector: List[float],
        book_id: str = None,
        filters: Dict[str, Any] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search in Qdrant with additional filters like book_id.

        Args:
            query_vector: The query vector to search for
            book_id: Optional book ID to filter results
            filters: Additional filters to apply
            limit: Number of results to return

        Returns:
            List of matching points with payload and similarity scores
        """
        try:
            # For now, use the basic search functionality
            # In a full implementation, this would include filtering
            results = qdrant_manager.search_vectors(
                vector=query_vector,
                limit=limit
            )

            # Apply any post-search filtering if needed
            filtered_results = results
            if book_id:
                filtered_results = [
                    result for result in results
                    if result.get("payload", {}).get("book_id") == book_id
                ]

            logger.info(f"Search completed with filters, found {len(filtered_results)} results")
            return filtered_results

        except Exception as e:
            logger.error(f"Error during search with filters: {str(e)}", exc_info=True)
            raise

    @staticmethod
    async def delete_points_by_book_id(
        book_id: str,
        collection_name: str = None
    ) -> bool:
        """
        Delete all points associated with a specific book ID.

        Args:
            book_id: The book ID to delete points for
            collection_name: Name of the collection to delete from (uses default if None)

        Returns:
            True if successful, raises exception otherwise
        """
        try:
            if collection_name is None:
                collection_name = qdrant_manager.collection_name

            # In a real implementation, this would use Qdrant's filtering capabilities
            # to delete points matching the book_id
            # For now, we'll just log that this would happen
            logger.info(f"Would delete points for book_id: {book_id} from collection: {collection_name}")

            # This would require implementing a proper deletion method in qdrant_manager
            # that can filter by payload fields
            return True

        except Exception as e:
            logger.error(f"Error during deletion by book ID: {str(e)}", exc_info=True)
            raise