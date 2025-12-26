from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Optional, Any
from .config import settings
import logging

logger = logging.getLogger(__name__)


class QdrantManager:
    """
    Manages interactions with Qdrant vector database for book content storage and retrieval.
    """

    def __init__(self):
        """
        Initialize the Qdrant client with configuration from settings.
        """
        if settings.qdrant_api_key:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                prefer_grpc=True
            )
        else:
            self.client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port
            )

        self.collection_name = settings.qdrant_collection_name

    def create_collection(self, vector_size: int = 1024, distance: str = "Cosine"):
        """
        Create a collection in Qdrant for storing book content embeddings.

        Args:
            vector_size: Size of the embedding vectors (default 1024 for Cohere embeddings)
            distance: Distance metric for similarity search (default "Cosine")
        """
        try:
            # Check if collection already exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance[distance]
                    )
                )
                logger.info(f"Collection '{self.collection_name}' created successfully")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise

    def upsert_vectors(self, points: List[Dict[str, Any]]):
        """
        Upsert vectors into the Qdrant collection.

        Args:
            points: List of points to upsert, each with id, vector, and payload
        """
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Upserted {len(points)} vectors to collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error upserting vectors: {e}")
            raise

    def search_vectors(self, vector: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection.

        Args:
            vector: The query vector to search for
            limit: Number of results to return

        Returns:
            List of matching points with payload and similarity scores
        """
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=limit
            )

            # Format results to return relevant information
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'id': result.id,
                    'payload': result.payload,
                    'score': result.score
                })

            logger.info(f"Search completed, found {len(formatted_results)} results")
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.

        Returns:
            Dictionary with collection information
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'name': info.config.params.vectors.size,
                'vector_size': info.config.params.vectors.size,
                'distance': info.config.params.vectors.distance,
                'point_count': info.points_count
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            raise

    def delete_collection(self):
        """
        Delete the collection (useful for testing/reinitialization).
        """
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' deleted")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise


# Global instance
qdrant_manager = QdrantManager()