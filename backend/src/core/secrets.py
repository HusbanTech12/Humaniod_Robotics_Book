import os
from typing import Optional
from .config import settings


class SecretsManager:
    """
    Manages secure access to sensitive information like API keys.
    """

    @staticmethod
    def get_cohere_api_key() -> Optional[str]:
        """
        Get the Cohere API key from environment variables.

        Returns:
            The Cohere API key or None if not set
        """
        return settings.cohere_api_key

    @staticmethod
    def get_qdrant_api_key() -> Optional[str]:
        """
        Get the Qdrant API key from environment variables.

        Returns:
            The Qdrant API key or None if not set
        """
        return settings.qdrant_api_key

    @staticmethod
    def get_database_url() -> Optional[str]:
        """
        Get the database URL from environment variables.

        Returns:
            The database URL or None if not set
        """
        return settings.database_url

    @staticmethod
    def get_neon_database_url() -> Optional[str]:
        """
        Get the Neon database URL from environment variables.

        Returns:
            The Neon database URL or None if not set
        """
        return settings.neon_database_url

    @staticmethod
    def get_api_key() -> str:
        """
        Get the public API key.

        Returns:
            The public API key
        """
        return settings.api_key

    @staticmethod
    def get_admin_api_key() -> str:
        """
        Get the admin API key.

        Returns:
            The admin API key
        """
        return settings.admin_api_key

    @staticmethod
    def validate_api_key(provided_key: str, admin: bool = False) -> bool:
        """
        Validate an API key against the stored keys.

        Args:
            provided_key: The API key to validate
            admin: Whether to check against admin key (default: False)

        Returns:
            True if the key is valid, False otherwise
        """
        if admin:
            return provided_key == settings.admin_api_key
        else:
            return provided_key == settings.api_key