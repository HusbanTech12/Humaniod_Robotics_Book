from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from .secrets import SecretsManager
import logging

logger = logging.getLogger(__name__)

security = HTTPBearer()


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security), admin: bool = False) -> bool:
    """
    Verify the provided API key against stored keys.

    Args:
        credentials: HTTP authorization credentials
        admin: Whether to check against admin key (default: False)

    Returns:
        True if the key is valid, raises HTTPException otherwise
    """
    provided_key = credentials.credentials

    if not SecretsManager.validate_api_key(provided_key, admin=admin):
        logger.warning(f"Invalid API key provided: {provided_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    # Log successful authentication (without revealing the key)
    key_type = "admin" if admin else "public"
    logger.info(f"Successful {key_type} API key authentication")
    return True


def verify_admin_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> bool:
    """
    Verify the provided API key against the admin key.

    Args:
        credentials: HTTP authorization credentials

    Returns:
        True if the admin key is valid, raises HTTPException otherwise
    """
    return verify_api_key(credentials, admin=True)


def get_current_user_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Get the current user's API key after verification.

    Args:
        credentials: HTTP authorization credentials

    Returns:
        The verified API key
    """
    provided_key = credentials.credentials

    # Check if it's a regular API key
    if SecretsManager.validate_api_key(provided_key, admin=False):
        return provided_key
    # Check if it's an admin API key
    elif SecretsManager.validate_api_key(provided_key, admin=True):
        return provided_key
    else:
        logger.warning(f"Invalid API key provided: {provided_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )


class APIKeyValidator:
    """
    A class-based approach to API key validation with additional features.
    """

    @staticmethod
    def validate_key(provided_key: str, admin: bool = False) -> bool:
        """
        Validate an API key.

        Args:
            provided_key: The API key to validate
            admin: Whether to check against admin key

        Returns:
            True if valid, False otherwise
        """
        return SecretsManager.validate_api_key(provided_key, admin)

    @staticmethod
    def get_key_type(provided_key: str) -> Optional[str]:
        """
        Determine the type of API key.

        Args:
            provided_key: The API key to check

        Returns:
            'admin', 'public', or None if invalid
        """
        if SecretsManager.validate_api_key(provided_key, admin=True):
            return 'admin'
        elif SecretsManager.validate_api_key(provided_key, admin=False):
            return 'public'
        else:
            return None