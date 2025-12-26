import logging
import sys
from typing import Optional
from .config import settings


def setup_logging() -> None:
    """
    Set up logging configuration for the application.
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if settings.debug else logging.INFO)

    # Create console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if settings.debug else logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)

    # Add handler to root logger
    if not root_logger.handlers:
        root_logger.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: The name of the logger

    Returns:
        A configured logger instance
    """
    logger = logging.getLogger(name)
    return logger


# Initialize logging
setup_logging()