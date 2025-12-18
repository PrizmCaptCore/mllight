"""Logging utilities"""

import logging
from typing import Optional


def setup_logger(
    name: str = __name__,
    level: str = "info",
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup and configure logger

    Args:
        name: Logger name
        level: Logging level (debug, info, warning, error)
        format_string: Custom format string

    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = '%(asctime)s - %(levelname)s - %(message)s'

    # Convert string level to logging level
    level = level.upper()
    numeric_level = getattr(logging, level, logging.INFO)

    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        force=True  # Override existing config
    )

    logger = logging.getLogger(name)

    return logger
