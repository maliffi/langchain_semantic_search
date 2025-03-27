"""
Logger configuration module.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file
load_dotenv()

# Get log level from environment variables or default to INFO
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Create logs directory if it doesn't exist
logs_dir = Path(__file__).parent.parent.parent / "logs"
logs_dir.mkdir(exist_ok=True)

# Configure logger
logger.remove()  # Remove default handler

# Add console handler
logger.add(
    sys.stderr,
    level=LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)

# Add file handler for all logs
logger.add(
    logs_dir / "app.log",
    rotation="10 MB",  # Rotate when file reaches 10 MB
    retention="1 month",  # Keep logs for 1 month
    level=LOG_LEVEL,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
)

# Add file handler for error logs
logger.add(
    logs_dir / "error.log",
    rotation="10 MB",
    retention="1 month",
    level="ERROR",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
)


def get_logger(name: str):
    """
    Get a configured logger instance with the given name.

    Args:
        name: The name of the logger.

    Returns:
        A configured logger instance.
    """
    return logger.bind(name=name)
