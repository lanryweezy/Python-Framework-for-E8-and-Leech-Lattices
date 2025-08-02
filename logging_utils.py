"""
Logging utilities for E8 Leech Lattice Framework
"""

import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """
    Sets up logging for the application.
    
    Args:
        log_level (str): The minimum level of messages to log (e.g., "INFO", "DEBUG").
        log_file (str, optional): Path to a file where logs should be written. If None, logs only to console.
    """
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level.upper())

    # Clear existing handlers to prevent duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level.upper())
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_file_path = Path(os.path.expanduser(log_file))
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf8",
        )
        file_handler.setLevel(log_level.upper())
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger instance for a given name.
    
    Args:
        name (str): The name of the logger (usually __name__ of the module).
        
    Returns:
        logging.Logger: The logger instance.
    """
    return logging.getLogger(name)

# Initialize logging with default settings
setup_logging()

