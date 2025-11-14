# src/logger.py

import logging
import os
from datetime import datetime

# -----------------------------------------------------------------------------
# LOGGING CONFIGURATION
# -----------------------------------------------------------------------------

# Create a logs directory (if it doesn’t exist)
LOGS_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Generate a log file name based on the current date and time
LOG_FILE_NAME = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILE_NAME)

# -----------------------------------------------------------------------------
# Configure the logging format
# -----------------------------------------------------------------------------
logging.basicConfig(
    filename=LOG_FILE_PATH,
    filemode='a',  # Append to log file
    format="[%(asctime)s] %(levelname)s in %(name)s (line %(lineno)d): %(message)s",
    level=logging.INFO,
)

# -----------------------------------------------------------------------------
# Console (stream) handler for real-time logs in terminal
# -----------------------------------------------------------------------------
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")
console_handler.setFormatter(console_formatter)
logging.getLogger().addHandler(console_handler)

# -----------------------------------------------------------------------------
# Helper to get a custom logger per module
# -----------------------------------------------------------------------------
def get_logger(name: str = None) -> logging.Logger:
    """
    Returns a configured logger instance for the given module.
    Use this instead of importing logging directly.
    Example:
        from src.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Message")
    """
    return logging.getLogger(name or __name__)
