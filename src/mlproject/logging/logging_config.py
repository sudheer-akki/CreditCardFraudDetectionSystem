""" Module for debugging code and writing log """

import logging
import os
import sys

# Logging configuration
LOGGING_STR = "%(asctime)s: %(levelname)s: %(module)s: %(message)s"
#LOG_DIR = "logs"

def setup_logger(pkgname: str, log_dir: str = "logs"):
    """
    Sets up and returns a logger with the given package name.

    Args:
        pkgname (str): Name of the script or package for the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    LOG_FILEPATH = os.path.join(log_dir, "mlproject.log")
    # Create logger for the package
    logger = logging.getLogger(pkgname)
    logger.setLevel(logging.INFO)  # Default log level
    
    # Formatter for logs
    formatter = logging.Formatter(LOGGING_STR)
    
    # File handler
    file_handler = logging.FileHandler(LOG_FILEPATH)
    file_handler.setFormatter(formatter)
    
    # Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    
    # Prevent duplicate handlers
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    #Disable propagation to avoid duplicating logs in the root logger
    logger.propagate = False
    return logger
