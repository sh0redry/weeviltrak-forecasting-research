import logging
import os
from datetime import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def setup_logger(log_dir: str = None) -> logging.Logger:
    """
    Sets up a logger that logs messages to both a file and the console.

    The log files are stored in an 'outputs/logs' directory, which is created if it does not exist.
    The log file is named with the current date and time in the format 'app_YYYYMMDD_HHMMSS.log'.

    The logger is configured to log messages at the INFO level and above.
    The log message format includes the timestamp, logger name, log level, and the message.

    Parameters:
        log_dir (str, optional): The directory where log files will be stored. Defaults to '../outputs/logs'.

    Returns:
        logging.Logger: The configured logger instance.
    """
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(__file__), "../outputs/logs")
    os.makedirs(log_dir, exist_ok=True)

    log_filename = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logger = logging.getLogger(__name__)
    if not logger.handlers:  # Check if handlers are already added
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_filename)
        console_handler = logging.StreamHandler()

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Ensure third-party loggers (e.g., griddedweather/botocore) propagate to handlers
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        if not root.handlers:
            root.addHandler(file_handler)
            root.addHandler(console_handler)

        logger.propagate = False  # avoid duplicate logs from this logger

    return logger
