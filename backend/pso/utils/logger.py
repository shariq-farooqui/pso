import logging
import os


def get_logger(name: str) -> logging.Logger:
    """Returns a logger object with the specified name.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: A logger object with the specified name.
    """
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    handler = logging.StreamHandler()
    handler.setLevel(log_level)

    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
