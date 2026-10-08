import logging
import os

from CoScientist.config import get_settings

settings = get_settings()

# Use a named logger, NOT the root logger — configuring the root hijacks logging
# for the whole process (every third-party library).
logger = logging.getLogger("CoScientist")
logger.setLevel(logging.INFO)
logger.propagate = False

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

# Attach the file handler once — guard against duplicate handlers on re-import.
if not logger.handlers:
    os.makedirs(settings.storage.logging_path, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(settings.storage.logging_path, "app.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def get_logger():
    return logger
