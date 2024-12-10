import logging
import os

from rich.logging import RichHandler

logger = logging.getLogger("chatlas")
if len(logger.handlers) == 0:
    logger.addHandler(logging.NullHandler())

if os.getenv("CHATLAS_LOG", "").lower() == "info":
    formatter = logging.Formatter("%(name)s - %(message)s")
    handler = RichHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def log_model_default(model: str) -> str:
    logger.info(f"Defaulting to `model = '{model}'`.")
    return model


def log_tool_error(name: str, arguments: str, e: Exception):
    logger.info(
        f"Error invoking tool function '{name}' with arguments: {arguments}. "
        f"The error message is: '{e}'",
    )
