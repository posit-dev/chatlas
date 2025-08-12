import logging
import os
import warnings
from typing import Literal

from rich.logging import RichHandler


def _rich_handler() -> RichHandler:
    formatter = logging.Formatter("%(name)s - %(message)s")
    handler = RichHandler()
    handler.setFormatter(formatter)
    return handler


def setup_logger(x: str, level: Literal["debug", "info"]) -> logging.Logger:
    logger = logging.getLogger(x)
    if level == "debug":
        logger.setLevel(logging.DEBUG)
    elif level == "info":
        logger.setLevel(logging.INFO)
    # By adding a RichHandler to chatlas' logger, we can guarantee that they
    # never get dropped, even if the root logger's handlers are not
    # RichHandlers.
    if not any(isinstance(h, RichHandler) for h in logger.handlers):
        logger.addHandler(_rich_handler())
    logger.propagate = False
    return logger


logger = logging.getLogger("chatlas")
log_level = os.environ.get("CHATLAS_LOG")
if log_level:
    if log_level != "debug" and log_level != "info":
        warnings.warn(
            f"CHATLAS_LOG is set to '{log_level}', but the log level must "
            "be one of 'debug' or 'info'. Defaulting to 'info'.",
        )
        log_level = "info"

    # Manually setup the logger for each dependency we care about. This way, we
    # can ensure that the logs won't get dropped when a rich display is activate
    logger = setup_logger("chatlas", log_level)
    openai_logger = setup_logger("openai", log_level)
    anthropic_logger = setup_logger("anthropic", log_level)
    google_logger = setup_logger("google_genai.models", log_level)
    httpx_logger = setup_logger("httpx", log_level)

    # Add a RichHandler to the root logger if there are no handlers. Note that
    # if chatlas is imported before other libraries that set up logging, (like
    # openai, anthropic, or httpx), this will ensure that logs from those
    # libraries are also displayed in the rich console.
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(_rich_handler())

    # Warn if there are non-RichHandler handlers on the root logger.
    # TODO: we could consider something a bit more abusive here, like removing
    # non-RichHandler handlers from the root logger, but that could be
    # surprising to users.
    bad_handlers = [
        h.get_name() for h in root.handlers if not isinstance(h, RichHandler)
    ]
    if len(bad_handlers) > 0:
        warnings.warn(
            "When setting up logging handlers for CHATLAS_LOG, chatlas detected "
            f"non-rich handler(s) on the root logger named {bad_handlers}. "
            "As a result, logs handled those handlers may be dropped when the "
            "`echo` argument of `.chat()`, `.stream()`, etc., is something "
            "other than 'none'. This problem can likely be fixed by importing "
            "`chatlas` before other libraries that set up logging, or adding a "
            "RichHandler to the root logger before loading other libraries.",
        )


def log_model_default(model: str) -> str:
    logger.info(f"Defaulting to `model = '{model}'`.")
    return model


def log_tool_error(name: str, arguments: str, e: Exception):
    logger.info(
        f"Error invoking tool function '{name}' with arguments: {arguments}. "
        f"The error message is: '{e}'",
    )
