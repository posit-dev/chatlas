import logging
import os
import warnings

from rich.logging import RichHandler


def _rich_handler() -> RichHandler:
    formatter = logging.Formatter("%(name)s - %(message)s")
    handler = RichHandler()
    handler.setFormatter(formatter)
    return handler


logger = logging.getLogger("chatlas")

if os.environ.get("CHATLAS_LOG") == "info":
    # By adding a RichHandler to chatlas' logger, we can guarantee that they
    # never get dropped, even if the root logger's handlers are not
    # RichHandlers.
    logger.setLevel(logging.INFO)
    logger.addHandler(_rich_handler())
    logger.propagate = False

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
