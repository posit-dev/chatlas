from typing import TYPE_CHECKING, Iterable, Optional

from ._abc import ChatWithTools
from ._utils import ToolFunction

if TYPE_CHECKING:
    import google.generativeai.types as gtypes  # pyright: ignore[reportMissingTypeStubs]
    from google.generativeai import (
        GenerativeModel,  # pyright: ignore[reportMissingTypeStubs]
    )

    ContentDict = gtypes.ContentDict
    FunctionLibraryType = gtypes.FunctionLibraryType


class GoogleChat(ChatWithTools["ContentDict"]):
    _messages: list["ContentDict"] = []
    _tool_schemas: list["FunctionLibraryType"] = []
    _tool_functions: dict[str, ToolFunction] = {}

    def __init__(
        self,
        client: "GenerativeModel | None" = None,
        model: Optional[str] = None,
        tools: Iterable[ToolFunction] = (),
    ) -> None:
        self._model = model
        for tool in tools:
            self.register_tool(tool)
        if client is None:
            client = self._get_client()
        self.client = client

    def _get_client(self) -> "GenerativeModel":
        try:
            from google.generativeai import GenerativeModel

            return GenerativeModel()
        except ImportError:
            raise ImportError(
                f"The {self.__class__.__name__} class requires the `ollama` package. "
                "Install it with `pip install ollama`."
            )
