from typing import TYPE_CHECKING, AsyncGenerator, Optional

from ._abc import BaseChat
from ._utils import ToolFunction

if TYPE_CHECKING:
    from google.generativeai.types import (
        ContentType,
        FunctionLibraryType,
        GenerationConfigType,
        RequestOptionsType,
    )
    from google.generativeai.types.content_types import ToolConfigType, protos
    from google.generativeai.types.safety_types import SafetySettingOptions

    Content = protos.Content


class GoogleChat(BaseChat["Content"]):
    _messages: list["Content"] = []
    _tool_schemas: list["FunctionLibraryType"] = []
    _tool_functions: dict[str, ToolFunction] = {}

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gemini-pro",
        system_prompt: Optional["ContentType"] = None,
        tools: Optional["FunctionLibraryType"] = None,
        tool_config: Optional["ToolConfigType"] = None,
        safety_settings: Optional["SafetySettingOptions"] = None,
        generation_config: Optional["GenerationConfigType"] = None,
    ) -> None:
        try:
            from google.generativeai import GenerativeModel
        except ImportError:
            raise ImportError(
                f"The {self.__class__.__name__} class requires the `google-generativeai` package. "
                "Install it with `pip install google-generativeai`."
            )

        if api_key is not None:
            import google.generativeai as genai

            genai.configure(api_key=api_key)

        self.client = GenerativeModel(
            model_name=model,
            system_instruction=system_prompt,
            tools=tools,
            tool_config=tool_config,
            safety_settings=safety_settings,
            generation_config=generation_config,
        )

        # https://github.com/google-gemini/cookbook/blob/main/quickstarts/Function_calling.ipynb
        self.__chat = self.client.start_chat(
            enable_automatic_function_calling=tools is not None
        )

    async def response_generator(
        self,
        user_input: str,
        *,
        stream: Optional[bool] = None,
        request_options: "Optional[RequestOptionsType]" = None,
    ) -> AsyncGenerator[str, None]:
        # Google doesn't currently support streaming + tool calls
        if stream is None:
            if self.__chat.enable_automatic_function_calling:
                stream = False
            else:
                stream = True

        if stream:
            response = self.__chat.send_message(
                user_input, stream=True, request_options=request_options
            )

            for chunk in response:
                yield chunk.text
        else:
            response = self.__chat.send_message(
                user_input, stream=False, request_options=request_options
            )
            yield response.text

    def messages(self) -> list["protos.Content"]:
        return self.__chat.history
