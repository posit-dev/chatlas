import json
from typing import TYPE_CHECKING, Any, AsyncGenerator, Iterable, Optional, Sequence

from . import _utils
from ._abc import BaseChatWithTools
from ._merge import merge_dicts
from ._utils import ToolFunction

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types.chat import (
        ChatCompletionAssistantMessageParam,
        ChatCompletionMessageParam,
        ChatCompletionMessageToolCallParam,
        ChatCompletionToolMessageParam,
        ChatCompletionToolParam,
    )
    from openai.types.chat_model import ChatModel


class OpenAIChat(BaseChatWithTools["ChatCompletionMessageParam"]):
    _messages: list["ChatCompletionMessageParam"] = []
    _tool_schemas: list["ChatCompletionToolParam"] = []
    _tool_functions: dict[str, _utils.ToolFunctionAsync] = {}

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: "ChatModel" = "gpt-4o",
        system_prompt: Optional[str] = None,
        tools: Iterable[ToolFunction] = (),
        client: "AsyncOpenAI | None" = None,
    ):
        """
        Start a chat powered by OpenAI

        Parameters
        ----------
        api_key
            Your OpenAI API key.
        model
            The model to use for the chat.
        system_prompt
            A system prompt to use for the chat.
        tools
            A list of tools (i.e., function calls) to use for the chat.
        client
            An `openai.AsyncOpenAI` client instance to use for the chat. Use
            this to customize stuff like `base_url`, `timeout`, etc.

        Raises
        ------
        ImportError
            If the `openai` package is not installed.
        """
        self._model = model
        self._system_prompt = system_prompt
        for tool in tools:
            self.register_tool(tool)
        if client is None:
            client = self._get_client()
        if api_key is not None:
            client.api_key = api_key
        self.client = client

    def _get_client(self) -> "AsyncOpenAI":
        try:
            from openai import AsyncOpenAI

            return AsyncOpenAI()
        except ImportError:
            raise ImportError(
                f"The {self.__class__.__name__} class requires the `openai` package. "
                "Install it with `pip install openai`."
            )

    # TODO: make this an overloads (based on stream) and
    # suitable TypeDicts on kwargs
    async def response_generator(
        self,
        user_input: str,
        *,
        stream: bool = True,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Generate response(s) given a user input.

        Parameters
        ----------
        user_input
            The user input to the chat.
        stream
            Whether to stream the responses.
        kwargs
            Additional parameters to pass to the OpenAI's
            `chat.completions.create` method.
        """
        self._add_message({"role": "user", "content": user_input})
        while True:
            async for chunk in self._submit_messages(stream, **kwargs):
                yield chunk
            if not await self._invoke_tools():
                break

    async def _submit_messages(
        self,
        stream: bool,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        from openai.types.chat import ChatCompletionAssistantMessageParam

        model = kwargs.pop("model", self._model)
        tools: list["ChatCompletionToolParam"] = kwargs.pop("tools", [])
        tools.extend(self._tool_schemas)

        if stream:
            response = await self.client.chat.completions.create(
                messages=self.messages(include_system_prompt=True),
                model=model,
                tools=tools if tools else None,  # type: ignore
                stream=True,
                **kwargs,
            )
            result = None
            async for chunk in response:
                d = chunk.choices[0].delta
                if result is None:
                    result = d.model_dump()
                else:
                    result = merge_dicts(result, d.model_dump())
                if d.content:
                    yield d.content

            if result is not None:
                self._add_message(ChatCompletionAssistantMessageParam(**result))
        else:
            response = await self.client.chat.completions.create(
                messages=self.messages(include_system_prompt=True),
                model=model,
                tools=tools if tools else None,  # type: ignore
                stream=False,
                **kwargs,
            )
            message = response.choices[0].message
            msg = ChatCompletionAssistantMessageParam(**message.model_dump())
            self._add_message(msg)

            if message.content:
                yield message.content

    def messages(
        self, *, include_system_prompt: bool = False
    ) -> list["ChatCompletionMessageParam"]:
        """
        Get the messages in the chat.

        Parameters
        ----------
        include_system_prompt
            Whether to include the system prompt in the messages.

        Returns
        -------
        list[ChatCompletionMessageParam]
            The messages in the chat.
        """

        if include_system_prompt and self._system_prompt is not None:
            return [{"role": "system", "content": self._system_prompt}, *self._messages]
        return self._messages

    def _add_messages(self, messages: Sequence["ChatCompletionMessageParam"]):
        self._messages.extend(messages)

    def _add_message(self, message: "ChatCompletionMessageParam"):
        self._messages.append(message)

    def register_tool(
        self,
        func: ToolFunction,
        *,
        schema: Optional["ChatCompletionToolParam"] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameter_descriptions: Optional[dict[str, str]] = None,
        strict: bool = False,
    ):
        """
        Register a tool to use in the chat.

        Parameters
        ----------
        func
            The tool function (i.e., a Python function).
        schema
            The tool schema to use. If not provided, it will be auto-generated from
            the function.
        name
            The name of the tool. If not provided, it will be taken from the
            function.
        description
            The description of the tool. If not provided, it will be taken from
            the function's docstring.
        parameter_descriptions
            Descriptions for the parameters of the tool function.
        strict
            Whether to use strict mode for the tool.

        Raises
        ------
        ValueError
            If the tool name is already registered
        ValueError
            If the tool schema is invalid
        ValueError
            If the result of calling the tool results in an error
        """
        if schema is None:
            final_schema = self._transform_tool_schema(
                _utils.func_to_schema(func, name, description, parameter_descriptions),
                strict=strict,
            )
        else:
            final_schema = schema

        name = final_schema["function"]["name"]

        self._tool_schemas = [
            x for x in self._tool_schemas if x["function"]["name"] != name
        ]
        self._tool_schemas.append(final_schema)
        self._tool_functions[name] = _utils.wrap_async(func)

    @staticmethod
    def _transform_tool_schema(
        tool: "_utils.ToolSchema", strict: bool = False
    ) -> "ChatCompletionToolParam":
        fn = tool["function"]
        name = fn["name"]
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": fn["description"],
                "parameters": {
                    "type": "object",
                    "properties": fn["parameters"]["properties"],
                    "required": fn["parameters"]["required"],
                },
                "strict": strict,
            },
        }

    async def _invoke_tools(self) -> bool:
        if self._tool_functions:
            last = self.messages()[-1]
            assert last["role"] == "assistant"
            tool_messages = await self._call_tools(last)
            if len(tool_messages) > 0:
                self._add_messages(tool_messages)
                return True
        return False

    async def _call_tools(
        self, last_message: "ChatCompletionAssistantMessageParam"
    ) -> Sequence["ChatCompletionToolMessageParam"]:
        tool_calls = last_message.get("tool_calls", None)
        if tool_calls is None:
            return []
        res: list["ChatCompletionToolMessageParam"] = []
        for x in tool_calls:
            msg = await self._call_tool(x)
            res.append(msg)
        return res

    async def _call_tool(
        self,
        tool_call: "ChatCompletionMessageToolCallParam",
    ) -> "ChatCompletionToolMessageParam":
        name = tool_call["function"]["name"]
        tool_fun = self._tool_functions.get(name, None)
        if tool_fun is None:
            raise ValueError(f"Tool {name} not found.")

        args_str = tool_call["function"]["arguments"]
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON arguments for tool {name}")

        try:
            result = await tool_fun(**args)
        except Exception as e:
            raise ValueError(f"Error calling tool {name}: {e}")

        return {
            "role": "tool",
            "content": json.dumps({name: result, **args}),
            "tool_call_id": tool_call["id"],
        }
