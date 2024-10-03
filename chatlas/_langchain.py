import json
from typing import TYPE_CHECKING, Any, AsyncGenerator, Iterable, Optional, Sequence

from . import _utils
from ._abc import BaseChat
from ._utils import ToolFunction

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage, ToolCall, ToolMessage
    from langchain_core.runnables import Runnable


class LangChainChat(BaseChat["BaseMessage"]):
    _messages: list["BaseMessage"] = []
    _tool_functions: dict[str, _utils.ToolFunctionAsync] = {}

    def __init__(
        self,
        model: "BaseChatModel | Runnable",
        *,
        system_prompt: Optional[str] = None,
        tools: Iterable[ToolFunction] = (),
    ) -> None:
        """
        Start a chat powered by LangChain.

        Parameters
        ----------
        model
            The chat model or runnable to use for the chat.
        system_prompt
            A system prompt to use for the chat.
        tools
            A list of tools (i.e., function calls) to use for the chat.
        """

        try:
            import langchain_core  # noqa: F401
        except ImportError:
            raise ImportError(
                f"The {self.__class__.__name__} class requires the `langchain-core` package. "
                "Please install it with `pip install langchain-core`."
            )

        if tools:
            from langchain_core.language_models.chat_models import BaseChatModel
            from langchain_core.tools import tool

            if not isinstance(model, BaseChatModel):
                raise ValueError(
                    "When `tools` are provided to the `LangChainChat()` constructor, the `chat_model` "
                    "must be a `ChatModel`, not a `Runnable`. Instead of passing `tools` to `LangChainChat()` "
                    "consider registering them with the `ChatModel` via it's `.bind_tools()` method."
                )

            model = model.bind_tools([tool(func) for func in tools])
            for func in tools:
                name = func.__name__
                self._tool_functions[name] = _utils.wrap_async(func)

        self._system_prompt = system_prompt
        self.model = model

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
            The user input to generate responses for.
        stream
            Whether to stream the responses or not.
        kwargs
            Additional keyword arguments passed to the chat model's
            `ainvoke()`/`astream()` method.
        """

        from langchain_core.messages import HumanMessage

        self._add_message(HumanMessage(content=user_input))
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
        if stream:
            from langchain_core.messages import AIMessage, AIMessageChunk

            response = self.model.astream(
                self.messages(include_system_prompt=True),
                **kwargs,
            )

            result = None
            async for chunk in response:
                if not isinstance(chunk, AIMessageChunk):
                    raise TypeError(
                        "Expected each iteration to be of class 'AIMessageChunk', "
                        f"but got {type(chunk)} instead"
                    )

                if result is None:
                    result = chunk
                else:
                    result = result + chunk
                async for content in self._yield_strings(chunk.content):
                    yield content

            if result is not None:
                msg = result.model_dump()
                msg["type"] = "ai"
                self._add_message(AIMessage(**msg))

        else:
            from langchain_core.messages import BaseMessage

            result = await self.model.ainvoke(
                self.messages(include_system_prompt=True),
                **kwargs,
            )
            if not isinstance(result, BaseMessage):
                raise TypeError(
                    "Expected result to be of class 'BaseMessage', "
                    f"but got {type(result)} instead"
                )
            self._add_message(result)

            async for content in self._yield_strings(result.content):
                yield content

    def messages(self, *, include_system_prompt: bool = False) -> list["BaseMessage"]:
        """
        Get the messages in the chat.

        Parameters
        ----------
        include_system_prompt
            Whether to include the system prompt in the messages or not.

        Returns
        -------
        list[BaseMessage]
            The messages in the chat.
        """

        if not include_system_prompt:
            return self._messages
        if not self._system_prompt:
            return self._messages

        from langchain_core.messages import SystemMessage
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(self._system_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        return prompt.format_messages(messages=self.messages())

    def _add_messages(self, messages: Sequence["BaseMessage"]) -> None:
        self._messages.extend(messages)

    def _add_message(self, message: "BaseMessage") -> None:
        self._messages.append(message)

    async def _invoke_tools(self) -> bool:
        if self._tool_functions:
            from langchain_core.messages import AIMessage

            last = self.messages()[-1]
            assert isinstance(last, AIMessage)
            tool_messages = await self._call_tools(last)
            if len(tool_messages) > 0:
                self._add_messages(tool_messages)
                return True
        return False

    async def _call_tools(self, last_message: "AIMessage") -> list["ToolMessage"]:
        tool_calls = last_message.tool_calls
        res: list["ToolMessage"] = []
        for x in tool_calls:
            msg = await self._call_tool(x)
            res.append(msg)
        return res

    async def _call_tool(self, tool_call: "ToolCall") -> "ToolMessage":
        from langchain_core.messages import ToolMessage

        name = tool_call["name"]
        tool_fun = self._tool_functions.get(name, None)
        if tool_fun is None:
            raise ValueError(f"Tool {name} not found.")

        args = tool_call["args"]
        if not isinstance(args, dict):
            raise ValueError(
                f"Expected args for tool {name} to be a dictionary."
                f"Got {type(args)} instead."
            )

        try:
            result = await tool_fun(**args)
        except Exception as e:
            raise ValueError(f"Error calling tool {name}: {e}")

        return ToolMessage(
            tool_call_id=tool_call["id"],
            content=json.dumps({name: result, **args}),
            type="tool",
        )

    # The type for content here comes from BaseMessage.content
    async def _yield_strings(
        self, content: str | list[str | dict[str, Any]]
    ) -> AsyncGenerator[str, None]:
        if isinstance(content, str):
            yield content
        else:
            for y in content:
                if isinstance(y, str):
                    yield y
                elif isinstance(y, dict):
                    if "text" in y:  # langchain-anthropic
                        yield y["text"]
                    elif "content" in y:
                        yield y["content"]
