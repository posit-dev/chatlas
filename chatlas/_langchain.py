# from typing import TYPE_CHECKING, AsyncGenerator, Iterable, Optional, Union

# from ._abc import LLMClientWithTools
# from ._utils import ToolFunction

# if TYPE_CHECKING:
#     from langchain_core.language_models import BaseChatModel
#     from langchain_core.messages import (
#         AIMessage,
#         HumanMessage,
#         SystemMessage,
#         ToolCall,
#         ToolMessage,
#     )
#     # from langchain_core.tools import tool

#     LangChainMessage = Union[AIMessage, HumanMessage, SystemMessage, ToolMessage]


# class LangChain(LLMClientWithTools["LangChainMessage"]):
#     _messages: list["LangChainMessage"] = []
#     _tool_schemas: list["ToolCall"] = []
#     _tool_functions: dict[str, ToolFunction] = {}

#     def __init__(
#         self,
#         client: "BaseChatModel",
#         model: Optional[str] = None,
#         tools: Iterable[ToolFunction] = (),
#     ) -> None:
#         self.client = client
#         self._model = model
#         for tool in tools:
#             self.register_tool(tool)

#     def register_tool(
#         self,
#         func: ToolFunction,
#         *,
#         name: Optional[str] = None,
#         schema: Optional[ToolCall] = None,
#     ) -> None:

#         from langchain_core.tools import tool

#         self.client.bind_tools([tool(func)])

#     async def generate_response(
#         self,
#         user_input: str,
#         *,
#         stream: bool = True,
#         **kwargs: Any,
#     ) -> AsyncGenerator[str, None]:
#         self._add_message({"role": "user", "content": user_input})
#         while True:
#             async for chunk in self._submit_messages(stream, **kwargs):
#                 yield chunk
#             if not self._invoke_tools():
#                 break
