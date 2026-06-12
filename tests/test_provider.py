from typing import Any, Optional

from chatlas._provider import Provider, StandardModelParams, StandardModelParamNames
from chatlas._content import Content
from chatlas._turn import AssistantTurn, Turn
from chatlas._tools import Tool, ToolBuiltIn
from pydantic import BaseModel


class _FakeProvider(Provider):
    """Minimal stub that implements every abstract method so we can test the concrete ones."""

    def list_models(self):
        return []

    def chat_perform(self, *, stream, turns, tools, data_model, kwargs):  # type: ignore[override]
        return None  # type: ignore[return-value]

    async def chat_perform_async(self, *, stream, turns, tools, data_model, kwargs):  # type: ignore[override]
        return None  # type: ignore[return-value]

    def stream_content(self, chunk):
        return chunk  # treat the passed Content as the "chunk"

    def stream_merge_chunks(self, completion, chunk):
        return None

    def stream_turn(self, completion, has_data_model):
        return AssistantTurn([])  # type: ignore[return-value]

    def value_turn(self, completion, has_data_model):
        return AssistantTurn([])  # type: ignore[return-value]

    def value_tokens(self, completion):
        return None

    def token_count(self, *args, tools, data_model):
        return 0

    async def token_count_async(self, *args, tools, data_model):
        return 0

    def translate_model_params(self, params: StandardModelParams):
        return {}  # type: ignore[return-value]

    def supported_model_params(self) -> set[StandardModelParamNames]:
        return set()
