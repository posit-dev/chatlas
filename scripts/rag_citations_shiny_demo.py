from dataclasses import dataclass
from typing import Sequence

from chatlas import ChatOpenAI
from shiny import App, Inputs, Outputs, Session, ui
from shinychat import Chat, chat_ui


@dataclass
class Chunk:
    text: str
    origin: str
    context: str


class Store:
    def __init__(self, chunks: Sequence[Chunk]):
        self._chunks = chunks

    def retrieve(self, text: str, top_k: int) -> Sequence[Chunk]:
        return self._chunks[:top_k]


app_ui = ui.page_fillable(
    ui.panel_title("Citation-aware RAG"),
    chat_ui("chat"),
    fillable_mobile=True,
)


def server(input_: Inputs, output: Outputs, session: Session) -> None:
    chat = Chat("chat")
    chat_client = ChatOpenAI(
        system_prompt=(
            "Use the document search tool to answer questions about Flurbo. "
            "Ground every answer in the returned search results."
        )
    )
    chat_client.rag.register_store(
        Store(
            [
                Chunk(
                    text=(
                        "The fictional Flurbo framework streams responses through "
                        "flb.stream(), which yields FlurboChunk objects."
                    ),
                    origin="kb://flurbo/streaming",
                    context="Flurbo streaming guide",
                )
            ]
        )
    )

    @chat.on_user_submit
    async def handle_user_input(user_input: str) -> None:
        response = await chat_client.stream_async(user_input, content="all")
        await chat.append_message_stream(response)


app = App(app_ui, server)
