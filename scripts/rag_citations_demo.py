import os
import sys
from dataclasses import dataclass
from typing import Sequence

from chatlas import ChatOpenAI
from chatlas.types import ContentCitation


@dataclass
class Chunk:
    text: str
    origin: str
    context: str


class Store:
    def __init__(self, chunks: Sequence[Chunk]):
        self._chunks = chunks

    def retrieve(self, query: str, top_k: int) -> Sequence[Chunk]:
        return self._chunks[:top_k]


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY before running this example.", file=sys.stderr)
        raise SystemExit(1)

    store = Store(
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
    chat = ChatOpenAI(
        system_prompt=(
            "Use the document search tool to answer questions about Flurbo. "
            "Ground every answer in the returned search results."
        )
    )
    chat.rag.register_store(store)
    chat.chat("How does Flurbo stream responses?", echo="all")

    turn = chat.get_last_turn(role="assistant")
    citations = (
        [content for content in turn.contents if isinstance(content, ContentCitation)]
        if turn is not None
        else []
    )
    if not citations:
        raise SystemExit("No citation was returned; the RAG check failed.")

    print(f"\nVerified citation: {citations[0].source}")


if __name__ == "__main__":
    main()
