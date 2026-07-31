from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

import orjson
from pydantic import BaseModel, Field, ValidationError
from pydantic_core import from_json

from ._content import (
    Content,
    ContentCitation,
    ContentJson,
    ContentText,
    ContentToolResult,
    DocumentSource,
    SearchResult,
    ToolSearchResults,
)
from ._logging import logger

if TYPE_CHECKING:
    from ._chat import Chat
    from ._turn import AssistantTurn

RETRIEVAL_TOOL_DESCRIPTION = """\
Search the registered document store for passages relevant to a query.
Use this whenever the user's question could be answered from the store's
documents. Ground your answer in the returned results."""


class CitedSegment(BaseModel):
    text: str = Field(
        description=(
            "A span of the answer, in plain prose. Concatenating every "
            "segment's text in order must produce the complete answer."
        )
    )
    chunk_ids: list[str] = Field(
        description=(
            "chunk_id values of the search results that directly support this "
            "span. Empty list if none. Start a new segment whenever the set "
            "of supporting sources changes."
        )
    )


class SegmentedAnswer(BaseModel):
    segments: list[CitedSegment]


class SegmentsDecoder:
    """Incrementally turn a streamed SegmentedAnswer JSON into contents.

    Citations for the most-recently-parsed segment are always withheld until
    either a later segment appears or `finish()` is called: under
    `allow_partial="trailing-strings"`, a trailing `chunk_ids` entry may
    itself be a truncated-but-valid JSON string (e.g. `"c12"` parsed
    mid-stream as `"c1"` — a wrong-but-valid id).
    """

    def __init__(self, chunks: Mapping[str, SearchResult]):
        self._chunks = chunks
        self._raw = ""
        self._seg_index = 0
        self._chars_emitted = 0

    def feed(self, delta: str) -> list[Content]:
        self._raw += delta
        try:
            data = from_json(self._raw, allow_partial="trailing-strings")
        except ValueError:
            return []
        segments = data.get("segments") if isinstance(data, dict) else None
        if not isinstance(segments, list):
            return []
        return self._advance(segments, final=False)

    def finish(self) -> list[Content]:
        try:
            data = from_json(self._raw, allow_partial="trailing-strings")
        except ValueError:
            # Nothing ever parsed: surface raw text rather than dropping it
            return [ContentText(text=self._raw)] if self._raw.strip() else []
        segments = data.get("segments") if isinstance(data, dict) else None
        if not isinstance(segments, list):
            return [ContentText(text=self._raw)] if self._raw.strip() else []
        return self._advance(segments, final=True)

    def _advance(self, segments: list[Any], *, final: bool) -> list[Content]:
        out: list[Content] = []
        while self._seg_index < len(segments):
            seg = segments[self._seg_index]
            text = seg.get("text", "") if isinstance(seg, dict) else ""
            is_last = self._seg_index == len(segments) - 1
            growth = text[self._chars_emitted :]
            if growth:
                out.append(ContentText.model_construct(text=growth))
                self._chars_emitted = len(text)
            if is_last and not final:
                break  # citations withheld: trailing chunk_ids may be truncated
            out.extend(segment_citations(seg, text, self._chunks))
            self._seg_index += 1
            self._chars_emitted = 0
        return out


def decode_segments_json(raw: str, chunks: Mapping[str, SearchResult]) -> list[Content]:
    """One-shot, strict decode of a complete SegmentedAnswer JSON string.

    Used for non-streaming responses and final-turn transformation. Falls
    back to treating the whole string as plain text if validation fails, so
    malformed JSON never crashes a turn.
    """
    try:
        answer = SegmentedAnswer.model_validate_json(raw)
    except ValidationError:
        return [ContentText(text=raw)]
    out: list[Content] = []
    for seg in answer.segments:
        if seg.text:
            out.append(ContentText.model_construct(text=seg.text))
        out.extend(segment_citations(seg.model_dump(), seg.text, chunks))
    return out


def segment_citations(
    seg: Any, grounded_span: str, chunks: Mapping[str, SearchResult]
) -> list[ContentCitation]:
    ids = seg.get("chunk_ids", []) if isinstance(seg, dict) else []
    out: list[ContentCitation] = []
    for chunk_id in ids:
        sr = chunks.get(chunk_id)
        if sr is None:
            logger.debug("Dropping citation with unknown chunk_id %r", chunk_id)
            continue
        out.append(
            ContentCitation(
                source=DocumentSource(id=sr.source or sr.id, title=sr.title),
                grounded_span=grounded_span,
                cited_quote=sr.text,
                extra={"chunk_id": sr.id},
            )
        )
    return out


@dataclass
class RegisteredStore:
    store: "RetrievalStore"
    top_k: int
    name: str
    description: Optional[str]


class RagManager:
    """Configure retrieval-augmented, citation-bearing chats. Via `chat.rag`."""

    def __init__(self, chat: "Chat"):
        self._chat = chat
        self._stores: dict[str, RegisteredStore] = {}
        # TODO: Add an eviction policy for chunks retained by long-lived chats.
        self._chunks: dict[str, SearchResult] = {}
        self._counter = 0

    def register_store(
        self,
        store: "RetrievalStore",
        *,
        top_k: int = 5,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        provider = self._chat.provider
        if (
            not provider.supports_native_search_results()
            and not provider.supports_tools_with_data_model()
        ):
            raise ValueError(
                f"Provider '{provider.name}' cannot combine tools with a "
                "response schema, which RAG citations require."
            )
        name = name or "search_documents"
        if name in self._stores:
            raise ValueError(
                f"A store named {name!r} is already registered. Pass a "
                "distinct `name=` to register another store."
            )
        reg = RegisteredStore(
            store=store,
            top_k=top_k,
            name=name,
            description=description,
        )
        self._chat.register_tool(self._make_retrieval_tool(reg), name=name, force=False)
        self._stores[name] = reg

    def unregister_store(self, name: str) -> None:
        reg = self._stores.pop(name)
        tools = self._chat.get_tools()
        self._chat.set_tools([t for t in tools if t.name != reg.name])

    @property
    def stores(self) -> dict[str, RegisteredStore]:
        return dict(self._stores)

    @property
    def chunks(self) -> dict[str, SearchResult]:
        return dict(self._chunks)

    def uses_segments_schema(self) -> bool:
        if not self._stores:
            return False
        return not self._chat.provider.supports_native_search_results()

    def transform_turn(self, turn: "AssistantTurn") -> "AssistantTurn":
        """Splice decoded prose/citations in for the raw segments-JSON content.

        Called on the final turn of a hand-rolled-tier response, whose JSON
        output arrives as a `ContentJson` (or `ContentText`, before a provider
        tags it) carrying the `SegmentedAnswer` payload. Every other content
        (tool requests, thinking, etc.) is left untouched.
        """
        new_contents: list[Content] = []
        for content in turn.contents:
            if isinstance(content, ContentJson):
                raw = orjson.dumps(content.value).decode()
                new_contents.extend(decode_segments_json(raw, self._chunks))
            elif isinstance(content, ContentText):
                new_contents.extend(decode_segments_json(content.text, self._chunks))
            else:
                new_contents.append(content)
        # Content is the base class; contents is typed as list[ContentUnion]
        # (discriminated union). At runtime all Content subclasses are ContentUnion
        # members, so the assignment is safe (same reasoning as TurnAccumulator).
        turn.contents = new_contents  # type: ignore[assignment]
        return turn

    def register_chunks(self, chunks: Sequence["ChunkLike"]) -> list[SearchResult]:
        out: list[SearchResult] = []
        for chunk in chunks:
            self._counter += 1
            sr = normalize_chunk(chunk, id=f"c{self._counter}")
            self._chunks[sr.id] = sr
            out.append(sr)
        return out

    def _make_retrieval_tool(self, reg: RegisteredStore):
        def retrieve(query: str) -> ContentToolResult:
            """Search the document store.

            Parameters
            ----------
            query
                What to look for, phrased as a focused search query.
            """
            chunks = reg.store.retrieve(query, reg.top_k)
            results = self.register_chunks(chunks)
            return ContentToolResult(value=ToolSearchResults(results=results))

        retrieve.__name__ = reg.name
        inner_doc = inspect.cleandoc(retrieve.__doc__ or "")
        retrieve.__doc__ = f"{reg.description or RETRIEVAL_TOOL_DESCRIPTION}\n\n{inner_doc}"
        return retrieve


@runtime_checkable
class ChunkLike(Protocol):
    """A retrieved chunk: `text` is required; `origin` (stable source string),
    `context` (human label, e.g. heading trail), and `attributes` (dict) are
    read via `getattr` when present. raghilda's `Chunk` satisfies this."""

    text: str


@runtime_checkable
class RetrievalStore(Protocol):
    """Anything with `retrieve(text, top_k) -> chunks`. raghilda's
    `BaseStore` satisfies this; so does any user class with one method."""

    def retrieve(self, text: str, top_k: int) -> Sequence[ChunkLike]: ...


def normalize_chunk(chunk: ChunkLike, id: str) -> SearchResult:  # noqa: A002
    origin: Optional[str] = getattr(chunk, "origin", None)
    context: Optional[str] = getattr(chunk, "context", None)
    attributes = getattr(chunk, "attributes", None)
    return SearchResult(
        id=id,
        text=chunk.text,
        source=origin,
        title=context,
        extra=dict(attributes) if attributes else {},
    )
