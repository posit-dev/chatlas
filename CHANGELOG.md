# Changelog

<!--
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
-->


## [0.23.0] - 2026-09-04

### New features

* `Chat` gains `close()` and `close_async()` methods (plus context-manager support) for releasing resources held by the provider -- HTTP connection pools, the Snowflake Snowpark session/connection, and (via `close_async()`) MCP server sessions. This is useful in long-lived applications like Shiny that create a chat per user session: `session.on_ended(chat.close)`. Providers only close resources they created themselves; caller-supplied clients are left open.

* `ChatSnowflake()` gains a `session` parameter for supplying an existing `snowflake.snowpark.Session`, mirroring `ChatDatabricks()`'s `workspace_client`. This lets one session be shared across multiple chats; `Chat.close()` only closes sessions that chatlas created itself, leaving caller-supplied sessions open.

* When running on Posit Connect, chatlas now forwards the Shiny viewer's session token to Connect's LLM gateway (as a `Posit-Connect-User-Session-Token` header) so gateway usage can be attributed to the viewer. This happens automatically for Shiny content and only affects requests to the gateway.

### Changes

- `ChatHuggingFace()`'s default `model` is now `Qwen/Qwen3-235B-A22B-Instruct-2507` (previously `meta-llama/Llama-3.1-8B-Instruct`), matching ellmer's default. (#414)
- `ChatBedrock()` now defaults `base_url` to the official AWS SDKs' endpoint override environment variables when set: `AWS_ENDPOINT_URL_BEDROCK_RUNTIME` for `api="converse"`, and `AWS_ENDPOINT_URL_BEDROCK_MANTLE` for `api="messages"` and `api="responses"`. Similarly, `ChatAnthropic()` respects the `ANTHROPIC_BASE_URL` environment variable (via the anthropic SDK). Setting these variables is enough to route requests through a proxy or gateway, so you don't have to pass `base_url` on every call.

### Bug fixes

* `ChatDatabricks()` no longer drops the assistant's reply from the conversation when a GPT-OSS endpoint streams typed content. The typed part array was merged into the accumulated completion before it was normalized, so every later text delta was appended to it one character at a time and the finished turn came back empty. (#409)
* `.to_solver()` no longer corrupts the system prompt or the prior turns it reads out of Inspect AI's message state. The system prompt was being set to the `repr()` of the `ChatMessageSystem` object rather than its text, and message content arriving in Inspect AI's `str` form (rather than as a list of `Content`) was iterated one character at a time. (#407)
* `ChatGoogle()` no longer raises `ValueError: Unknown content type: ContentThinking` on the second and later turns when `reasoning` is enabled; thinking content is now replayed to the model as thought parts, and the `thought_signature` on thought parts is preserved (previously only tool-call parts kept it). (#403)
* `ChatOllama()` now distinguishes a remote endpoint it can't reach from a genuinely missing local install, and validates a supplied `model` against `/api/tags` at construction time instead of only when `model` is omitted. (#393)

## [0.22.0] - 2026-08-25

### Changes

- `ChatAnthropic()`, `ChatBedrock()`, and `ChatPosit()` now require `anthropic>=1.0.0`. As a result, custom `http_client`s passed to Anthropic-backed providers must now be `httpx2` clients (rather than `httpx`), matching the anthropic SDK's own requirement.
- `ChatBedrock()` and `ChatBedrockAnthropic()` now raise at construction if no AWS region can be resolved (from the `aws_region` argument, the `AWS_REGION`/`AWS_DEFAULT_REGION` environment variables, or the AWS profile), rather than silently defaulting to `us-east-1`. This behavior change is inherited from anthropic 1.0.
- On Anthropic-backed providers, the `temperature`, `top_p`, and `top_k` model parameters are deprecated: anthropic 1.0 removed them from the request schema since current models ignore them. chatlas now forwards them via `extra_body` (with a `DeprecationWarning`) so older models that still honor them keep working; this forwarding will be removed in a future release.

### Added

- `Chat` gains a settable `.conversation_id` property. When set, the identifier is recorded as the `gen_ai.conversation.id` attribute on the OpenTelemetry `chat` and `invoke_agent` spans, allowing backends to group spans belonging to the same conversation (per the OpenTelemetry GenAI semantic conventions). Developer-facing: intended for frameworks that manage conversation history; chatlas never generates an identifier on its own.

### Bug fixes

* `ChatDatabricks()` no longer crashes on GPT-OSS endpoints that return `message.content` / `delta.content` as a list of typed parts instead of a plain string; text parts are concatenated back into a string and reasoning summaries become `ContentThinking`/`ContentThinkingDelta`, both streaming and non-streaming. (#392)

## [0.21.2] - 2026-08-20

### Bug fixes

* Rich tool results containing images or PDFs remain semantic `ContentToolResult` objects in saved chat history, so restoring a conversation no longer exposes provider-only XML and media as user-authored content.
* `ChatOpenAI()` web-search citations no longer report the inline Markdown source link as `ContentCitation.grounded_span`; OpenAI's citation offsets identify that marker rather than the answer text it supports. (#388)
* `ChatOpenAI()` no longer crashes while streaming web-search citations with current OpenAI SDKs, which emit annotations as model objects rather than dictionaries.
* `ChatAnthropic()` now identifies Claude 5 and later models as supporting native structured output, so automatic mode does not fall back to tools. (#390)

## [0.21.1] - 2026-08-12

### Bug fixes

* `ChatGoogle()` and `ChatVertex()` now emit grounded citations immediately after
  the answer text they support while streaming, before the related web search or
  URL-fetch activity. This keeps streamed citations attached to their answer
  text instead of separating them into a later activity segment.
* OpenAI-based providers now support the OpenAI 3 SDK and its native `httpx2`
  clients. Custom clients are typed and documented as `httpx2`; legacy `httpx`
  clients remain supported at runtime during migration. (#387)
* `ChatAnthropic()` citations backed by `document_index` (from `tool_web_fetch()` results and document/PDF attachments) now resolve to a source URL, instead of always coming back without one. Anthropic counts that index across every document-shaped block in the whole request, including ones from prior turns, which chatlas wasn't accounting for. (#382)
* Token cost lookups (`.get_cost()`, `token_usage()`) no longer crash on models with no input price (e.g. output-only video generation models on Bedrock), and read ellmer's current pricing data format, which now wraps the price list in a versioned envelope rather than a bare array. (#382)
* `Chat.to_solver()` now reports `total_tokens` including cached input tokens, matching what Inspect's own `generate()` path records. Previously the total was just input plus output, so under a prompt-cache hit an eval run through the solver under-reported its total by exactly the number of cached tokens. (#216)

### Breaking changes

* Custom `Provider` implementations must accept a `turns` keyword argument on `.stream_content()`, `.stream_turn()`, and `.value_turn()`. chatlas passes the complete request history to these hooks so providers can resolve response references to content from earlier turns. (#382)

## [0.21.0] - 2026-08-04

### New features

* New `ChatBedrock()` gives full access to AWS Bedrock's model catalog — Nova, Llama, Mistral, DeepSeek, Qwen, plus the GPT-5 family, Grok 4.3, and Gemma 4 — not just Claude, none of which were previously available through chatlas. It replaces `ChatBedrockAnthropic()` as the recommended entrypoint; the right request format (`"converse"`, `"responses"`, or `"messages"`) is picked automatically from the model name, or set `api` explicitly.

### Improvements

* Updated default models to match the latest generation:
  * Anthropic / BedrockAnthropic / Posit: `claude-sonnet-5`
  * OpenAI / Completions / OpenRouter: `gpt-5.6-terra`
* Echoing turns in the console and notebooks got a round of display improvements:
  * Reasoning/thinking content now actually shows up — it used to silently disappear, since it was wrapped in literal `<thinking>` tags that a markdown renderer treated as an HTML block and dropped. It renders in a collapsible "Thinking" panel (a `<details>` block in notebooks) that stays open while streaming and collapses once done, and is capped to the most recent lines when long. (#361)
  * Long tool results no longer flood the screen — they collapse/truncate with a clear count of what's hidden, scrolling internally in notebooks beyond a bounded height. (#361)
  * Images from models or tools now render as compact thumbnails instead of raw base64 data.
  * Web search, fetch, and citation activity is now visible too, grouped into a "Searched the web" / "Read the web" panel that marks which sources were actually cited. (#256)
  * All of these size limits are tunable via `Chat.set_echo_options()` (`tool_result_max_lines`, `tool_result_max_height`, `thinking_max_lines`, `image_max_lines`, `web_activity_max_sources`), and can be turned off entirely with `None`.
* Registering a built-in tool (`tool_web_search()`, `tool_web_fetch()`) with a provider that can't run it now fails immediately with a clear error naming the tool and provider, instead of silently no-op'ing or dying deep inside a later request. (#367)

### Bug fixes

* `echo="all"` no longer displays tool results twice — once in full as part of the user turn, and again on their own.
* `Chat.set_echo_options(css_styles=)` now actually applies in notebooks.
* Tool names and argument names are now HTML-escaped in notebook/shiny rendering, closing an HTML-injection hole.
* A tool that reports progress by yielding more than once, or an MCP server that answers a call with several content parts (text plus an image, say), no longer breaks the request.
* `register_mcp_tools_stdio_async()` and `register_mcp_tools_http_stream_async()` no longer fail when an MCP server leaves some tool annotations unset.
* `ContentCitation`, `ContentToolRequestFetch`, and `ContentToolResponseFetch` now actually render, instead of silently vanishing due to a markdown link-reference parsing quirk.
* `ChatAnthropic()` (and `ChatBedrockAnthropic()`) now bill refusal-fallback turns at the correct (serving model's) rate rather than the originally requested model's, mirroring [ellmer's equivalent fix](https://github.com/tidyverse/ellmer/pull/1058).

### Breaking changes

* `ChatGithub()` is now defunct: it always raises `RuntimeError`. GitHub Models was retired on 2026-07-30, so the underlying API no longer works. Use `ChatGoogle()` (offers a free tier) or `ChatPosit()` (offers a free trial) instead.

## [0.20.0] - 2026-07-29

### New features

* New `content_document_file()` and `content_document_url()` prepare plain text, Markdown, CSV, code, and (on `ChatOpenAI()`) binary office files like `.docx`/`.xlsx`/`.doc`/`.xls`/`.rtf`/`.odt` for chat input, returning a new `ContentDocument` type. Previously the only way to attach a non-PDF file was to read and string-interpolate it yourself, which lost the filename and OpenAI's spreadsheet parsing. Provider support varies:
  * `ChatOpenAI()` (Responses API) accepts every type above.
  * `ChatGoogle()` accepts text-ish types; the binary office formats raise.
  * `ChatAnthropic()` accepts documents it can treat as plain text; binary formats raise.
  * `ChatOpenAICompletions()` sends the document as-is; OpenAI's own Chat Completions endpoint only accepts `application/pdf` and rejects the rest, though other OpenAI-compatible backends may accept more. Use `ChatOpenAI()` against OpenAI proper.

  Like `content_pdf_url()`, `content_document_url()` doesn't download up front: `ChatOpenAI()` references the URL directly, and other providers fetch lazily.
* `ImageContentTypes` (and so `content_image_file()`/`content_image_url()`) now accepts `image/heic` and `image/heif`, which `ChatGoogle()` supports natively. Other providers raise a clear error rather than sending a format they'll reject. Resizing HEIC/HEIF images requires the optional `pillow-heif` package; without it, pass `resize="none"`.
* `content_pdf_url()` no longer downloads the PDF's bytes up front. Anthropic and `ChatOpenAI()` (the Responses API) can reference the URL directly, so the bytes are only downloaded -- and cached -- if the target provider actually needs them (`ChatGoogle()`, `ChatOpenAICompletions()`). This reduces bandwidth and request payload size, which matters given Anthropic's 32 MB and OpenAI's 50 MB request limits. Accordingly, `ContentPDF.data` is now `Optional[bytes]` (it's `None` when only a `url` is set); a validator requires at least one of `data`/`url`. `content_pdf_url()` also now takes the `filename` from the URL's last path segment when it ends in `.pdf` (e.g. `apples.pdf`), rather than always generating `file_001.pdf`; URLs without a usable name still fall back to the generated one.
* `Chat` gains a `.files` accessor for uploading files to a provider once and referencing them across turns without re-sending bytes, plus listing, fetching metadata, downloading, and deleting them. Supported for OpenAI, Anthropic, and Google Gemini. A new `ContentUploaded` type represents the reference and can be constructed directly to point at a file uploaded out-of-band (e.g. a Vertex `gs://` URI). For Google, `upload()` waits for Gemini to finish processing large media (video, audio) before returning, since the API rejects references to files that aren't yet `ACTIVE`.
* Web search and fetch results now surface their citations across all three providers (OpenAI, Anthropic, Google), both progressively during streaming and on the final turn. `ContentCitation` nests a typed `source` (a `Source` subclass — `WebSource` today, carrying `url`/`title`) instead of flat `url`/`title` fields, and carries `grounded_span` (the answer-side span it grounds) plus `cited_quote` (the source-side quote, populated for `ChatAnthropic()` web search). `source` is optional — a citation can ground answer text with no resolvable link. `ContentCitation`, `Source`, and `WebSource` are exported from `chatlas.types`. A future file/document/RAG source becomes another `Source` subclass without breaking `ContentCitation.source`; note that `ContentToolResponseSearch.sources` is typed narrowly as `list[WebSource]` and would need widening at that point.
  * When streaming with `content="all"`, `ContentCitation` objects are emitted as citations arrive — interleaved with text for OpenAI and Anthropic, at stream-end for Google. Its position in the stream (relative to surrounding text) is the placement signal for rendering footnote markers.
  * On the final turn, `ContentCitation` items appear in the turn's `contents` list after the `ContentText` they ground, in the order the provider reported them. Since a turn's text arrives as one accumulated `ContentText`, position no longer narrows a citation to a span within it — use `grounded_span` for that.
* `batch_chat()` now supports `ChatGoogle()` (Gemini Developer API batch jobs). Batch is also now documented as supported for `ChatGroq()`, which already worked via its OpenAI-compatible provider. (Vertex AI is not supported, since its batch API requires GCS bucket URIs instead of inline requests.)
* `ChatOllama()` gains a `reasoning_effort` parameter to enable extended "thinking" for models that support it (e.g. qwen3, gpt-oss).
* `Chat.token_count()` gained an `include=` argument: `"new"` (default) counts just the given input, while `"complete"` estimates the total tokens for the next request, including history and system prompt where the provider supports it.

### Improvements

* `ChatGoogle()` and `ChatVertex()` now default to `gemini-3.5-flash` instead of the older `gemini-2.5-flash`.
* `ChatGroq()` now defaults to `openai/gpt-oss-20b` instead of `llama-3.1-8b-instant`.
* Built-in web search and fetch content (`ContentToolRequestSearch`/`ContentToolResponseSearch` and `ContentToolRequestFetch`/`ContentToolResponseFetch`) is now also emitted while streaming with `content="all"`, for `ChatOpenAI()`, `ChatAnthropic()`, and `ChatGoogle()`. Previously it appeared only on the completed turn, so a UI had no way to show search activity until the whole response had arrived.
* `ContentToolResponseFetch` gained a normalized `status` field (`"success"`, `"error"`, or `None` when the provider doesn't report an outcome). Providers' finer-grained reasons (Anthropic's `url_not_allowed`, Google's `PAYWALL`, …) aren't aligned across providers, so they stay available in `extra`.
* `ChatOpenAI().token_count()` now uses OpenAI's token-counting endpoint for accurate, tool-aware counts instead of a local `tiktoken` estimate.

### Changes

* MCP support now requires `mcp>=2.0.0`. The 2.0 release of the `mcp` SDK renamed its model fields (and removed `mcp.server.fastmcp.FastMCP`), so older `mcp` versions are no longer compatible. This only affects users of the optional `mcp` extra (i.e., `register_mcp_tools_*()`).
* `Turn.finish_reason` is now normalized to a consistent set of values (`"success"`, `"tool_use"`, `"max_tokens"`, `"content_filter"`, `"context_window"`, `"stop_sequence"`) across most providers, so you no longer need provider-specific logic to check why a turn ended. Previously each provider surfaced its own raw string (e.g. Anthropic's `"end_turn"`/`"tool_use"` vs. OpenAI Completions' `"stop"`/`"tool_calls"` vs. Google's `"STOP"`/`"SAFETY"`), so the same outcome could require different checks depending on which `Chat*()` you used. Reasons chatlas doesn't yet recognize still pass through unchanged.

### Bug fixes

* `ChatGoogle()` no longer errors when mixing custom tools and built-in tools (e.g. `tool_web_search()`) on Gemini 3+ models.
* Turns containing web search/fetch content can now be passed to a different provider (e.g. `ChatAnthropic().set_turns(openai_chat.get_turns())`). Previously this raised `ValueError: Unsupported content type` on `ChatOpenAI()`, and `ChatAnthropic()` forwarded the other provider's raw payload as if it were its own, producing an invalid request. Each provider now replays only the built-in tool content it produced and drops the rest.
* `ChatOpenAI()` web search `open_page` actions now surface as `ContentToolRequestFetch` (with the URL) rather than a `ContentToolRequestSearch` whose "query" was the URL, so renderers no longer show "searched for: https://…". Relatedly, a `search` action that reports only the plural `queries` field no longer falls through to the literal string `"web search"`.
* `ChatGoogle()` now records its built-in web search and URL-context work in the assistant turn, as `ContentToolRequestSearch`/`ContentToolResponseSearch` for grounded searches and `ContentToolRequestFetch`/`ContentToolResponseFetch` for fetched URLs. Previously `tool_web_search()` and `tool_web_fetch()` worked but reported nothing about what was searched or fetched, unlike `ChatAnthropic()` and `ChatOpenAI()`. Google's raw `grounding_metadata`/`url_metadata` is kept on each item's `extra`.
* `.chat_structured()` now explains itself when the response is cut short. Previously, extracting a data model large enough to hit the model's output limit failed with a bare `JSONDecodeError` pointing at a column number in the truncated JSON, giving no hint that `max_tokens` was the problem (#315). It now raises a `ValueError` naming `max_tokens` and suggesting you raise it. Responses truncated by the context window, or stopped by the provider's content filter, are reported the same way. Plain `.chat()` warns instead of erroring, since a partial response is still usable there — previously it returned truncated text with no indication anything was missing.
* Streaming two adjacent pieces of same-typed content that define no merge behavior (e.g. two tool requests) no longer raises `TypeError`. They are now appended as separate content instead.

### Breaking changes

* `ContentToolResponseSearch.urls` (a `list[str]`) has been replaced by `.sources` (a `list[WebSource]`), each carrying the result's `url` and `title`. Code reading `.urls` should switch to `[s.url for s in x.sources]`.
* The `Provider` abstract base class changed shape, which affects third-party `Provider` subclasses (not users of the built-in `Chat*()` functions): `stream_text()` was removed, and `stream_content()` both returns a `Sequence[Content]` (subsuming what `stream_text()` did) and takes a second `completion` argument holding the merged-so-far completion. Implementations needing state across chunks should read it from `completion` rather than storing it on `self`, since one provider instance is shared across forked chats.
* `Provider.token_count()`/`token_count_async()` now take a `turns: list[Turn]` argument instead of `*args: Content | str` (affects custom `Provider` subclasses only).

## [0.19.2] - 2026-07-08

### New features

* The `.app()` method now includes latest shinychat features like history, file attachments, etc.

## [0.19.1] - 2026-07-01

### New features

* Added `ChatPosit()` for chatting via the [Posit AI](https://posit.ai/) gateway. (#323)

## [0.19.0] - 2026-06-15

### New features

* chatlas is now instrumented with [OpenTelemetry](https://opentelemetry.io/) (OTel) out of the box, making it much easier to see how your app behaves in production — where time goes, how many tokens you're spending, which tools run, and where things fail. Without writing any tracing code, you get spans that capture the full structure of a conversation as one connected trace: an `invoke_agent` span over the whole chat loop, a `chat` span per model call, and an `execute_tool` span per tool invocation, with attributes (token usage, response model/ID, tool errors) that follow the [OTel GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/). Because chatlas keeps its spans active during each call, HTTP spans from provider instrumentors and any spans your own tools emit nest underneath automatically. Point it at any OTel-compatible backend (Logfire, Datadog, Honeycomb, Jaeger, …); message content is omitted by default and opt-in via `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true`. See the [monitoring guide](https://posit-dev.github.io/chatlas/get-started/monitor.html) to get started. (#310)
* `Chat` gains a `model` property to get (or set) the model after the chat is created. Setting it does not validate the model name.
* `ChatGoogle()`'s `reasoning` parameter now accepts a string thinking level (`"minimal"`, `"low"`, `"medium"`, or `"high"`) in addition to an integer token budget.
* `ChatAnthropic()`'s `reasoning` parameter now accepts a string effort level (`"low"`, `"medium"`, `"high"`, `"xhigh"`, or `"max"`) to enable Claude's adaptive thinking, in addition to an integer token budget.

### Bug fixes

* OpenAI-compatible providers (e.g., `ChatOllama()` with models like qwen3) now capture thinking content returned in a `reasoning` field, not just `reasoning_content`. Previously this thinking content was silently dropped.

## [0.18.1] - 2026-05-21

### Improvements

* `Content.tagify()` implementations (`ContentToolRequest`, `ContentToolResult`, `ContentThinking`) now annotate their return type as `htmltools.Tagified` and fully tagify their output, complying with htmltools 0.7.0's tightened Tagifiable contract. Embedding these contents inside another `.tagify()` recursion no longer trips the new boundary check in htmltools 0.7.0. (#311)

### Bug fixes

* `ContentPDF` is now exported from `chatlas.types`, matching all other `Content` subclasses. (#312)

## [0.18.0] - 2026-05-12

### New features

* New `StreamController` class for cooperative stream cancellation. Pass a controller to `.stream()` or `.stream_async()` and call `controller.cancel()` to stop the stream cleanly (e.g., from a Shiny "stop generating" button). The partial response is preserved in conversation history. (#279)

### Improvements

* `ChatAnthropic()` and `ChatBedrockAnthropic()` now use Anthropic's native structured outputs API for Claude 4.5+ models, enabling streaming with `data_model`. Older models fall back to the tool-based approach. A new `structured_output_mode` parameter (`"auto"`, `"native"`, or `"tool"`) lets you override the auto-detection. (#263)
* When a stream is interrupted (closed early, cancelled, or errors), the accumulated content is now saved as a partial `AssistantTurn` so conversation state isn't lost. Partial turns display `[interrupted]` (or the cancellation reason) in the `Chat` repr and are excluded from token/cost accounting. (#279)
* `ChatBedrockAnthropic()` now defaults to `cache="5m"`, enabling prompt caching by default — matching `ChatAnthropic()`'s behavior. (#308)
* `ChatOpenAI()` now warns when `base_url` points to a non-OpenAI host, guiding users to `ChatOpenAICompletions()` for third-party backends like vLLM, Ollama, and LiteLLM. (#285)

### Bug fixes

* Fixed thinking content being silently dropped during streaming for completions-based providers (DeepSeek, Groq, OpenRouter, etc.). The streaming path was returning finalized `ContentThinking` objects instead of `ContentThinkingDelta` fragments, which the `TurnAccumulator` didn't recognize. (#301)
* Fixed `model_dump(mode="json")` failing on `Turn`s containing `bytes` fields (e.g., `ContentPDF.data`, `thought_signature` in `ContentToolRequest`/`ContentThinking` extras). Bytes values are now base64-encoded during serialization and decoded on validation, so JSON round-trips work correctly.
* `batch_chat()`, `batch_chat_text()`, and `batch_chat_structured()` now correctly return `None` when `wait=False` and the job is still incomplete. Previously they returned `[]`, making it impossible to distinguish "all requests failed" from "job not done yet". (#306)
* `ChatDatabricks()` (and other `ChatOpenAICompletions()` providers) no longer fail with HTTP 400 when the conversation history contains empty assistant content, which can occur during tool calling. (#305)

## [0.17.0] - 2026-05-11

### New features

* `ChatOpenAICompletions()` (and providers built on it like `ChatDeepSeek`, `ChatOpenRouter`, etc.) now extracts `reasoning_content` from model responses as `ContentThinking` objects. A new `preserve_thinking` parameter controls whether reasoning content is sent back to the API in multi-turn conversations; it defaults to `False` but is set to `True` for `ChatDeepSeek` (required for V4 tool-calling) and `ChatOpenRouter` (recommended for quality). (#295)

### Improvements

* `.stream()` and `.stream_async()` now handle thinking content differently by mode. With `content="text"`, thinking is suppressed entirely. With `content="all"`, thinking fragments are yielded as `ContentThinkingDelta` objects with a `phase` property (`"start"`, `"body"`, or `"end"`) that communicates block boundaries to downstream consumers without injecting synthetic strings into the stream. (#299, #297, #294)
* Updated default models across all providers to current generation: (#292)
  * Anthropic: `claude-sonnet-4-6`
  * Bedrock: `us.anthropic.claude-sonnet-4-6`
  * Snowflake: `claude-sonnet-4-6`
  * Databricks: `databricks-claude-sonnet-4-6`
  * OpenAI / Completions / OpenRouter / Portkey: `gpt-5.4`
  * GitHub: `gpt-5`
  * Deepseek: `deepseek-v4-flash`
  * Perplexity: `sonar`
* Updated token pricing data from LiteLLM. (#292)
* `ChatBedrockAnthropic()` gains a `reasoning` parameter for extended thinking, matching the existing parameter on `ChatAnthropic()`. (#286)

## [0.16.0] - 2026-04-16

### New features

* New `ChatLMStudio()` provider for chatting with local models via [LM Studio](https://lmstudio.ai). (#280)
* The `.stream()` and `.stream_async()` methods now yield `ContentThinking` objects (instead of plain strings) for thinking/reasoning content when `content="all"`. This allows downstream packages like shinychat to provide specific UI for thinking content. (#276)
* Built-in tools (`tool_web_search()`, `tool_web_fetch()`) now include `description` and `annotations` properties, making their metadata consistent with user-defined tools created by `Tool()`. (#278)

### Bug fixes

* Fixed OpenAI streaming crash (`AttributeError: 'NoneType' object has no attribute 'output'`) caused by a new `response.rate_limits.updated` event emitted after `response.completed`. (#282)
* Fixed tool calling with Google thinking models (e.g., `gemini-3-flash-preview`) failing with a 400 `INVALID_ARGUMENT` error about a missing `thought_signature`. The signature is now preserved and forwarded in subsequent turns. (#274)
* OpenAI's `web_search_call` no longer errors on non-search action types like `open_page` and `find_in_page`. (#277)

## [0.15.2] -- 2026-02-27

### Bug fixes

* Fixed compatibility with rich >= 14.3.0 and Anthropic SDK v0.82+. (#269)


## [0.15.1] -- 2026-01-22

### New features

* `.stream()` and `.stream_async()` now support a `data_model` parameter for structured data extraction while streaming. (#262)
* `.to_solver()` now supports a `data_model` parameter for structured data extraction in evals. When provided, the solver uses `.chat_structured()` instead of `.chat()` and outputs JSON-serialized data. (#264)

### Bug fixes

* Fixed `ContentToolResult` with an `error` not being JSON serializable. When a tool call failed, calling `.get_turns()` followed by `.model_dump_json()` would raise a `PydanticSerializationError`. (#267)

## [0.15.0] - 2026-01-06

### New features

* `ChatOpenAI()`, `ChatAnthropic()`, and `ChatGoogle()` gain a new `reasoning` parameter to easily opt-into, and fully customize, reasoning capabilities. (#202, #260)
    * A new `ContentThinking` content type was added and captures the "thinking" portion of a reasoning model. (#192)
* Added "built-in" web search and URL fetch tools `tool_web_search()` and `tool_web_fetch()`:
    * `tool_web_search()` is supported by OpenAI, Claude (Anthropic), and Google (Gemini).
    * `tool_web_fetch()` is supported by Claude (requires beta header) and Google.
    * New content types `ContentToolRequestSearch`, `ContentToolResponseSearch`, `ContentToolRequestFetch`, and `ContentToolResponseFetch` capture web tool interactions.
* Added `ToolBuiltIn` class to assist with specifying provider-specific built-in tools. This enables provider-specific functionality like OpenAI's image generation to be registered and used as tools. Built-in tools pass raw provider definitions directly to the API rather than wrapping Python functions. (#214)
* `ChatOpenAI()` and `ChatAzureOpenAI()` gain a new `service_tier` parameter to request a specific service tier (e.g., `"flex"` for slower/cheaper or `"priority"` for faster/more expensive). (#204)
* `ChatAuto()` now accepts `"claude"` as an alias for `"anthropic"`, reflecting Anthropic's rebranding of developer tools under the Claude name. (#239)

### Changes

* `repr()` now generally gives the same result as `str()` for many classes (`Chat`, `Turn`, `Content`, etc). This leads to a more human-readable result (and is closer to the result that gets `echo`ed by `.chat()`). (#245)
* The `Chat.get_cost()` method's `options` parameter was renamed to `include`. (#244)
* When supplying a `model` to `.register_tool(tool_func, model=ToolModel)`, the defaults for the `model` must match the `tool_func` defaults. Previously, if `tool_func` had defaults, but `ToolModel` didn't, those defaults would get silently ignored. (#253)

### Improvements

* `Chat` and `Turn` now have a `_repr_markdown_` method and an overall improved `repr()` experience. (#245)
* `ChatSnowflake()` now sets the `application` config parameter for partner identification. Defaults to `"py_chatlas"` but can be overridden via the `SF_PARTNER` environment variable. (#209)

### Bug fixes

* Fixed structured data extraction with `ChatAnthropic()` failing for Pydantic models containing nested types (e.g., `list[NestedModel]`). The issue was that `$defs` (containing nested type definitions) was incorrectly placed inside the schema, breaking JSON `$ref` pointer references. (#100)
* Fixed MCP tools failing with OpenAI providers due to strict mode schema validation. OpenAI's strict mode rejects standard JSON Schema features like `format: "uri"` and requires all properties in the `required` array. MCP tools now set `strict=false` to use standard JSON Schema conventions. (#255)
* Fixed MCP tools not working with `ChatGoogle()`. (#257)
* Tool functions parameters that are `typing.Annotated` with a `pydantic.Field` (e.g., `def add(x: Annotated[int, Field(description="First number")])`) are now handled correctly. (#251)


## [0.14.0] - 2025-12-09

### New features

* `ChatOpenAI()` (and `ChatAzureOpenAI()`) gain access to latest models, built-in tools, etc. as a result of moving to the new [Responses API](https://platform.openai.com/docs/api-reference/responses). (#192)
* Added new family of functions (`parallel_chat()`, `parallel_chat_text()`, and `parallel_chat_structured()`) for submitting multiple prompts at once with some basic rate limiting toggles. (#188)
* Tools can now return image or PDF content types, with `content_image_file()` or `content_pdf_file()` (#231).
    * As a result, the experimental `ContentToolResultImage` and `ContentToolResultResource` were removed since this new support for generally supporting `ContentImage` and `ContentPDF` renders those content types redundant.
* Added support for systematic evaluation via [Inspect AI](https://inspect.aisi.org.uk/). This includes:
    * A new `.export_eval()` method for exporting conversation history as an Inspect eval dataset sample. This supports multi-turn conversations, tool calls, images, PDFs, and structured data.
    * A new `.to_solver()` method for translating chat instances into Inspect solvers that can be used with Inspect's evaluation framework.
    * A new `Turn.to_inspect_messages()` method for converting turns to Inspect's message format.
    * Comprehensive documentation in the [Evals guide](https://posit-dev.github.io/chatlas/misc/evals.html).
* `ChatAnthropic()` and `ChatBedrockAnthropic()` gain new `cache` parameter to control caching. For `ChatAnthropic()`, it defaults to `"5m"`, which should (on average) reduce the cost of your chats. For `ChatBedrockAnthropic()`, it defaults to `"none"`, since caching isn't guaranteed to be widely supported (#215)
* Added rudimentary support for a new `ContentThinking` type. (#192)

### Changes

* `ChatOpenAI()` (and `ChatAzureOpenAI()`) move from OpenAI's Completions API to [Responses API](https://platform.openai.com/docs/api-reference/responses). If this happens to break behavior, change `ChatOpenAI()` -> `ChatOpenAICompletions()` (or `ChatAzureOpenAI()` -> `ChatAzureOpenAICompletions()`). (#192)
* The `Turn` class is now a base class with three specialized subclasses: `UserTurn`, `AssistantTurn`, and `SystemTurn`. Use these new classes to construct turns by hand. (#224)
* The `.set_model_params()` method no longer accepts `kwargs`. Instead, use the new `chat.kwargs_chat` attribute to set chat input parameters that persist across the chat session. (#212)
* `Provider` implementations now require an additional `.value_tokens()` method. Previously, it was assumed that token info was logged and attached to the `Turn` as part of the `.value_turn()` method. The logging and attaching is now handled automatically. (#194)

### Improvements

* `ChatAnthropic()` and `ChatBedrockAnthropic()` now default to Claude Sonnet 4.5.
* `ChatGroq()` now defaults to llama-3.1-8b-instant.
* `Chat.chat()`, `Chat.stream()`, and related methods now automatically complete dangling tool requests when a chat is interrupted during a tool call loop, allowing the conversation to be resumed without causing API errors (#230).
* `content_pdf_file()` and `content_pdf_url()` now include relevant `filename` information. (#199)

### Bug fixes

* `.set_model_params()` now works correctly for `.*_async()` methods. (#198)
* `.chat_structured()` results are now included correctly into the multi-turn conversation history. (#203)
* `ChatAnthropic()` now drops empty assistant turns to avoid API errors when tools return side-effect only results. (#226)

## [0.13.2] - 2025-10-02

### Improvements

* `ContentToolResult`'s `.get_model_value()` method now calls `.to_json(orient="record")` (instead of `.to_json()`) when relevant. As a result, if a tool call returns a Pandas `DataFrame` (or similar), the model now receives a less confusing (and smaller) JSON format. (#183)

### Bug fixes

* `ChatAzureOpenAI()` and `ChatDatabricks()` now work as expected when a `OPENAI_API_KEY` environment variable isn't present. (#185)

## [0.13.1] - 2025-09-18

### Bug fixes

* `ChatGithub()` once again uses the appropriate `base_url` when generating reponses (problem introduced in v0.11.0). (#182)

## [0.13.0] - 2025-09-10

### New features

* Added support for submitting multiple chats in one batch. With batch submission, results can take up to 24 hours to complete, but in return you pay ~50% less than usual. For more, see the [reference](https://posit-dev.github.io/chatlas/reference/) for `batch_chat()`, `batch_chat_text()`, `batch_chat_structured()` and `batch_chat_completed()`. (#177)
* The `Chat` class gains new `.chat_structured()` (and `.chat_structured_async()`) methods. These methods supersede the now deprecated `.extract_data()` (and `.extract_data_async()`). The only difference is that the new methods return a `BaseModel` instance (instead of a `dict()`), leading to a better type hinting/checking experience.  (#175)
* The `.get_turns()` method gains a `tool_result_role` parameter. Set `tool_result_role="assistant"` to collect tool result content (plus the surrounding assistant turn contents) into a single assistant turn. This is convenient for display purposes and more generally if you want the tool calling loop to be contained in a single turn. (#179) 

### Improvements

* The `.app()` method now:
    * Enables bookmarking by default (i.e., chat session survives page reload). (#179)
    * Correctly renders pre-existing turns that contain tool calls. (#179)

## [0.12.0] - 2025-09-08

### Breaking changes

* `ChatAuto()`'s first (optional) positional parameter has changed from `system_prompt` to `provider_model`, and `system_prompt` is now a keyword parameter. As a result, you may need to change `ChatAuto("[system prompt]")` -> `ChatAuto(system_prompt="[system prompt]")`. In addition, the `provider` and `model` keyword arguments are now deprecated, but continue to work with a warning, as are the previous `CHATLAS_CHAT_PROVIDER` and `CHATLAS_CHAT_MODEL` environment variables. (#159)

### New features

* `ChatAuto()`'s new `provider_model` takes both provider and model in a single string in the format `"{provider}/{model}"`, e.g. `"openai/gpt-5"`. If not provided, `ChatAuto()` looks for the `CHATLAS_CHAT_PROVIDER_MODEL` environment variable, defaulting to `"openai"` if neither are provided. Unlike previous versions of `ChatAuto()`, the environment variables are now used *only if function arguments are not provided*. In other words, if `provider_model` is given, the `CHATLAS_CHAT_PROVIDER_MODEL` environment variable is ignored. Similarly, `CHATLAS_CHAT_ARGS` are only used if no `kwargs` are provided. This improves interactive use cases, makes it easier to introduce application-specific environment variables, and puts more control in the hands of the developer. (#159)
* The `.register_tool()` method now: 
  * Accepts a `Tool` instance as input. This is primarily useful for binding things like `annotations` to the `Tool` in one place, and registering it in another. (#172)
  * Supports function parameter names that start with an underscore. (#174)
* The `ToolAnnotations` type gains an `extra` key field -- providing a place for providing additional information that other consumers of tool annotations (e.g., [shinychat](https://posit-dev.github.io/shinychat/)) may make use of.

### Bug fixes

* `ChatAuto()` now supports recently added providers such as `ChatCloudflare()`, `ChatDeepseek()`, `ChatHuggingFace()`, etc. (#159)

## [0.11.1] - 2025-08-29

### New features

* `.register_tool()` gains a `name` parameter (useful for overriding the name of the function). (#162)

### Bug fixes

* `ContentToolRequest` is (once again) serializable to/from JSON via Pydantic. (#164)
* `.register_tool(model=model)` no longer unexpectedly errors when `model` contains `pydantic.Field(alias='_my_alias')`. (#161)

### Changes

* `.register_tool(annotations=annotations)` drops support for `mcp.types.ToolAnnotations()` and instead expects a dictionary of the same info. (#164)

## [0.11.0] - 2025-08-26

### New features

* The `Chat` class gains a new `.list_models()` method for obtaining a list of model ids/names, pricing info, and more. (#155)
* `Chat`'s `.register_tool()` method gains an `annotations` parameter, which is useful for describing the tool and its behavior. This information is attached to `ContentToolRequest()` and `ContentToolResult()` (via the `.request` parameter) objects when tool calls occur. To include these objects in streaming content, make sure to set `.stream(content="all")`. (#156)

### Improvements

* Tools registered via MCP (e.g., `.register_mcp_tools_http_stream_async()`) now automatically pick up on tool annotations. (#156)

### Changes

* `ChatGithub()` changed its default for `base_url` from <https://models.inference.ai.azure.com> to <https://models.github.ai/inference/>. As a result, more models are available (by default). (#155)

## [0.10.0] - 2025-08-19

### New features

* Added `ChatCloudflare()` for chatting via [Cloudflare AI](https://developers.cloudflare.com/workers-ai/get-started/rest-api/). (#150)
* Added `ChatDeepSeek()` for chatting via [DeepSeek](https://www.deepseek.com/). (#147)
* Added `ChatOpenRouter()` for chatting via [Open Router](https://openrouter.ai/). (#148)
* Added `ChatHuggingFace()` for chatting via [Hugging Face](https://huggingface.co/). (#144)
* Added `ChatMistral()` for chatting via [Mistral AI](https://mistral.ai/). (#145)
* Added `ChatPortkey()` for chatting via [Portkey AI](https://portkey.ai/). (#143)

### Changes

* `ChatAnthropic()` and `ChatBedrockAnthropic()` now default to Claude Sonnet 4.0.

### Bug fixes

* Fixed an issue where chatting with some models was leading to `KeyError: 'cached_input'`. (#149)


## [0.9.2] - 2025-08-08

### Improvements

* `Chat.get_cost()` now covers many more models and also takes cached tokens into account. (#133)
* Avoid erroring when tool calls occur with recent versions of `openai` (> v1.99.5). (#141)


## [0.9.1] - 2025-07-09

### Bug fixes

* Fixed an issue where `.chat()` wasn't streaming output properly in (the latest build of) Positron's Jupyter notebook. (#131)

* Needless warnings and errors are no longer thrown when model pricing info is unavailable. (#132)

## [0.9.0] - 2025-07-02

### New features

* `Chat` gains a handful of new methods:
    * `.register_mcp_tools_http_stream_async()` and `.register_mcp_tools_stdio_async()`: for registering tools from a [MCP server](https://modelcontextprotocol.io/). (#39)
    * `.get_tools()` and `.set_tools()`: for fine-grained control over registered tools. (#39)
    * `.set_model_params()`: for setting common LLM parameters in a model-agnostic fashion. (#127)
    * `.get_cost()`: to get the estimated cost of the chat. Only popular models are supported, but you can also supply your own token prices. (#106)
    * `.add_turn()`: to add `Turn`(s) to the current chat history. (#126)
* Tool functions passed to `.register_tool()` can now `yield` numerous results. (#39)
* A `ContentToolResultImage` content class was added for returning images from tools. It is currently only works with `ChatAnthropic`. (#39)
* A `Tool` can now be constructed from a pre-existing tool schema (via a new `__init__` method). (#39)
* The `Chat.app()` method gains a `host` parameter. (#122)
* `ChatGithub()` now supports the more standard `GITHUB_TOKEN` environment variable for storing the API key. (#123)

### Changes

#### Breaking Changes

* `Chat` constructors (`ChatOpenAI()`, `ChatAnthropic()`, etc) no longer have a `turns` keyword parameter. Use the `.set_turns()` method instead to set the (initial) chat history. (#126)
* `Chat`'s `.tokens()` methods have been removed in favor of `.get_tokens()` which returns both cumulative tokens in the turn and discrete tokens. (#106)

#### Other Changes

* `Tool`'s constructor no longer takes a function as input. Use the new `.from_func()` method instead to create a `Tool` from a function. (#39)
* `.register_tool()` now throws an exception when the tool has the same name as an already registered tool. Set the new `force` parameter to `True` to force the registration. (#39)

### Improvements

* `ChatGoogle()` and `ChatVertex()` now default to Gemini 2.5 (instead of 2.0). (#125)
* `ChatOpenAI()` and `ChatGithub()` now default to GPT 4.1 (instead of 4o). (#115)
* `ChatAnthropic()` now supports `content_image_url()`. (#112)
* HTML styling improvements for `ContentToolResult` and `ContentToolRequest`. (#39)
* `Chat`'s representation now includes cost information if it can be calculated. (#106)
* `token_usage()` includes cost if it can be calculated. (#106)

### Bug fixes

* Fixed an issue where `httpx` client customization (e.g., `ChatOpenAI(kwargs = {"http_client": httpx.Client()})`) wasn't working as expected (#108)

### Developer APIs

* The base `Provider` class now includes a `name` and `model` property. In order for them to work properly, provider implementations should pass a `name` and `model` along to the `__init__()` method. (#106)
* `Provider` implementations must implement two new abstract methods: `translate_model_params()` and `supported_model_params()`.

## [0.8.1] - 2025-05-30

* Fixed `@overload` definitions for `.stream()` and `.stream_async()`.

## [0.8.0] - 2025-05-30

### New features

* New `.on_tool_request()` and `.on_tool_result()` methods register callbacks that fire when a tool is requested or produces a result. These callbacks can be used to implement custom logging or other actions when tools are called, without modifying the tool function (#101).
* New `ToolRejectError` exception can be thrown from tool request/result callbacks or from within a tool function itself to prevent the tool from executing. Moreover, this exception will provide some context for the the LLM to know that the tool didn't produce a result because it was rejected. (#101)

### Improvements

* The `CHATLAS_LOG` environment variable now enables logs for the relevant model provider. It now also supports a level of `debug` in addition to `info`. (#97)
* `ChatSnowflake()` now supports tool calling. (#98)
* `Chat` instances can now be deep copied, which is useful for forking the chat session. (#96)

### Changes

* `ChatDatabricks()`'s `model` now defaults to `databricks-claude-3-7-sonnet` instead of `databricks-dbrx-instruct`. (#95)
* `ChatSnowflake()`'s `model` now defaults to `claude-3-7-sonnet` instead of `llama3.1-70b`. (#98)

### Bug fixes

* Fixed an issue where `ChatDatabricks()` with an Anthropic `model` wasn't handling empty-string responses gracefully. (#95)


## [0.7.1] - 2025-05-10

* Added `openai` as a hard dependency, making installation easier for a wide range of use cases. (#91)

## [0.7.0] - 2025-04-22

### New features

* Added `ChatDatabricks()`, for chatting with Databrick's [foundation models](https://docs.databricks.com/aws/en/machine-learning/model-serving/score-foundation-models). (#82)
* `.stream()` and `.stream_async()` gain a `content` argument. Set this to `"all"` to include `ContentToolResult`/`ContentToolRequest` objects in the stream. (#75)
* `ContentToolResult`/`ContentToolRequest` are now exported to `chatlas` namespace. (#75)
* `ContentToolResult`/`ContentToolRequest` gain a `.tagify()` method so they render sensibly in a Shiny app. (#75)
* A tool can now return a `ContentToolResult`. This is useful for:
    * Specifying the format used for sending the tool result to the chat model (`model_format`). (#87)
    * Custom rendering of the tool result (by overriding relevant methods in a subclass). (#75)
* `Chat` gains a new `.current_display` property. When a `.chat()` or `.stream()` is currently active, this property returns an object with a `.echo()` method (to echo new content to the display). This is primarily useful for displaying custom content during a tool call. (#79)

### Improvements

* When a tool call ends in failure, a warning is now raised and the stacktrace is printed. (#79)
* Several improvements to `ChatSnowflake()`:
  * `.extract_data()` is now supported.
  *  `async` methods are now supported. (#81)
  * Fixed an issue with more than one session being active at once. (#83)
* `ChatAnthropic()` no longer chokes after receiving an output that consists only of whitespace. (#86)
* `orjson` is now used for JSON loading and dumping. (#87)

### Changes

* The `echo` argument of the `.chat()` method defaults to a new value of `"output"`. As a result, tool requests and results are now echoed by default. To revert to the previous behavior, set `echo="text"`. (#78)
* Tool results are now dumped to JSON by default before being sent to the model. To revert to the previous behavior, have the tool return a `ContentToolResult` with `model_format="str"`. (#87)

### Breaking changes

* The `.export()` method's `include` argument has been renamed to `content` (to match `.stream()`). (#75)

## [0.6.1] - 2025-04-03

### Bug fixes

* Fixed a missing dependency on the `requests` package.

## [0.6.0] - 2025-04-01

### New features

* New `content_pdf_file()` and `content_pdf_url()` allow you to upload PDFs to supported models. (#74)

### Improvements

* `Turn` and `Content` now inherit from `pydantic.BaseModel` to provide easier saving to and loading from JSON. (#72)

## [0.5.0] - 2025-03-18

### New features

* Added a `ChatSnowflake()` class to interact with [Snowflake Cortex LLM](https://docs.snowflake.com/en/user-guide/snowflake-cortex/llm-functions). (#54)
* Added a `ChatAuto()` class, allowing for configuration of chat providers and models via environment variables. (#38, thanks @mconflitti-pbc)

### Improvements

* Updated `ChatAnthropic()`'s `model` default to `"claude-3-7-sonnet-latest"`. (#62)
* The version is now accessible as `chatlas.__version__`. (#64)
* All provider-specific `Chat` subclasses now have an associated extras in chatlas. For example, `ChatOpenAI` has `chatlas[openai]`, `ChatPerplexity` has `chatlas[perplexity]`, `ChatBedrockAnthropic` has `chatlas[bedrock-anthropic]`, and so forth for the other `Chat` classes. (#66)

### Bug fixes

* Fixed an issue with content getting duplicated when it overflows in a `Live()` console. (#71)
* Fix an issue with tool calls not working with `ChatVertex()`. (#61)


## [0.4.0] - 2025-02-19

### New features

* Added a `ChatVertex()` class to interact with Google Cloud's Vertex AI. (#50)
* Added `.app(*, echo=)` support. This allows for chatlas to change the echo behavior when running the Shiny app. (#31)

### Improvements

* Migrated `ChatGoogle()`'s underlying python SDK from `google-generative` to `google-genai`. As a result, streaming tools are now working properly. (#50)

### Bug fixes

* Fixed a bug where synchronous chat tools would not work properly when used in a `_async()` context. (#56)
* Fix broken `Chat`'s Shiny app when `.app(*, stream=True)` by using async chat tools. (#31)
* Update formatting of exported markdown to use `repr()` instead of `str()` when exporting tool call results. (#30)

## [0.3.0] - 2024-12-20

### New features

* `Chat`'s `.tokens()` method gains a `values` argument. Set it to `"discrete"` to get a result that can be summed to determine the token cost of submitting the current turns. The default (`"cumulative"`), remains the same (the result can be summed to determine the overall token cost of the conversation).
* `Chat` gains a `.token_count()` method to help estimate token cost of new input. (#23)

### Bug fixes

* `ChatOllama` no longer fails when a `OPENAI_API_KEY` environment variable is not set.
* `ChatOpenAI` now correctly includes the relevant `detail` on `ContentImageRemote()` input.
* `ChatGoogle` now correctly logs its `token_usage()`. (#23)


## [0.2.0] - 2024-12-11

First stable release of `chatlas`, see the website to learn more <https://posit-dev.github.io/chatlas/>
