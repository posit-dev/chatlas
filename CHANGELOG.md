# Changelog

<!--
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
-->

## [UNRELEASED]

### Improvements

* `ContentToolResult`'s `.get_model_value()` method now calls `.to_json(orient="record")` (instead of `.to_json()`) when relevant. As a result, if a tool call returns a Pandas `DataFrame`, the model will receive a more understandable data format. (#183)

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
