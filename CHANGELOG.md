# Changelog

<!--
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
-->

## [UNRELEASED]

### New features

* Added `ChatDatabricks()`, for chatting with Databrick's [foundation models](https://docs.databricks.com/aws/en/machine-learning/model-serving/score-foundation-models). (#82)
* `.stream()` and `.stream_async()` gain a `content` argument. Set this to `"all"` to include `ContentToolRequest` and `ContentToolResponse` instances in the stream. (#75)
* `ContentToolRequest` and `ContentToolResponse` are now exported to `chatlas` namespace. (#75)
* `ContentToolRequest` and `ContentToolResponse` now have `.tagify()` methods, making it so they can render automatically in a Shiny chatbot. (#75)
* `ContentToolResult` instances can be returned from tools. This allows for custom rendering of the tool result. (#75)
* `Chat` gains a new `.current_display` property. When a `.chat()` or `.stream()` is currently active, this property returns an object with a `.echo()` method (to echo new content to the display). This is primarily useful for displaying custom content during a tool call. (#79)

### Improvements

* When a tool call ends in failure, a warning is now raised and the stacktrace is printed. (#79)
* Several improvements to `ChatSnowflake()`:
  * `.extract_data()` is now supported.
  *  `async` methods are now supported. (#81)
  * Fixed an issue with more than one session being active at once. (#83)

### Changes

* The `echo` argument of the `.chat()` method defaults to a new value of `"output"`. As a result, tool requests and results are now echoed by default. To revert to the previous behavior, set `echo="text"`. (#78)

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
