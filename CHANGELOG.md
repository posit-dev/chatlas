# Changelog

<!--
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
-->

## [Development]

### New features

* Added `.app(*, echo=)` support. This allows for chatlas to change the echo behavior when running the Shiny app. (#31)
* Added support for `Chat` providers to be [MCP clients](https://modelcontextprotocol.io/) and register with remote tools over SSE or stdio.

### Bug fixes

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
