# Changelog

<!--
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
-->

## [UNRELEASED]

### Breaking changes

* The `token_usage()` and `tokens_reset()` functions have been removed. Use the new `.token_usage()` method on the `Chat` instance instead. (#23)
* The `.tokens()` method on the `Chat` instance was removed because you usually only care about `.token_usage()`. If you do indeed want the input/output tokens for each turn, you can `.get_turns()` on the chat instance, and then get the `.tokens` of each turn. (#23)

### New features

* The `Chat` class gains a `.token_count()` method to help estimate input tokens before sending it to the LLM. (#23)

### Bug fixes

* `ChatOllama` no longer fails when a `OPENAI_API_KEY` environment variable is not set.
* `ChatOpenAI` now correctly includes the relevant `detail` on `ContentImageRemote()` input.


## [0.2.0] - 2024-12-11

First stable release of `chatlas`, see the website to learn more <https://posit-dev.github.io/chatlas/> 
