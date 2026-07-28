# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Architecture

### Typing Best Practices

This project prioritizes strong typing that leverages provider SDK types directly:

- **Use provider SDK types**: Import and use types from `openai.types`, `anthropic.types`, `google.genai.types`, etc. rather than creating custom TypedDicts or dataclasses that mirror them. This ensures compatibility with SDK updates and provides better IDE support.
- **Use `@overload` for provider-specific returns**: When a method returns different types based on a provider argument, use `@overload` with `Literal` types to give callers precise return type information.
- **Explore SDK types interactively**: Use `python -c "from <sdk>.types import <Type>; print(<Type>.__annotations__)"` to inspect available fields and nested types when implementing provider-specific features.

## Pointers to lazily-loaded context

- Testing and VCR cassette workflow: `tests/CLAUDE.md`
- Building and publishing the docs: `docs/CLAUDE.md`
- Adding a new LLM provider: the `adding-a-provider` skill

## Connections to ellmer

This project is the Python equivalent of the R package ellmer. The source code for ellmer is available in a sibling directory to this project. Before implementing new features or bug fixes in chatlas, it may be useful to consult  the ellmer codebase to: (1) check whether the feature/fix already exists on the R side and (2) make sure the projects are aligned in terms of stylistic approaches. Note also that ellmer itself has a CLAUDE.md file which has a useful overview of the project. 

## Differences from ellmer

One important difference to note is that in `ellmer::chat_openai()` uses the completions API, while `chatlas.ChatOpenAI()` uses the responses API. Look to `ellmer::chat_openai_responses()` and `chatlas.ChatOpenAICompletions()` for the "non-default" API.

## Access to gh CLI

If ellmer or chatlas issues or PRs are referenced, try using the `gh` CLI tool to gain necessary context. They live at `tidyverse/ellmer` and `posit-dev/chatlas`