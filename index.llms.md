# Overview

![chatlas website banner image](./logos/hero/hero.png)

chatlas

Your friendly guide to building LLM chat apps in Python with less effort and more clarity.

[![PyPI](https://img.shields.io/pypi/v/chatlas?logo=python&logoColor=white&color=orange)](https://pypi.org/project/chatlas/) [![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://choosealicense.com/licenses/mit/) [![versions](https://img.shields.io/pypi/pyversions/chatlas.svg)](https://pypi.org/project/chatlas) [![Python Tests](https://github.com/posit-dev/chatlas/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/posit-dev/chatlas)

## Quick start

Get started in 3 simple steps:

1.  Choose a model provider, such as [ChatOpenAI](https://posit-dev.github.io/chatlas/reference/ChatOpenAI.html) or [ChatAnthropic](https://posit-dev.github.io/chatlas/reference/ChatAnthropic.html).
2.  Visit the provider’s [reference](reference/index.llms.md) page to get setup with necessary credentials.
3.  Create the relevant `Chat` client and start chatting!

``` python
from chatlas import ChatOpenAI

# Optional (but recommended) model and system_prompt
chat = ChatOpenAI(
    model="gpt-5.4",
    system_prompt="You are a helpful assistant.",
)

# Optional tool registration
def get_current_weather(lat: float, lng: float):
    "Get the current weather for a given location."
    return "sunny"

chat.register_tool(get_current_weather)

# Send user prompt to the model for a response.
chat.chat("How's the weather in San Francisco?")
```

``` python
# 🛠️ tool request
get_current_weather(37.7749, -122.4194)
```

    # ✅ tool result
    sunny

The current weather in San Francisco is sunny.

## Install

Install the latest stable release [from PyPI](https://pypi.org/project/chatlas/):

``` bash
pip install -U chatlas
```

## Why chatlas?

🚀 **Opinionated design**: most problems just need the right [model](get-started/models.llms.md), [system prompt](get-started/system-prompt.llms.md), and [tool calls](get-started/tools.llms.md). Spend more time mastering the fundamentals and less time navigating needless complexity.

🧩 [**Model agnostic**](get-started/models.llms.md): try different models with minimal code changes.

🌊 [**Stream output**](get-started/chat.llms.md): automatically in notebooks, at the console, and your favorite IDE. You can also [stream](get-started/stream.llms.md) responses into bespoke applications (e.g., [chatbots](get-started/chatbots.llms.md)).

🛠️ [**Tool calling**](get-started/tools.llms.md): give the LLM “agentic” capabilities by simply writing Python function(s).

🔄 [**Multi-turn chat**](get-started/chat.llms.md#chat-history): history is retained by default, making the common case easy.

🖼️ [**Multi-modal input**](get-started/chat.llms.md#multi-modal-input): submit input like images, pdfs, and more.

📂 [**Structured output**](get-started/structured-data.llms.md): easily extract structure from unstructured input.

⏱️ [**Async**](get-started/async.llms.md): supports async operations for efficiency and scale.

✏️ [**Autocomplete**](get-started/models.llms.md#auto-complete): easily discover and use provider-specific [parameters](get-started/parameters.llms.md) like `temperature`, `max_tokens`, and more.

🔍 **Inspectable**: tools for [debugging](get-started/debug.llms.md) and [monitoring](get-started/monitor.llms.md) in production.

🔌 **Extensible**: add new [model providers](reference/Provider.llms.md), [content types](reference/types.Content.llms.md), and more.

## Next steps

Next we’ll learn more about what [model providers](get-started/models.llms.md) are available and how to approach picking a particular model. If you already have a model in mind, or just want to see what chatlas can do, skip ahead to [hello chat](get-started/chat.llms.md).
