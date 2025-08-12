# chatlas <a href="https://posit-dev.github.io/chatlas"><img src="docs/logos/hex/logo.png" align="right" height="138" alt="chatlas website" /></a>

<p>
<!-- badges start -->
<a href="https://pypi.org/project/chatlas/"><img alt="PyPI" src="https://img.shields.io/pypi/v/chatlas?logo=python&logoColor=white&color=orange"></a>
<a href="https://choosealicense.com/licenses/mit/"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License"></a>
<a href="https://pypi.org/project/chatlas"><img src="https://img.shields.io/pypi/pyversions/chatlas.svg" alt="versions"></a>
<a href="https://github.com/posit-dev/chatlas"><img src="https://github.com/posit-dev/chatlas/actions/workflows/test.yml/badge.svg?branch=main" alt="Python Tests"></a>
<!-- badges end -->
</p>

Your friendly guide to building LLM chat apps in Python with less effort and more clarity.

## Install

Install the latest stable release [from PyPI](https://pypi.org/project/chatlas/):

```bash
pip install -U chatlas
```

Or, install the latest development version from GitHub:

```bash
pip install -U git+https://github.com/posit-dev/chatlas
```

## Quick start

Get started in 3 simple steps:

1. Choose a model provider, such as [ChatOpenAI](https://posit-dev.github.io/chatlas/reference/ChatOpenAI.html) or [ChatAnthropic](https://posit-dev.github.io/chatlas/reference/ChatAnthropic.html).
2. Visit the provider's [reference](https://posit-dev.github.io/chatlas/reference) page to get setup with necessary credentials.
3. Create the relevant `Chat` client and start chatting!

```python
from chatlas import ChatOpenAI

# Optional (but recommended) model and system_prompt
chat = ChatOpenAI(
    model="gpt-4.1-mini",
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


<img src="docs/images/chatlas-hello.png" alt="Model response output to the user query: 'How's the weather in San Francisco?'" width="67%" style="display: block; margin-left: auto; margin-right: auto">


Learn more at <https://posit-dev.github.io/chatlas>