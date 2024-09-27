# chatlas

A simple and consistent interface for chatting with various LLMs from Ollama, Anthropic, OpenAI, and others.
Working directly with the Python packages from these LLM providers is suprisingly complicated since they support such a wide range of use cases. 
Projects like LangChain and LiteLLM help in this regard, but even they can be a bit much for simple use cases.
`chatlas` provides an even simpler interface for chatting with multiple LLMs, while still supporting important capabilities like streaming, tool calling, and async.

TODO: gif of chatlas in action

## Install

`chatlas` isn't yet on pypi, but you can install from Github:

```
pip install git+https://github.com/posit-dev/chatlas
```

## Getting started

To start chatting with an LLM, you'll first need to choose a provider. 
Options like [Anthropic](#anthropic) and [OpenAI](#openai) require an account and API key to use, and also send your input to a remote server for processing.
[Ollama](#ollama), on the other hand, is a local model that can run on your own machine for free.

### Ollama

To use Ollama, first download and run the [Ollama](https://ollama.com/) executable. Then [choose a model](https://ollama.com/library), like llama 3.2, to download (from the command line):

```shell
ollama run llama-3.2
```

You'll also want the Python package:

```shell
pip install olama
```

Now, you're read to chat via `chatlas`:

```python
import chatlas
llm = chatlas.Ollama(model="llama-3.2")
llm.chat("What is 1+1?")
```

### Anthropic

To use Anthropic's models (i.e., Claude), you'll need to sign up for an account and [get an API key](https://docs.anthropic.com/en/api/getting-started).
You'll also want the Python package:

```shell
pip install anthropic
```

Now, simply paste your API key into the `chatlas.Anthropic` constructor (consider securely [managing your credentials](#managing-credentials) if sharing your code), and start chatting!

```python
import chatlas
llm = chatlas.Anthropic(api_key="...")
llm.chat("What is 1+1?")
```


### OpenAI

To use OpenAI's models (i.e., GPT), you'll need to sign up for an account and [get an API key](https://platform.openai.com/docs/quickstart).
You'll also want the Python package:

```shell
pip install openai
```

Now, simply paste your API key into the `chatlas.OpenAI` constructor (consider securely [managing your credentials](#managing-credentials) if sharing your code), and start chatting!

```python
import chatlas
llm = chatlas.OpenAI(api_key="...")
llm.chat("What is 1+1?")
```


### Google

To use Google's models (i.e., Gemini), you'll need to sign up for an account and [get an API key](https://ai.google.dev/gemini-api/docs/get-started/tutorial?lang=python).
You'll also want the Python package:

```shell
pip install google-generativeai
```

Now, simply paste your API key into the `chatlas.Google` constructor (consider securely [managing your credentials](#managing-credentials) if sharing your code), and start chatting!

```python
import chatlas
llm = chatlas.Google(api_key="...")
llm.chat("What is 1+1?")
```

## Managing credentials

Pasting an API key into `chatlas` constructor (e.g., `chatlas.OpenAI(api_key="...")`) is the simplest way to get started, and is fine for interactive use, but it's not OK for code that may be shared with others.
Instead, consider using environment variables or a configuration file to manage your credentials.
One popular way to manage credentials is to use a `.env` file to store your credentials, and then use the `python-dotenv` package to load them into your environment.

```shell
pip install python-dotenv
```

```shell
# .env
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
```

```python
import chatlas
from dotenv import load_dotenv

load_dotenv()
llm = chatlas.Anthropic()
llm.chat("What is 1+1?")
```

Another option, which makes interactive use easier, is to load your environment variables into the shell before starting Python (maybe in a `.bashrc`, `.zshrc`, etc. file):

```shell
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
```


## Tool (function) calling

Tool calling is a powerful feature enabling the LLM to call external programs to help answer your questions.
`chatlas` makes it easy to provide Python functions as tools for the LLM to call, and handles the communication between the LLM and your function.

To provide a tool, just define function(s) and pass them to the `chatlas` constructor.
Make sure to annotate your function with types to help the LLM understand what it should expect and return.
Also provide a docstring to help the LLM understand what your function does.

```python
import chatlas

def get_current_weather(location: str, unit: str = "fahrenheit") -> int:
    """Get the current weather in a location."""
    if "boston" in location.lower():
        return 12 if unit == "fahrenheit" else -11
    elif "new york" in location.lower():
        return 20 if unit == "fahrenheit" else -6
    else:
        return 72 if unit == "fahrenheit" else 22

llm = chatlas.OpenAI(tools=[get_current_weather])
llm.chat("What's the weather like in Boston, New York, and London today?")
```


## Program with chatlas

So far, we've been using the `chat()` method to interact with LLMs, which is designed for interactive use at the Python console.
Instead of `chat()`, you can use `response_generator()` to get an async generator that yields strings, allowing you to send the LLM's response to some other destination.
For example, let's write a simple program that writes the LLM's response to a file:

```python
import asyncio
import chatlas
llm = chatlas.Anthropic()
response = llm.response_generator("What is 1+1?")

async def main():
    async for chunk in response:
        with open("response.txt", "a") as f:
            f.write(chunk)

asyncio.run(main())
```

## Shiny (web) application

`chatlas` can be used inside a [Shiny](https://shiny.posit.co/py/) app to provide a chat interface to your LLM.

```python
import chatlas
from shiny.express import ui

llm = chatlas.Anthropic()

chat = ui.Chat(id = "chat")

chat.ui()

@chat.on_user_submit
def on_user_submit(input):
    response = llm.response_generator(input)
    chat.append_message_stream(response)
```