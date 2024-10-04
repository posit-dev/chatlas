# chatlas

Easily chat with various LLM models from Ollama, Anthropic, OpenAI, and more.
`chatlas` is intentionally minimal, making it easy to get started, while also supporting advanced features like tool calling, streaming, and async. 
It also provides an interactive chat console, web app, and more general programmatic extension points.

## Install

`chatlas` isn't yet on pypi, but you can install from Github:

```
pip install git+https://github.com/posit-dev/chatlas
```

## Get started

To start, you'll need to create an instance of a particular `Chat` [implementation](#chat-implementations).
For example, to use [Ollama](#ollama), first create the `OllamaChat` object:

```python
from chatlas import OllamaChat
chat = OllamaChat(model="llama3.2")
```

Then, you start chatting by calling the `.chat()` method:

```python
chat.chat("Show me a short Python snippet")
```

Or, better yet, for multi-turn conversations, start a Python `.console()`:

```python
chat.console()
```

<div align="center">
<img width="600" alt="Using chatlas at the Python console" src="https://github.com/user-attachments/assets/df10b558-867a-407e-affe-3aa80cc18bb7">
</div>


And, for a better copy/paste and browsing experience, consider using the `.app()` method to launch a [Shiny](https://shiny.posit.co/py/) web app:

```python
chat.app()
```

<div align="center">
<img width="600" alt="Using chatlas in a Shiny web app" src="https://github.com/user-attachments/assets/be4c1328-d7ff-49a9-9ac4-d1cf9a68b80d">
</div>


Also, at any point, you can access the chat history via `.messages()`:

```python
chat.messages()
```

See the [advanced features](#advanced-features) section below for more involved features like tool calling, async, and streaming.


## Chat implementations

`chatlas` supports various LLM models from Ollama, Anthropic, OpenAI, and Google out of the box.
Options like `AnthropicChat` and `OpenAIChat` require an account and API key to use, and also send your input to a remote server for response generation.
[Ollama](#ollama), on the other hand, provides a way to run open source models that run locally on your own machine, so is a good option for privacy and cost reasons.

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
from chatlas import OllamaChat
chat = OllamaChat(model="llama3.2")
chat.console()
```

### Anthropic

To use Anthropic's models (i.e., Claude), you'll need to sign up for an account and [get an API key](https://docs.anthropic.com/en/api/getting-started).
You'll also want the Python package:

```shell
pip install anthropic
```

Paste your API key into the `AnthropicChat()` constructor to start chatting, but also consider securely [managing your credentials](#managing-credentials):

```python
from chatlas import AnthropicChat
chat = AnthropicChat(api_key="...")
chat.console()
```


### OpenAI

To use OpenAI's models (i.e., GPT), you'll need to sign up for an account and [get an API key](https://platform.openai.com/docs/quickstart).
You'll also want the Python package:

```shell
pip install openai
```

Paste your API key into the `OpenAIChat()` constructor to start chatting, but also consider securely [managing your credentials](#managing-credentials):

```python
from chatlas import OpenAIChat
chat = OpenAIChat(api_key="...")
chat.console()
```


### Google

To use Google's models (i.e., Gemini), you'll need to sign up for an account and [get an API key](https://ai.google.dev/gemini-api/docs/get-started/tutorial?lang=python).
You'll also want the Python package:

```shell
pip install google-generativeai
```

Paste your API key into the `GoogleChat()` constructor to start chatting, but also consider securely [managing your credentials](#managing-credentials):

```python
from chatlas import GoogleChat
chat = GoogleChat(api_key="...")
chat.console()
```

### Role your own

You can also use your own models by implementing the `BaseChat` interface.
This is a good option if you're motivated to add support for model(s) not yet supported by `chatlas`.
At a minimum, you'll need to implement the `response_generator()` and `messages()` methods.

```python
from chatlas import BaseChat

class MyCustomChat(BaseChat):
    async def response_generator(self, user_input: str):
        ...

    def messages(self):
        ...
```


## Managing credentials

Pasting an API key into a chat constructor (e.g., `OpenAIChat(api_key="...")`) is the simplest way to get started, and is fine for interactive use, but is problematic for code that may be shared with others.
Instead, consider using environment variables or a configuration file to manage your credentials.
One popular way to manage credentials is to use a `.env` file to store your credentials, and then use the `python-dotenv` package to load them into your environment.

```shell
pip install python-dotenv
```

```shell
# .env
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
```

```python
from chatlas import Anthropic
from dotenv import load_dotenv

load_dotenv()
chat = AnthropicChat()
chat.console()
```

Another option, which makes interactive use easier, is to load your environment variables into the shell before starting Python (maybe in a `.bashrc`, `.zshrc`, etc. file):

```shell
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...
```

## Advanced features

### Tool (function) calling

Tool calling is a powerful feature enabling the LLM to call external programs to help answer your questions.
`chatlas` makes it easy to provide Python functions as tools for the LLM to call, and handles the communication between the LLM and your function.

To provide a tool, just define function(s) and pass them to the `chatlas` constructor.
Make sure to annotate your function with types to help the LLM understand what it should expect and return.
Also provide a docstring to help the LLM understand what your function does.

```python
from chatlas import AnthropicChat

def get_current_weather(location: str, unit: str = "fahrenheit") -> int:
    """Get the current weather in a location."""
    if "boston" in location.lower():
        return 12 if unit == "fahrenheit" else -11
    elif "new york" in location.lower():
        return 20 if unit == "fahrenheit" else -6
    else:
        return 72 if unit == "fahrenheit" else 22

chat = AnthropicChat(tools=[get_current_weather])
chat.chat("What's the weather like in Boston, New York, and London today?")
```


### Response generators

For more control over what happens with the LLM's response, use the `response_generator()` method.
This method returns an [async generator](https://superfastpython.com/asynchronous-generators-in-python/) that yields strings, allowing you to send the LLM's response to some other destination.
For example, here's some code to write the LLM's response to a (temporary) file:

```python
import asyncio
from chatlas import AnthropicChat
import tempfile
chat = AnthropicChat()
response = chat.response_generator("What is 1+1?")

async def main():
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        async for line in response:
            f.write(line + "\n")
        
        temp_file_name = f.name

    with open(temp_file_name) as f:
        print(f.read())

asyncio.run(main())
```


### Build your own Shiny app

Pass user input from a Shiny `Chat()` component to a `chatlas` response generator to embed a chat interface in your own Shiny app.

```python
from chatlas import AnthropicChat
from shiny import ui

chat = ui.Chat(
  id="chat", 
  messages=["Hi! How can I help you today?"],
)

llm = AnthropicChat()

@chat.on_user_submit
def _(message):
    response = llm.response_generator(message)
    chat.append_message_stream(response)
```