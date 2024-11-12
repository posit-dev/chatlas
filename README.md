# chatlas

Easily chat with various LLM models from Ollama, Anthropic, OpenAI, and more.
`chatlas` is intentionally minimal -- making it easy to get started, while also supporting important features like streaming, tool calling, images, async, and more.

## Install

`chatlas` isn't yet on pypi, but you can install from Github:

```bash
pip install git+https://github.com/cpsievert/chatlas
```

After installing, you'll want to pick a [model provider](#model-providers), and get [credentials](#managing-credentials) set up (if necessary). Here, we demonstrate usage with OpenAI, but the concepts here apply to other implementations as well.

## Model providers

`chatlas` supports a variety of model providers:

* Anthropic (Claude): [`ChatAnthropic()`](https://cpsievert.github.io/chatlas/reference/ChatAnthropic.html).
* GitHub model marketplace: [`ChatGithub()`](https://cpsievert.github.io/chatlas/reference/ChatGithub.html).
* Google (Gemini): [`ChatGoogle()`](https://cpsievert.github.io/chatlas/reference/ChatGoogle.html).
* Groq: [`ChatGroq()`](https://cpsievert.github.io/chatlas/reference/ChatGroq.html).
* Ollama local models: [`ChatOllama()`](https://cpsievert.github.io/chatlas/reference/ChatOllama.html).
* OpenAI: [`ChatOpenAI()`](https://cpsievert.github.io/chatlas/reference/ChatOpenAI.html).
* perplexity.ai: [`ChatPerplexity()`](https://cpsievert.github.io/chatlas/reference/ChatPerplexity.html).

As well as enterprise cloud providers:

* AWS Bedrock: [`ChatBedrockAnthropic()`](https://cpsievert.github.io/chatlas/reference/ChatBedrockAnthropic.html).
* Azure OpenAI: [`ChatAzureOpenAI()`](https://cpsievert.github.io/chatlas/reference/ChatAzureOpenAI.html).


## Model choice

If you're using chatlas inside your organisation, you'll typically need to use whatever you're allowed to. If you're using chatlas for your own personal exploration, we recommend starting with:

`ChatOpenAI()`, which currently defaults to `model="gpt-4o-mini"`. You might want to try `model="gpt-4o"` for more demanding task and if you want to force complex reasoning, `model="o1-mini"`.

`ChatAnthropic()`, which defaults to Claude 3.5 Sonnet. This currently appears to be the best model for code generation.

If you want to put a lot of data in the prompt, try `ChatGoogle()` which defaults to Gemini 1.5 Flash and supports 1 million tokens, compared to 200k for Claude 3.5 Sonnet and 128k for GPT 4o mini.

## Using chatlas

You can chat via `chatlas` in several different ways, depending on whether you are working interactively or programmatically. They all start with creating a new chat object:

```python
from chatlas import ChatOpenAI

chat = ChatOpenAI(
  model = "gpt-4o-mini",
  system_prompt = "You are a friendly but terse assistant.",
)
```

Chat objects are stateful: they retain the context of the conversation, so each new query can build on the previous ones. This is true regardless of which of the various ways of chatting you use.

### Interactive console

The most interactive, least programmatic way of using `chatlas` is to chat with it directly in your console with `chat.console()` or in your browser with `chat.app()`.

```python
chat.console()
```

```
Entering chat console. Press Ctrl+C to quit.

?> Who created Python?

Python was created by Guido van Rossum. He began development in the late 1980s and released the first     
version in 1991. 

?> Where did he develop it?

Guido van Rossum developed Python while working at Centrum Wiskunde & Informatica (CWI) in the            
Netherlands.     
```

The chat console is useful for quickly exploring the capabilities of the model, especially when you've customized the chat object with tool integrations (covered later).

The chat app is similar to the chat console, but it runs in your browser. It's useful if need more interactive capabilities like easy copy-paste.

```python
chat.app()
```

<div align="center">
<img width="667" alt="Screenshot 2024-11-06 at 11 43 34 AM" src="https://github.com/user-attachments/assets/e43f60cb-3686-435a-bd11-8215cb024d2e">
</div>


Again, keep in mind that the chat object retains state, so when you enter the chat console, any previous interactions with that chat object are still part of the conversation, and any interactions you have in the chat console will persist even after you exit back to the Python prompt.


### Interactive chat

The second most interactive way to chat is by calling the `chat()` method (from the normal Python prompt or in a script):

```python
_ = chat.chat("What preceding languages most influenced Python?")
```

```
Python was primarily influenced by ABC, with additional inspiration from C,
Modula-3, and various other languages.
```

Since `chat()` is designed for interactive use, it prints the response to the console as it arrives. Once the response is complete, it returns the response as a string, so to avoid printing it twice, assign the result to a variable.

### Programmatic chat

If you want to do something else with the response as it arrives, use the `submit()` method. It returns a [generator](https://wiki.python.org/moin/Generators) that yields strings as they arrive from the model (in small chunks, when `stream=True`). This is useful when you want to process the response as it arrives in a memory-efficient way.

```python
response = chat.submit("What is 1+1?")
for x in response:
    print(x, end="")
```

```
1 + 1 equals 2.
```

The `.submit()` method defaults to `stream=True` (meaning the response is streamed in small chunks), but you can set `stream=False` to get the entire response at once. In this case, you still get a generator, but it yields the entire response at once. This is primarily useful as workaround: some models happen to not support certain features (like tools) when streaming. Also, more generally, sometimes it's useful to have response before displaying anything.

```python
response = chat.submit("What is 1+1?", stream=False)
for x in response:
    print(x)
```

```
1 + 1 equals 2.
```

For a more compelling example, note that you can pass the result of `.submit()` directly to Shiny's [`ui.Chat` component](https://shiny.posit.co/py/components/display-messages/chat/) to create a chat interface in your own [Shiny](https://shiny.rstudio.com/py) app.

```python
from chatlas import ChatAnthropic
from shiny import ui

chat = ui.Chat(
  id="chat", 
  messages=["Hi! How can I help you today?"],
)

llm = ChatAnthropic()

@chat.on_user_submit
def _():
    response = llm.submit(chat.user_input())
    chat.append_message_stream(response)
```

### Vision (Image Input)

To ask questions about images, you can pass one or more additional input arguments using `content_image_file()` and/or `content_image_url()`:

```python
from chatlas import content_image_url

_ = chat.chat(
    content_image_url("https://www.python.org/static/img/python-logo.png"),
    "Can you explain this logo?"
)
```

```
The Python logo features two intertwined snakes in yellow and blue,
representing the Python programming language. The design symbolizes...
```

The `content_image_url()` function takes a URL to an image file and sends that URL directly to the API. The `content_image_file()` function takes a path to a local image file and encodes it as a base64 string to send to the API. Note that by default, `content_image_file()` automatically resizes the image to fit within 512x512 pixels; set the `resize` parameter to "high" if higher resolution is needed.


## Managing credentials

Pasting an API key into a chat constructor (e.g., `ChatOpenAI(api_key="...")`) is the simplest way to get started, and is fine for interactive use, but is problematic for code that may be shared with others.
Instead, consider using environment variables or a configuration file to manage your credentials.
One popular way to manage credentials is to use a `.env` file to store your credentials, and then use the `python-dotenv` package to load them into your environment.

```shell
pip install python-dotenv
```

```shell
# .env
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GITHUB_PAT=...
GOOGLE_API_KEY=...
GROQ_API_KEY=...
PERPLEXITY_API_KEY=...
```

```python
from chatlas import Anthropic
from dotenv import load_dotenv

load_dotenv()
chat = ChatAnthropic()
chat.console()
```

Another, more general, solution is to load your environment variables into the shell before starting Python (maybe in a `.bashrc`, `.zshrc`, etc. file):

```shell
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...
```
