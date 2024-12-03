# chatlas

chatlas provides a simple and unified interface to popular large language models (LLMs) in Python. The interface is intentionally minimal and focuses on making it easy to get started and rapidly prototype, while also supporting table stakes features like streaming output, structured data extraction, function (tool) calling, images, async, and more.

(Looking for something similar to chatlas, but in R? Check out [elmer](https://elmer.tidyverse.org/)!)

## Install

`chatlas` isn't yet on pypi, but you can install from Github:

```bash
pip install git+https://github.com/posit-dev/chatlas
```

## Model providers

`chatlas` supports a variety of model providers. See the [API reference](https://posit-dev.github.io/chatlas/reference/index.html) for more details (like managing credentials) on each provider.

* Anthropic (Claude): [`ChatAnthropic()`](https://posit-dev.github.io/chatlas/reference/ChatAnthropic.html).
* GitHub model marketplace: [`ChatGithub()`](https://posit-dev.github.io/chatlas/reference/ChatGithub.html).
* Google (Gemini): [`ChatGoogle()`](https://posit-dev.github.io/chatlas/reference/ChatGoogle.html).
* Groq: [`ChatGroq()`](https://posit-dev.github.io/chatlas/reference/ChatGroq.html).
* Ollama local models: [`ChatOllama()`](https://posit-dev.github.io/chatlas/reference/ChatOllama.html).
* OpenAI: [`ChatOpenAI()`](https://posit-dev.github.io/chatlas/reference/ChatOpenAI.html).
* perplexity.ai: [`ChatPerplexity()`](https://posit-dev.github.io/chatlas/reference/ChatPerplexity.html).

It also supports the following enterprise cloud providers:

* AWS Bedrock: [`ChatBedrockAnthropic()`](https://posit-dev.github.io/chatlas/reference/ChatBedrockAnthropic.html).
* Azure OpenAI: [`ChatAzureOpenAI()`](https://posit-dev.github.io/chatlas/reference/ChatAzureOpenAI.html).


## Model choice

If you're using chatlas inside your organisation, you'll be limited to what your org allows, which is likely to be one provided by a big cloud provider (e.g. `ChatAzureOpenAI()` and `ChatBedrockAnthropic()`). If you're using chatlas for your own personal exploration, you have a lot more freedom so we have a few recommendations to help you get started:

- `ChatOpenAI()` or `ChatAnthropic()` are both good places to start. `ChatOpenAI()` defaults to **GPT-4o**, but you can use `model = "gpt-4o-mini"` for a cheaper lower-quality model, or `model = "o1-mini"` for more complex reasoning.  `ChatAnthropic()` is similarly good; it defaults to **Claude 3.5 Sonnet** which we have found to be particularly code at writing code.

- `ChatGoogle()` is great for large prompts, because it has a much larger context window than other models. It allows up to 1 million tokens, compared to Claude 3.5 Sonnet's 200k and GPT-4o's 128k.

- `ChatOllama()`, which uses [Ollama](https://ollama.com), allows you to run models on your own computer. The biggest models you can run locally aren't as good as the state of the art hosted models, but they also don't share your data and and are effectively free.

## Using chatlas

You can chat via `chatlas` in several different ways, depending on whether you are working interactively or programmatically. They all start with creating a new chat object:

```python
from chatlas import ChatOpenAI

chat = ChatOpenAI(
  model = "gpt-4o",
  system_prompt = "You are a friendly but terse assistant.",
)
```

Chat objects are stateful: they retain the context of the conversation, so each new query can build on the previous ones. This is true regardless of which of the various ways of chatting you use.

### Interactive console

From a `chat` instance, you can start an interactive, multi-turn, conversation in the console (via `.console()`) or in a browser (via `.app()`).

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

The chat app is similar to the chat console, but it runs in your browser. It's useful if you need more interactive capabilities like easy copy-paste:

```python
chat.app()
```

<div style="display:flex;justify-content:center;">
<img width="667" alt="A web app for chatting with an LLM via chatlas" src="https://github.com/user-attachments/assets/e43f60cb-3686-435a-bd11-8215cb024d2e" class="border rounded">
</div>


Keep in mind that the `chat` object retains state, so when you enter the console or app, any previous interactions with that chat object are still part of the conversation, and any interactions you have in the chat console will persist after you exit back to the Python prompt. This is true regardless of which of the various chat methods you use to submit queries.

### The `.chat()` method

For a more programmatic approach, you can use the `.chat()` method to ask a question and get a response. By default, the response prints to a [rich](https://github.com/Textualize/rich) console as it streams in:

```python
chat.chat("What preceding languages most influenced Python?")
```

```
Python was primarily influenced by ABC, with additional inspiration from C,
Modula-3, and various other languages.
```

If you want to ask a question about an image, you can pass one or more additional input arguments using `content_image_file()` and/or `content_image_url()`:

```python
from chatlas import content_image_url

chat.chat(
    content_image_url("https://www.python.org/static/img/python-logo.png"),
    "Can you explain this logo?"
)
```

```
The Python logo features two intertwined snakes in yellow and blue,
representing the Python programming language. The design symbolizes...
```

To get the full response as a string, use the built-in `str()` function. Optionally, you can also suppress the rich console output by setting `echo="none"`:

```python
response = chat.chat("Who is Posit?", echo="none")
print(str(response))
```

As we'll cover in later articles, `echo="all"` can also be useful for debugging, as it shows additional information, such as tool calls.

### The `.stream()` method

If you want to do something with the response in real-time (i.e., as it arrives in chunks), use the `.stream()` method. This method returns an iterator that yields each chunk of the response as it arrives:

```python
response = chat.stream("Who is Posit?")
for chunk in response:
    print(chunk, end="")
```

The `.stream()` method can also be useful if you're building a chatbot or other interactive applications that needs to display responses as they arrive.


### Conversation history

Remember that regardless of how we interact with the model, the `chat` instance retains the conversation history, which you can access at any time:

```python
chat.turns()
```

Each turn represents a either a user's input or a model's response. It holds all the avaliable information about content and metadata of the turn. This can be useful for debugging, logging, or for building more complex conversational interfaces.

For cost and efficiency reasons, you may want to alter the conversation history. Currently, the main way to do this is to `.set_turns()`:

```python
# Remove all but the last two turns
chat.set_turns(chat.turns()[-2:])
```

### Learn more

If you're new to world LLMs, you might want to read the [Get Started](https://posit-dev.github.io/chatlas/get-started.html) guide, which covers some basic concepts and terminology.

Once you're comfortable with the basics, you can explore more advanced topics:

* [Customize the system prompt](https://posit-dev.github.io/chatlas/prompt-design.html)
* [Extract structured data](https://posit-dev.github.io/chatlas/structured-data.html)
* [Tool (function) calling](https://posit-dev.github.io/chatlas/tool-calling.html)
* [Build a web chat app](https://posit-dev.github.io/chatlas/web-apps.html)

The [API reference](https://posit-dev.github.io/chatlas/reference/index.html) is also a useful overview of all the tooling available in `chatlas`, including starting examples and detailed descriptions.
