---
title: Streams
callout-appearance: simple
---

The [`.stream()`](../reference/Chat.qmd#stream) method returns a [generator](https://stackoverflow.com/questions/1756096/understanding-generators-in-python) that yields the model's response one chunk at a time.
This makes it a better choice than [`.chat()`](../reference/Chat.qmd#chat) for programming bespoke experiences such as [chatbot](chatbots.qmd) apps, where some other framework is responsible for consuming and displaying the response.


```python
import chatlas as ctl

chat = ctl.ChatOpenAI(
    model="gpt-4.1-mini",
    system_prompt="You are a helpful assistant.",
)

stream = chat.stream("My name is Chatlas.")
for chunk in stream:
    print(chunk)
```

```
Hello 
Chatlas.
How 
can 
I 
help?
```


Just like `.chat()`, once the model is done responding (i.e., the generator is exhausted), the user and assistant turns are stored on the `Chat` instance. That means, you can [save](chat.qmd#save-history) and [manage](chat.qmd#manage-history) them same as you would with `.chat()`.

```python
len(chat.get_turns())
```

```
2
```


::: callout-tip
`.stream()` accepts [multi-modal input](chat.qmd#multi-modal-input), just like `.chat()`.
:::

## Content types


`.stream()` also provides access to rich content types beyond just text.
To gain access to these, set the `content` parameter to `"all"`.
If the response includes things like tool calls, the stream will yield the relevant content types as they are generated.
As we'll learn later, this can be useful for [displaying tool calls](../tool-calling/displays.qmd) in something like a [chatbot](chatbots.qmd) app.

```python
def get_current_weather(lat: float, lng: float):
    "Get the current weather for a given location."
    return "sunny"

chat.register_tool(get_current_weather)

stream = chat.stream(
  "How's the weather in San Francisco?", 
  content="all"
)
for chunk in stream:
   print(type(chunk))
```


```
<class 'chatlas._content.ContentToolRequest'>
<class 'chatlas._content.ContentToolResult'>
<class 'str'>
<class 'str'>
<class 'str'>
<class 'str'>
```

## Wrapping generators

Sometimes it's useful to wrap the `.stream()` generator up into another generator function.

This is useful for adding additional functionality, such as:

* Adding a delay between chunks
* Filtering out certain content types
* Adding custom formatting
* etc.


```python
import time

def stream_with_delay(prompt: str, delay: float = 0.5):
    """
    Stream the model's response with a delay between chunks.
    """
    stream = chat.stream(prompt)
    for chunk in stream:
        time.sleep(delay)
        yield chunk

for chunk in stream_with_delay("How's the weather in San Francisco?"):
    print(chunk)
```

```
The current weather in San Francisco is sunny.
```
