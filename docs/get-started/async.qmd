---
title: Async
callout-appearance: simple
---

Important [`Chat`](../reference/Chat.qmd) methods such as `.chat()`, `.stream()`, etc., are synchronous, but are also available in an asynchronous form. Most important amongst these is [`.stream_async()`](../reference/Chat.qmd#stream_async) -- the recommended way to stream in a production environment where multiple users may be streaming at the same time.

The next article uses `.stream_async()` to build performant [chatbot apps](chatbots.qmd), but below is a more minimal example of how to it works.
Note that, in order to use async methods, you need to `await` the result inside an `async` function.

```python
import asyncio
import chatlas as ctl

chat = ctl.ChatOpenAI(
    model="gpt-4.1-mini",
    system_prompt="You are a helpful assistant.",
)

async def do_stream(prompt: str):
    stream = await chat.stream_async(prompt)
    async for chunk in stream:
        print(chunk)

asyncio.run(do_stream("My name is Chatlas."))
```

```python
Hello
Chatlas.
How
can
I
help?
```